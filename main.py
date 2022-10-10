import os
import pprint
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import time
from prettytable import PrettyTable

from core.data_util import save_json
from core.runner_utils import set_th_config, filter_checkpoints, get_last_checkpoint
from core.config import config, update_config
from core.meters import AverageMeter, MultiItemAverageMeter
from core.optim import build_optimizer_and_scheduler
import datasets
import models


def parse_args():
    parser = argparse.ArgumentParser(description="Train localization network")

    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument("--gpus", help="gpus", type=str)
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="print progress bar"
    )
    parser.add_argument("--tag", help="tags shown in log", type=str)
    parser.add_argument(
        "--mode", default="train", help="train, test, test_train", type=str
    )
    parser.add_argument("--query_layers", help="query_lstm_num_layers", type=int)
    parser.add_argument("--video_layers", help="video_lstm_num_layers", type=int)
    parser.add_argument(
        "--num_heads", help="head of multi-head self_attention_layer", type=int
    )
    parser.add_argument("--post_layers", help="post_attention_layers", type=int)
    parser.add_argument("--num_step", help="num_step", type=int)
    parser.add_argument("--l1", help="loss weight for lamda1", type=int)
    parser.add_argument("--l2", help="loss weight for lamda2", type=int)
    parser.add_argument(
        "--shuffle", action="store_true", help="shuffle video frame when test"
    )
    parser.add_argument(
        "--extend", action="store_true", help="extend time length of input"
    )
    parser.add_argument(
        "--flip_time", action="store_true", help="flip the input in time direction"
    )
    parser.add_argument(
        "--post_process",
        help="post_process type: choice of [MultiHeadAttention, DaMultiHeadAttention, MultiLSTMAttention, MultiConvAttention]",
        type=str,
    )
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag
    if args.query_layers:
        config.MODEL.PARAMS.query_lstm_num_layers = args.query_layers
    if args.video_layers:
        config.MODEL.PARAMS.video_lstm_num_layers = args.video_layers
    if args.post_layers:
        config.MODEL.PARAMS.post_attention_layers = args.post_layers
    if args.post_process:
        config.MODEL.PARAMS.post_attention = args.post_process
    if args.num_step:
        config.MODEL.PARAMS.num_step = args.num_step
    if args.num_heads:
        config.MODEL.PARAMS.num_heads = args.num_heads
    if args.shuffle:
        config.TEST.SHUFFLE_VIDEO_FRAME = args.shuffle
    if args.extend:
        config.DATASET.EXTEND_TIME = args.extend
    if args.flip_time:
        config.DATASET.FLIP_TIME = args.flip_time
    if args.l1:
        config.LOSS.LOCALIZATION = args.l1
    if args.l2:
        config.LOSS.MATCH = args.l2


def iterator(dataset_name, split):
    if split == "train":
        train_dataset = getattr(datasets, dataset_name)("train")
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=False,
            collate_fn=datasets.collate_fn,
        )
    elif split == "val":
        val_dataset = getattr(datasets, dataset_name)("val")
        dataloader = DataLoader(
            val_dataset,
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=False,
            collate_fn=datasets.collate_fn,
        )
    elif split == "test":
        test_dataset = getattr(datasets, dataset_name)("test")
        dataloader = DataLoader(
            test_dataset,
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=False,
            collate_fn=datasets.collate_fn,
        )
    elif split == "train_no_shuffle":
        eval_train_dataset = getattr(datasets, dataset_name)("train")
        dataloader = DataLoader(
            eval_train_dataset,
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=False,
            collate_fn=datasets.collate_fn,
        )
    else:
        raise NotImplementedError

    return dataloader


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


if __name__ == "__main__":
    args = parse_args()
    reset_config(config, args)

    # set pytorch and numpy
    set_th_config(12345)

    model_name = config.MODEL.NAME
    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    count_parameters(model)
    # Device configuration
    cuda_str = "cuda" if args.gpus is None else "cuda:{}".format(args.gpus)
    device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
    model.to(device)

    # create model dir
    cfg_name = os.path.basename(args.cfg).split(".yaml")[0]
    home_dir = os.path.join("results", config.DATASET.NAME) + "/" + cfg_name
    if not config.TAG is None:
        home_dir = home_dir + "_" + config.TAG
    model_dir = os.path.join(home_dir, "checkpoints")
    event_dir = os.path.join(home_dir, "event")

    if args.mode.lower() == "train":
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(event_dir):
            os.makedirs(event_dir)
        # create SummaryWriter()
        writer = SummaryWriter(log_dir=event_dir)

        # create logger
        head = "%(asctime)-15s %(message)s"
        logging.basicConfig(
            filename=str(os.path.join(home_dir, "log.txt")), format=head
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger("").addHandler(console)

        logger.info("\n" + pprint.pformat(args))
        logger.info("\n" + pprint.pformat(config))

        score_writer = open(
            os.path.join(home_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )

        dataloader_train = iterator(config.DATASET.NAME, "train")
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            lr=config.TRAIN.LR,
            num_train_steps=config.TRAIN.MAX_EPOCH * len(dataloader_train),
            warmup_proportion=0,
        )
        # optimizer, scheduler = build_optimizer_and_scheduler(
        #     model,
        #     lr=config.TRAIN.LR,
        #     num_train_steps=config.TRAIN.MAX_EPOCH,
        #     warmup_proportion=0)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=config.TRAIN.MILE_STONE,
        #     gamma=config.TRAIN.GAMMA,
        #     last_epoch=0)
        best_r1i7 = -1.0
        global_step = 0
        logger.info("start training...")
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(config.TRAIN.MAX_EPOCH):
            model.train()
            step = 0
            step_epoch = len(dataloader_train)
            for data, anno in dataloader_train:
                global_step += 1
                step += 1
                batch_word_vectors = data["batch_word_vectors"].to(device)
                batch_pos_tags = data["batch_pos_tags"].to(device)
                batch_txt_mask = data["batch_txt_mask"].squeeze(2).to(device)
                batch_vis_feats = data["batch_vis_feats"].to(device)
                batch_vis_mask = data["batch_vis_mask"].squeeze(2).to(device)
                # batch_start_label = data['batch_start_label'].to(device)
                # batch_end_label = data['batch_end_label'].to(device)
                batch_internel_label = data["batch_internel_label"].to(device)
                batch_start_frame = data["batch_start_frame"].to(device)
                batch_end_frame = data["batch_end_frame"].to(device)
                with torch.cuda.amp.autocast():
                    output = model(
                        batch_word_vectors,
                        batch_pos_tags,
                        batch_txt_mask,
                        batch_vis_feats,
                        batch_vis_mask,
                    )
                    start_logits, end_logits, additional_logits = (
                        output[0],
                        output[1],
                        output[2],
                    )
                    # compute loss
                    # kl_loss, triplet_loss, distance = 0, 0, 0
                    kl_loss = model.aligment_score(
                        output[3],
                        output[4],
                        batch_txt_mask,
                        batch_vis_mask,
                        batch_internel_label,
                    )
                    kl_loss2 = model.aligment_score(
                        output[3],
                        output[6],
                        batch_txt_mask,
                        batch_vis_mask,
                        batch_internel_label,
                    )
                    # kl_loss3 = model.aligment_score(
                    #     output[7],
                    #     output[8],
                    #     batch_txt_mask,
                    #     batch_vis_mask,
                    #     batch_internel_label,
                    # )
                    kl_loss = kl_loss + kl_loss2
                    # kl_loss = kl_loss + kl_loss3

                    loc_loss, match_loss = model.compute_loss(
                        start_logits,
                        end_logits,
                        additional_logits,
                        batch_start_frame,
                        batch_end_frame,
                        batch_internel_label,
                        batch_vis_mask,
                    )
                    early_loss = model.early_pred_loss(
                        output[4], output[5], batch_internel_label, batch_vis_mask
                    )
                    # early_loss = 0
                    total_loss = (
                        config.LOSS.LOCALIZATION * loc_loss
                        + config.LOSS.MATCH * match_loss
                        + config.LOSS.KL * kl_loss
                        + config.LOSS.EARLY * early_loss
                    )
                    # total_loss = config.LOSS.LOCALIZATION * loc_loss + config.LOSS.MATCH * match_loss

                if global_step % 50 == 0:
                    logger.info(
                        "epoch: {}, step: {}/{}, lr: {:.6f},  total: {:.4f}, loc: {:.4f}, match: {:.4f}, kl: {:.6f}, early: {:.4f}".format(
                            epoch + 1,
                            step,
                            step_epoch,
                            optimizer.state_dict()["param_groups"][0]["lr"],
                            total_loss,
                            loc_loss,
                            match_loss,
                            kl_loss,
                            early_loss,
                        )
                    )
                writer.add_scalar("Loss_train/all", total_loss, global_step)
                writer.add_scalar("Loss_train/loc", loc_loss, global_step)
                writer.add_scalar("Loss_train/kl", kl_loss, global_step)
                # writer.add_scalar('Loss_train/distance', distance, global_step)
                writer.add_scalar("Loss_train/match", match_loss, global_step)
                writer.add_scalar("Loss_train/early", early_loss, global_step)
                writer.add_scalar(
                    "learning_rate",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                    global_step,
                )
                # compute and apply gradients
                optimizer.zero_grad()
                # total_loss.backward()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.LOSS.CLIP_NORM
                )  # clip gradient
                # train batch end
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

            # train epoch end, then evaluate
            model.eval()
            if config.TEST.EVAL_TRAIN:
                # trainset eval
                r1i3, r1i5, r1i7, mi, score_str, statistics_str = model.eval_test(
                    model=model,
                    data_loader=iterator(config.DATASET.NAME, "train_no_shuffle"),
                    device=device,
                    mode="train",
                    epoch=epoch + 1,
                )
                # logger.info("statistics results of {}'s train dataset ".format(
                #     config.DATASET.NAME))
                # logger.info("\n" + statistics_str)
                logger.info(
                    "training dataset results  Epoch: %2d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f"
                    % (epoch + 1, r1i3, r1i5, r1i7, mi)
                )
                writer.add_scalar("Accuracy_train/r1i3", r1i3, epoch + 1)
                writer.add_scalar("Accuracy_train/r1i5", r1i5, epoch + 1)
                writer.add_scalar("Accuracy_train/r1i7", r1i7, epoch + 1)
                writer.add_scalar("Accuracy_train/miou", mi, epoch + 1)
            if not config.DATASET.NO_VAL:
                # val set eval
                r1i3, r1i5, r1i7, mi, score_str, statistics_str = model.eval_test(
                    model=model,
                    data_loader=iterator(config.DATASET.NAME, "val"),
                    device=device,
                    mode="val",
                    epoch=epoch + 1,
                )
                # logger.info("statistics results of {}'s val dataset ".format(
                #     config.DATASET.NAME))
                # logger.info("\n" + statistics_str)
                logger.info(
                    "val dataset results  Epoch: %2d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f"
                    % (epoch + 1, r1i3, r1i5, r1i7, mi)
                )
                writer.add_scalar("Accuracy_val/r1i3", r1i3, epoch + 1)
                writer.add_scalar("Accuracy_val/r1i5", r1i5, epoch + 1)
                writer.add_scalar("Accuracy_val/r1i7", r1i7, epoch + 1)
                writer.add_scalar("Accuracy_val/miou", mi, epoch + 1)
            # testset eval
            tb = PrettyTable()
            tb.field_names = ["dataset", "epoch", "r1i3", "r1i5", "r1i7", "miou"]
            test_datasets = ["Charades", "ActivityNet", "TACoS"]
            if config.DATASET.NAME != "Combine":
                test_datasets = [config.DATASET.NAME]
            # for test_set in :
            for test_set in test_datasets:
                r1i3, r1i5, r1i7, mi, score_str, statistics_str = model.eval_test(
                    model=model,
                    data_loader=iterator(test_set, "test"),
                    device=device,
                    mode="test",
                    epoch=epoch + 1,
                )
                # logger.info(
                #     'test dataset results  Epoch: %2d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f'
                #     % (epoch + 1, r1i3, r1i5, r1i7, mi))
                # logger.info("statistics results of {}'s test dataset ".format(
                #     test_set))
                # logger.info("\n" + statistics_str)
                writer.add_scalar(
                    "Accuracy_test/{}_r1i3".format(test_set), r1i3, epoch + 1
                )
                writer.add_scalar(
                    "Accuracy_test/{}_r1i5".format(test_set), r1i5, epoch + 1
                )
                writer.add_scalar(
                    "Accuracy_test/{}_r1i7".format(test_set), r1i7, epoch + 1
                )
                writer.add_scalar(
                    "Accuracy_test/{}_miou".format(test_set), mi, epoch + 1
                )
                tb.add_row(
                    [
                        test_set,
                        epoch + 1,
                        "{:.6f}".format(r1i3),
                        "{:.4f}".format(r1i5),
                        "{:.4f}".format(r1i7),
                        "{:.4f}".format(mi),
                    ]
                )
                if test_set == config.DATASET.NAME or (
                    config.DATASET.NAME == "Combine" and test_set == "Charades"
                ):
                    if r1i7 >= best_r1i7:
                        best_r1i7 = r1i7
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                model_dir, "{}_{}.t7".format(model_name, epoch + 1)
                            ),
                        )
                        # only keep the top-3 model checkpoints
                        filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
            score_writer.write(tb.get_string() + "\n")
            score_writer.flush()
            logger.info("\n" + tb.get_string())
            del tb
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, "{}_final.model".format(model_name)),
        )
        score_writer.close()
        writer.close()
    elif args.mode.lower() == "test":
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        print("loadding pretrained weight...")
        filename = get_last_checkpoint(model_dir, suffix="t7")
        print("using ->{}<- ...".format(filename))
        model.load_state_dict(torch.load(filename))
        print("load done, start testing...")
        model.eval()
        config.DATASET.EXTEND_TIME = False
        config.DATASET.FLIP_TIME = False
        config.TEST.BATCH_SIZE = 2
        tb = PrettyTable()
        tb.field_names = ["dataset", "epoch", "r1i3", "r1i5", "r1i7", "miou"]
        # for test_set in ["Charades", "ActivityNet", "TACoS"]:
        for test_set in [config.DATASET.NAME]:
            start_time = time.time()
            r1i3, r1i5, r1i7, mi, score_str, statistics_str = model.eval_test(
                model=model,
                data_loader=iterator(test_set, "test"),
                device=device,
                mode="test",
                epoch=None,
                shuffle_video_frame=config.TEST.SHUFFLE_VIDEO_FRAME,
            )
            end_time = time.time()
            print("all time:", end_time - start_time)
            tb.add_row(
                [
                    test_set,
                    None,
                    "{:.6f}".format(r1i3),
                    "{:.6f}".format(r1i5),
                    "{:.6f}".format(r1i7),
                    "{:.6f}".format(mi),
                ]
            )
            print("statistics results of {}'s test dataset ".format(test_set))
            print(statistics_str)
        print(tb.get_string())
    elif args.mode.lower() == "test_train":
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        print("loadding pretrained weight...")
        filename = get_last_checkpoint(model_dir, suffix="t7")
        # filename = os.path.join(model_dir, '{}_final.model'.format(model_name))
        print("using ->{}<- ...".format(filename))
        model.load_state_dict(torch.load(filename))
        print("load done, start testing...")
        model.eval()
        config.DATASET.EXTEND_TIME = False
        config.DATASET.FLIP_TIME = False
        tb = PrettyTable()
        tb.field_names = ["dataset", "epoch", "r1i3", "r1i5", "r1i7", "miou"]
        for test_set in ["Charades", "ActivityNet", "TACoS"]:
            r1i3, r1i5, r1i7, mi, score_str, statistics_str = model.eval_test(
                model=model,
                data_loader=iterator(test_set, "train_no_shuffle"),
                device=device,
                mode="test",
                epoch=None,
                shuffle_video_frame=config.TEST.SHUFFLE_VIDEO_FRAME,
            )
            tb.add_row(
                [
                    test_set,
                    None,
                    "{:.6f}".format(r1i3),
                    "{:.6f}".format(r1i5),
                    "{:.6f}".format(r1i7),
                    "{:.6f}".format(mi),
                ]
            )
            print("statistics results of {}'s train dataset ".format(test_set))
            print(statistics_str)
        print(tb.get_string())
    elif args.mode.lower() == "debug":
        from core.runner_utils import index_to_time

        # vid = 'ZMY8M'
        # vid = '0PU21'
        # vid = 'W0QSB'
        vid = "4J1AP"
        print("video id: ", vid)
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        print("loadding pretrained weight...")
        filename = get_last_checkpoint(model_dir, suffix="t7")
        # filename = os.path.join(model_dir, '{}_final.model'.format(model_name))
        print("using ->{}<- ...".format(filename))
        model.load_state_dict(torch.load(filename))
        print("load done, start testing...")
        model.eval()
        train_dataset = getattr(datasets, "Charades")("train")
        train_dataset2 = getattr(datasets, "Charades")("test")
        train_dataset.annotations.extend(train_dataset2.annotations)
        anno = []
        for item in train_dataset.annotations:
            if item["video"] == vid:
                anno.append(item)
        train_dataset.annotations = anno
        dataloader = DataLoader(
            train_dataset,
            batch_size=len(anno),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=datasets.collate_fn,
        )
        with torch.no_grad():
            for idx, batch_data in enumerate(dataloader):
                data, annos = batch_data
                batch_word_vectors = data["batch_word_vectors"].to(device)
                batch_pos_tags = data["batch_pos_tags"].to(device)
                batch_txt_mask = data["batch_txt_mask"].squeeze(2).to(device)
                batch_vis_feats = data["batch_vis_feats"].to(device)
                batch_vis_mask = data["batch_vis_mask"].squeeze(2).to(device)
                # batch_start_label = data['batch_start_label'].to(device)
                # batch_end_label = data['batch_end_label'].to(device)
                batch_internel_label = data["batch_internel_label"].to(device)
                batch_start_frame = data["batch_start_frame"].to(device)
                batch_end_frame = data["batch_end_frame"].to(device)
                batch_extend_pre = data["batch_extend_pre"].to(device)
                batch_extend_suf = data["batch_extend_suf"].to(device)

                # compute predicted results
                with torch.cuda.amp.autocast():
                    output = model(
                        batch_word_vectors,
                        batch_pos_tags,
                        batch_txt_mask,
                        batch_vis_feats,
                        batch_vis_mask,
                    )
                start_logits, end_logits = output[0], output[1]
                start_indices, end_indices = model.extract_index(
                    start_logits, end_logits
                )
                start_indices = start_indices.cpu().numpy()
                end_indices = end_indices.cpu().numpy()
                batch_vis_mask = batch_vis_mask.cpu().numpy()
                print("video length:", batch_vis_mask.sum(1))
                batch_extend_pre = batch_extend_pre.cpu().numpy()
                batch_extend_suf = batch_extend_suf.cpu().numpy()
                print("gt_start:", batch_start_frame.cpu().numpy())
                print("gt_end:", batch_end_frame.cpu().numpy())
                print("pred_intermediate:")
                print(torch.sigmoid(output[5].squeeze()).cpu().numpy(), output[5].shape)
                print("pred_start:", start_indices)
                print("pred_end:", end_indices)
                # print("duration:",
                #       [annos[i]["duration"] for i in range(len(annos))])
                print([annos[i]["description"] for i in range(len(annos))])
                print("gt_time:", [annos[i]["times"] for i in range(len(annos))])
                for (
                    vis_mask,
                    start_index,
                    end_index,
                    extend_pre,
                    extend_suf,
                    anno,
                ) in zip(
                    batch_vis_mask,
                    start_indices,
                    end_indices,
                    batch_extend_pre,
                    batch_extend_suf,
                    annos,
                ):

                    start_time, end_time = index_to_time(
                        start_index,
                        end_index,
                        vis_mask.sum(),
                        extend_pre,
                        extend_suf,
                        anno["duration"],
                    )
                    print("pred_start:", start_time)
                    print("pred_end:", end_time)

    else:
        raise NotImplementedError
