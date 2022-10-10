import os
import glob
import random
import numpy as np
import torch
from torch.cuda.profiler import start
import torch.utils.data
import torch.backends.cudnn
from tqdm import tqdm
from prettytable import PrettyTable

from .data_util import index_to_time


def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def dcor(x, y):
    m, _ = x.shape
    assert len(x.shape) == 2
    assert len(y.shape) == 2

    dx = pairwise_dist(x)
    dy = pairwise_dist(y)

    dx_m = dx - dx.mean(dim=0)[None, :] - dx.mean(dim=1)[:, None] + dx.mean()
    dy_m = dy - dy.mean(dim=0)[None, :] - dy.mean(dim=1)[:, None] + dy.mean()

    dcov2_xy = (dx_m * dy_m).sum() / float(m * m)
    dcov2_xx = (dx_m * dx_m).sum() / float(m * m)
    dcov2_yy = (dy_m * dy_m).sum() / float(m * m)

    dcor = torch.sqrt(dcov2_xy) / torch.sqrt(
        (torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy)).clamp(min=0) + 1e-10)

    return dcor


def pairwise_dist(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    output = -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()
    return torch.sqrt(output.clamp(min=0) + 1e-10)


def filter_checkpoints(model_dir, suffix='t7', max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(
                os.path.basename(model_path).split('_')[1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix='t7'):
    model_filenames = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(
            os.path.basename(model_filename).split('_')[1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    last_checkpoint = sorted_tuples[-1]
    return last_checkpoint[1]


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


def cal_statistics(preds, durations):
    start_fre = [0] * 10
    end_fre = [0] * 10
    duration_fre = [0] * 10
    start_end_fre = [[0] * 10 for _ in range(10)]
    tb = PrettyTable()
    tb.field_names = [
        "type", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
        "1.0"
    ]
    for pred, duration in zip(preds, durations):
        start_f = int(pred[0] / duration * 10)
        end_f = min(int(pred[1] / duration * 10), 9)
        duration_f = min(int((pred[1] - pred[0]) / duration * 10), 9)
        start_fre[start_f] += 1
        end_fre[end_f] += 1
        duration_fre[duration_f] += 1
        start_end_fre[start_f][end_f] += 1
    assert len(preds) == len(durations)
    all_len = len(durations)
    for i in range(10):
        start_fre[i] /= all_len
        end_fre[i] /= all_len
        duration_fre[i] /= all_len
        for j in range(10):
            start_end_fre[i][j] /= all_len
            start_end_fre[i][j] = "{:.6f}".format(start_end_fre[i][j])
    start_fre = ["{:.6f}".format(s) for s in start_fre]
    end_fre = ["{:.6f}".format(s) for s in end_fre]
    duration_fre = ["{:.6f}".format(s) for s in duration_fre]
    tb.add_row(["start_fre"] + start_fre)
    tb.add_row(["end_fre"] + end_fre)
    tb.add_row(["duration_fre"] + duration_fre)
    tb.add_row(["--"] * 11)
    for i in range(10):
        tb.add_row([str((i + 1) / 10)] + start_end_fre[i])
    return tb.get_string()


def eval_test(model,
              data_loader,
              device,
              mode='test',
              epoch=None,
              global_step=None):
    ious = []
    with torch.no_grad():
        for idx, batch_data in tqdm(enumerate(data_loader),
                                    total=len(data_loader),
                                    desc='evaluate {}'.format(mode)):
            data, annos = batch_data
            batch_word_vectors = data['batch_word_vectors'].to(device)
            batch_txt_mask = data['batch_txt_mask'].squeeze().to(device)
            batch_vis_feats = data['batch_vis_feats'].to(device)
            batch_vis_mask = data['batch_vis_mask'].squeeze().to(device)

            # compute predicted results
            _, start_logits, end_logits = model(batch_word_vectors,
                                                batch_txt_mask,
                                                batch_vis_feats,
                                                batch_vis_mask)
            start_indices, end_indices = model.extract_index(
                start_logits, end_logits)
            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()
            batch_vis_mask = batch_vis_mask.cpu().numpy()
            for vis_mask, start_index, end_index, anno in zip(
                    batch_vis_mask, start_indices, end_indices, annos):
                start_time, end_time = index_to_time(start_index, end_index,
                                                     vis_mask.sum(),
                                                     anno["duration"])
                iou = calculate_iou(i0=[start_time, end_time],
                                    i1=anno['times'])
                ious.append(iou)
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)
    return r1i3, r1i5, r1i7, mi, score_str