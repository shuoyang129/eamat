import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

from core.config import config
from core.runner_utils import index_to_time, calculate_iou, calculate_iou_accuracy, cal_statistics

# from .attention import MultiHeadAttention, DaMultiHeadAttention
from . import attention
from .encoder import LSTMEncoder, MultiStepLSTMEncoder, TemporalContextModule
# from fusion import CQFusion, CosineFusion, InteractorFusion
from . import fusion
from .layers import Projection, Prediction, CQConcatenate, PositionalEmbedding, TransformerPositionalEmbedding
from .operation import Conv1D, mask_logits
from .triplet_loss import batch_all_triplet_loss, pairwise_distances

# torch.set_printoptions(profile="full", linewidth=1000, precision=2)


class EAMAT(nn.Module):
    def __init__(self):
        super(EAMAT, self).__init__()
        configs = config.MODEL.PARAMS
        self.debug_print = configs.DEBUG
        self.video_affine = Projection(in_dim=configs.video_feature_dim,
                                       dim=configs.dim,
                                       drop_rate=configs.drop_rate)

        self.query_affine = Projection(in_dim=configs.query_feature_dim,
                                       dim=configs.dim,
                                       drop_rate=configs.drop_rate)
        self.query_position = configs.query_position
        self.video_position = configs.video_position
        if self.query_position or self.video_position:
            self.v_pos_embedding = PositionalEmbedding(configs.dim, 500)
            self.q_pos_embedding = PositionalEmbedding(configs.dim, 30)
            # self.pos_embedding = TransformerPositionalEmbedding(configs.dim, 500,drop_rate=configs.drop_rate)

        query_attention_layer = getattr(attention,
                                        configs.query_attention)(configs)

        video_attention_layer = getattr(attention,
                                        configs.video_attention)(configs)

        self.query_encoder = nn.Sequential(*[
            copy.deepcopy(query_attention_layer)
            for _ in range(configs.query_attention_layers)
        ])
        self.video_encoder = nn.Sequential(*[
            copy.deepcopy(video_attention_layer)
            for _ in range(configs.video_attention_layers)
        ])

        # early_attention_layer = getattr(attention,
        #                                 configs.early_attention)(configs)
        # self.early_encoder = nn.Sequential(*[
        #     copy.deepcopy(early_attention_layer)
        #     for _ in range(configs.early_attention_layers)
        # ])

        self.entity_prediction_layer = Prediction(in_dim=configs.dim,
                                                  hidden_dim=configs.dim // 2,
                                                  out_dim=3,
                                                  drop_rate=configs.drop_rate)

        self.fg_prediction_layer = Prediction(in_dim=configs.dim,
                                              hidden_dim=configs.dim // 2,
                                              out_dim=1,
                                              drop_rate=configs.drop_rate)
        self.early_fusion_layer = getattr(fusion,
                                          configs.fusion_module)(configs.dim)
        self.fusion_layer = getattr(fusion, configs.fusion_module)(configs.dim)

        post_attention_layer = getattr(attention,
                                       configs.post_attention)(configs)
        self.post_attention_layer = nn.Sequential(*[
            copy.deepcopy(post_attention_layer)
            for _ in range(configs.post_attention_layers)
        ])
        self.video_encoder2 = nn.Sequential(*[
            copy.deepcopy(post_attention_layer)
            for _ in range(configs.video_attention_layers)
        ])
        # self.post_processing = MultiStepLSTMEncoder(
        #     in_dim=configs.dim,
        #     out_dim=configs.dim,
        #     num_layers=3,
        #     num_step=3,
        #     bi_direction=True,
        #     drop_rate=configs.drop_rate)
        # self.post_processing = getattr(
        #     attention, configs.self_attention)(configs.dim, configs.num_heads,
        #                                        configs.drop_rate)

        # self.post_processing = TemporalContextModule(
        #     in_dim=configs.dim,
        #     out_dim=configs.dim,
        #     kernels=configs.kernels,
        #     drop_rate=configs.drop_rate)

        # self.post_processing = nn.Sequential(*[
        #     Conv1D(configs.dim, configs.dim),
        #     nn.Tanh(),
        #     Conv1D(configs.dim, configs.dim),
        #     nn.Tanh(),
        #     Conv1D(configs.dim, configs.dim)
        # ])

        self.starting = Prediction(in_dim=configs.dim,
                                   hidden_dim=configs.dim // 2,
                                   out_dim=1,
                                   drop_rate=configs.drop_rate)
        self.ending = Prediction(in_dim=configs.dim,
                                 hidden_dim=configs.dim // 2,
                                 out_dim=1,
                                 drop_rate=configs.drop_rate)

        self.intering = Prediction(in_dim=configs.dim,
                                   hidden_dim=configs.dim // 2,
                                   out_dim=1,
                                   drop_rate=configs.drop_rate)

    def forward(self, batch_word_vectors, batch_pos_tags, batch_txt_mask,
                batch_vis_feats, batch_vis_mask):
        batch_vis_feats = self.video_affine(batch_vis_feats)
        if self.video_position:
            batch_vis_feats = batch_vis_feats + self.v_pos_embedding(
                batch_vis_feats)
        batch_vis_feats = batch_vis_feats * batch_vis_mask.unsqueeze(2)
        for i, module in enumerate(self.video_encoder):
            if i == 0:
                video_features = module(batch_vis_feats, batch_vis_mask)
            else:
                video_features = module(video_features, batch_vis_mask)
        for i, module in enumerate(self.video_encoder2):
            if i == 0:
                video_features2 = module(batch_vis_feats, batch_vis_mask)
            else:
                video_features2 = module(video_features2, batch_vis_mask)

        batch_word_vectors = self.query_affine(batch_word_vectors)
        if self.query_position:
            batch_word_vectors = batch_word_vectors + self.q_pos_embedding(
                batch_word_vectors)
        batch_word_vectors = batch_word_vectors * batch_txt_mask.unsqueeze(2)
        # add pos_tag embedding to query features
        # query_tag_embedding = F.normalize(
        #     self.query_tag_embedding(batch_pos_tags), dim=-1)
        # batch_word_vectors = batch_word_vectors + query_tag_embedding

        # query_features = self.query_encoder(query_features, batch_txt_mask)
        for i, module in enumerate(self.query_encoder):
            if i == 0:
                query_features = module(batch_word_vectors, batch_txt_mask)
            else:
                query_features = module(query_features, batch_txt_mask)
        # entity pred
        # entity_prob = self.entity_prediction_layer(query_features)
        # entity_prob = torch.softmax(10 * entity_prob, dim=-1)
        # # print("entity_prob: ", entity_prob.shape)  # B,N,3
        # entity_features = batch_word_vectors * entity_prob[:, :,
        #                                                    0].unsqueeze(2)
        # action_feature = batch_word_vectors * entity_prob[:, :, 1].unsqueeze(2)
        zeros = batch_pos_tags.new_zeros(batch_pos_tags.shape)
        ones = batch_pos_tags.new_ones(batch_pos_tags.shape)
        entity_prob = torch.where(
            torch.abs(batch_pos_tags - 2) < 1e-10, zeros, ones)
        action_prob = torch.where(
            torch.abs(batch_pos_tags - 1) < 1e-10, zeros, ones)
        entity_features = batch_word_vectors * entity_prob.unsqueeze(2)
        action_feature = batch_word_vectors * action_prob.unsqueeze(2)
        # print("e", entity_prob)
        # print("a", action_prob)
        entity_features = query_features + entity_features
        action_feature = query_features + action_feature

        # First stage
        entity_video_fused = self.early_fusion_layer(video_features,
                                                     entity_features,
                                                     batch_vis_mask,
                                                     batch_txt_mask)
        # for i, module in enumerate(self.early_encoder):
        #     entity_video_fused = module(entity_video_fused, batch_vis_mask)

        fg_prob = self.fg_prediction_layer(entity_video_fused)
        if not self.training and self.debug_print:
            print('fg_prob', torch.sigmoid(fg_prob))
        # fg_vis_feature = batch_vis_feats * torch.sigmoid(
        #     fg_prob) + video_features
        fg_vis_feature = (video_features2 +
                          video_features) * torch.sigmoid(fg_prob)

        fused_action_feature = self.fusion_layer(fg_vis_feature,
                                                 action_feature,
                                                 batch_vis_mask,
                                                 batch_txt_mask)

        # fused_action_feature += entity_video_fused * torch.sigmoid(fg_prob)

        for i, module in enumerate(self.post_attention_layer):
            fused_action_feature = module(fused_action_feature, batch_vis_mask)
        # fused_action_feature = self.post_processing(
        #     fused_feature * batch_vis_mask.unsqueeze(2)) + fused_action_feature

        pred_start = self.starting(fused_action_feature).squeeze(2)
        pred_start = mask_logits(pred_start, batch_vis_mask)

        pred_end = self.ending(fused_action_feature).squeeze(2)
        pred_end = mask_logits(pred_end, batch_vis_mask)

        pred_inter = self.intering(fused_action_feature).squeeze(2)
        # pred_inter = torch.sigmoid(pred_inter) * batch_vis_mask

        return pred_start, pred_end, pred_inter, query_features, video_features2, fg_prob.squeeze(
            2), video_features, batch_word_vectors, batch_vis_feats

    def compute_loss(self, pred_start, pred_end, pred_inter, start_labels,
                     end_labels, inter_label, mask):

        # start_regularity = self.regularization_score(pred_start)
        # end_regularity = self.regularization_score(pred_end)

        start_loss = self.compute_boundary_loss(pred_start, start_labels)
        end_loss = self.compute_boundary_loss(pred_end, end_labels)
        inter_loss = self.compute_location_loss(pred_inter, inter_label, mask)

        return start_loss + end_loss, inter_loss

    # def regularization_score(self, logits):
    #     score = torch.softmax(logits, dim=1)
    #     # global term to force most value to zeros
    #     global_term = torch.sum(score, dim=1).mean()
    #     # local term to force a gap for scores
    #     local_term = torch.sum(score * torch.log(score) * -1.0, dim=1).mean()
    #     return global_term + local_term

    def regression_loss(self, pred, targets):
        B, T = pred.shape
        frame_index = (torch.arange(T)).unsqueeze(0).repeat(
            (B, 1)).to(pred.device)
        delta = 10
        pred = (torch.softmax(pred * delta, dim=1) * frame_index).sum(1)
        loss = F.smooth_l1_loss(torch.log(pred + 1), torch.log(targets + 1))
        return loss

    def compute_boundary_loss(self, pred, targets):
        return F.cross_entropy(pred, targets.long())
        # return self.regression_loss(pred, targets)

    def compute_location_loss(self, pred, targets, mask):
        weights_per_location = torch.where(targets == 0.0, targets + 1.0,
                                           2.0 * targets)
        loss_per_location = nn.BCEWithLogitsLoss(reduction='none')(pred,
                                                                   targets)
        loss_per_location = loss_per_location * weights_per_location
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask,
                         dim=1) / (torch.sum(mask, dim=1) + 1e-13)
        return loss.mean()

    def early_pred_loss(self, video_features, pred, targets, mask):
        return self.compute_location_loss(pred, targets, mask)
        # B, T, channels = video_features.shape
        # video_features = F.normalize(video_features, p=2,
        #                              dim=-1)  # B, T, channels
        # video_sim = torch.bmm(video_features, video_features.transpose(1, 2))
        # # min_value = []
        # # for i in range(B):
        # #     start = int(start_frame[i])
        # #     end = int(end_frame[i])
        # #     middle = (start + end) // 2
        # #     min_value.append(
        # #         (video_sim[i][start][middle] + video_sim[i][end][middle]) / 2)
        # # min_value = torch.tensor(min_value) * 0.8
        # # similarity between GT segments
        # target_sim = video_sim * targets.unsqueeze(1) * targets.unsqueeze(2)
        # target_sim[target_sim <= 1e-10] = 1000
        # min_value = torch.min(target_sim.reshape(B, -1), dim=1,
        #                       keepdim=True)[0]
        # min_value = min_value.unsqueeze(1).repeat(1, T, T)
        # # similarity between GT and others segments
        # video_sim = video_sim * targets.unsqueeze(1).repeat(1, T, 1)
        # video_sim = torch.where(video_sim >= min_value, video_sim.new_ones(1),
        #                         video_sim.new_zeros(1))
        # video_sim = torch.sum(video_sim, dim=-1)
        # # 大于一半时为1
        # num = torch.sum(targets, dim=1).unsqueeze(1).repeat(1, T)
        # targets = torch.where(video_sim >= num / 2, video_sim.new_ones(1),
        #                       video_sim.new_zeros(1))
        # # print(video_sim, video_sim.shape)
        # # print(targets)
        # weights_per_location = torch.where(targets == 0.0, targets + 1.0,
        #                                    2.0 * targets)
        # loss_per_location = nn.BCEWithLogitsLoss(reduction='none')(pred,
        #                                                            targets)
        # loss_per_location = loss_per_location * weights_per_location
        # mask = mask.type(torch.float32)
        # loss = torch.sum(loss_per_location * mask,
        #                  dim=1) / (torch.sum(mask, dim=1) + 1e-13)
        # return loss.mean()

    def aligment_score(self,
                       query_features,
                       video_features,
                       query_mask,
                       video_mask,
                       inner_label,
                       GT_inner=True):
        B, T, channels = video_features.shape

        query_features = query_features.sum(1) / query_mask.sum(1).unsqueeze(1)
        query_features = F.normalize(query_features, p=2, dim=1)  # B, channels

        if GT_inner:
            frame_weights = inner_label / video_mask.sum(1, keepdim=True)
        else:
            norm_video = F.normalize(video_features, p=2, dim=-1)
            frame_weights = torch.bmm(query_features.unsqueeze(1),
                                      norm_video.transpose(1, 2))  # B,1,T
            frame_weights = mask_logits(frame_weights.squeeze(1),
                                        video_mask)  # B,T
            frame_weights = torch.softmax(frame_weights, dim=-1)

        # norm_video = F.normalize(video_features, p=2, dim=-1).contiguous()
        # # distance = torch.bmm(norm_video, norm_video.transpose(1, 2))
        # distance = torch.cdist(norm_video, norm_video)
        # # distance = 1 - distance
        # distance = distance * inner_label.unsqueeze(1) * inner_label.unsqueeze(
        #     2)
        # distance = torch.sum(distance.reshape(B, -1), dim=-1) / (
        #     torch.sum(inner_label, dim=1) * torch.sum(inner_label, dim=1) +
        #     1e-30)
        # distance = distance.mean()

        video_features = video_features * frame_weights.unsqueeze(2)
        video_features = video_features.sum(1)
        video_features = F.normalize(video_features, p=2, dim=1)
        video_sim = torch.matmul(video_features, video_features.T)
        video_sim = torch.softmax(video_sim, dim=-1)
        query_sim = torch.matmul(query_features, query_features.T)
        query_sim = torch.softmax(query_sim, dim=-1)
        kl_loss = (F.kl_div(query_sim.log(), video_sim, reduction='sum') +
                   F.kl_div(video_sim.log(), query_sim, reduction='sum')) / 2

        # triplet loss to enforce query feature close to referenced video features
        # features = torch.cat((query_features, video_features), dim=0)
        # labels = torch.cat((torch.arange(B), torch.arange(B)), dim=0)
        # labels = labels.to(query_features.device).detach()
        # triplet_loss, _ = batch_all_triplet_loss(labels,
        #                                          features,
        #                                          margin=0.2,
        #                                          squared=True)

        # print(kl_loss, triplet_loss, distance)
        # return kl_loss * 100 + triplet_loss * 10 + distance * 0.1
        # return kl_loss, triplet_loss, distance
        return kl_loss

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2),
                             end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0],
                                   dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0],
                                 dim=1)  # (batch_size, )
        return start_index, end_index

    @staticmethod
    def eval_test(model,
                  data_loader,
                  device,
                  mode='test',
                  epoch=None,
                  shuffle_video_frame=False):
        ious = []
        preds, durations = [], []
        with torch.no_grad():
            for idx, batch_data in tqdm(enumerate(data_loader),
                                        total=len(data_loader),
                                        desc='evaluate {}'.format(mode)):
                data, annos = batch_data
                batch_word_vectors = data['batch_word_vectors'].to(device)
                batch_pos_tags = data['batch_pos_tags'].to(device)
                batch_txt_mask = data['batch_txt_mask'].squeeze(2).to(device)
                batch_vis_feats = data['batch_vis_feats'].to(device)
                batch_vis_mask = data['batch_vis_mask'].squeeze(2).to(device)
                batch_extend_pre = data['batch_extend_pre'].to(device)
                batch_extend_suf = data['batch_extend_suf'].to(device)

                if shuffle_video_frame:
                    B = batch_vis_feats.shape[0]
                    for i in range(B):
                        T = batch_vis_mask[i].sum().int().item()
                        pre = batch_extend_pre[i].item()
                        new_T = torch.randperm(T)
                        batch_vis_feats[i, torch.arange(T) +
                                        pre] = batch_vis_feats[i, new_T + pre]
                # compute predicted results
                with torch.cuda.amp.autocast():
                    output = model(batch_word_vectors, batch_pos_tags,
                                   batch_txt_mask, batch_vis_feats,
                                   batch_vis_mask)
                start_logits, end_logits = output[0], output[1]
                start_indices, end_indices = model.extract_index(
                    start_logits, end_logits)
                start_indices = start_indices.cpu().numpy()
                end_indices = end_indices.cpu().numpy()
                batch_vis_mask = batch_vis_mask.cpu().numpy()
                batch_extend_pre = batch_extend_pre.cpu().numpy()
                batch_extend_suf = batch_extend_suf.cpu().numpy()

                for vis_mask, start_index, end_index, extend_pre, extend_suf, anno in zip(
                        batch_vis_mask, start_indices, end_indices,
                        batch_extend_pre, batch_extend_suf, annos):

                    start_time, end_time = index_to_time(
                        start_index, end_index, vis_mask.sum(), extend_pre,
                        extend_suf, anno["duration"])
                    # print(start_index, end_index, vis_mask.sum(), start_time,
                    #       end_time, anno["duration"])
                    iou = calculate_iou(i0=[start_time, end_time],
                                        i1=anno['times'])
                    ious.append(iou)
                    preds.append((start_time, end_time))
                    durations.append(anno["duration"])
        statistics_str = cal_statistics(preds, durations)
        r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
        r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
        r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
        mi = np.mean(ious) * 100.0
        # write the scores
        score_str = "Epoch {}\n".format(epoch)
        score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
        score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
        score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
        score_str += "mean IoU: {:.2f}\n".format(mi)
        return r1i3, r1i5, r1i7, mi, score_str, statistics_str
