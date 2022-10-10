import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
import numpy as np
import math

from .operation import Conv1D, mask_logits


class CQFusion(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQFusion, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(
            context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = torch.softmax(mask_logits(score, q_mask.unsqueeze(1)),
                               dim=2)  # (batch_size, c_seq_len, q_seq_len)
        score_t = torch.softmax(mask_logits(score, c_mask.unsqueeze(2)),
                                dim=1)  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t),
                           context)  # (batch_size, c_seq_len, dim)
        output = torch.cat(
            [context, c2q,
             torch.mul(context, c2q),
             torch.mul(context, q2c)],
            dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output * c_mask.unsqueeze(2)

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand(
            [-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand(
            [-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res

