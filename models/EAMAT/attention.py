import torch
import torch.nn as nn
import math

from .operation import Conv1D, mask_logits
from .encoder import MultiStepLSTMEncoder, TemporalContextModule


class MultiHeadAttention(nn.Module):
    def __init__(self, configs):
        super(MultiHeadAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        self.key = Conv1D(in_dim=dim,
                          out_dim=dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.value = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output


class MultiLSTMAttention(nn.Module):
    def __init__(self, configs):
        super(MultiLSTMAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        num_layers = configs.num_layers
        num_step = configs.num_step
        bi_direction = configs.bi_direction

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = MultiStepLSTMEncoder(in_dim=dim,
                                          out_dim=dim,
                                          num_layers=num_layers,
                                          num_step=num_step,
                                          bi_direction=bi_direction,
                                          drop_rate=drop_rate)
        self.key = MultiStepLSTMEncoder(in_dim=dim,
                                        out_dim=dim,
                                        num_layers=num_layers,
                                        num_step=num_step,
                                        bi_direction=bi_direction,
                                        drop_rate=drop_rate)
        self.value = MultiStepLSTMEncoder(in_dim=dim,
                                          out_dim=dim,
                                          num_layers=num_layers,
                                          num_step=num_step,
                                          bi_direction=bi_direction,
                                          drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output


class MultiConvAttention(nn.Module):
    def __init__(self, configs):
        super(MultiConvAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        kernels = configs.kernels

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = TemporalContextModule(in_dim=dim,
                                           out_dim=dim,
                                           kernels=kernels,
                                           drop_rate=drop_rate)
        self.key = TemporalContextModule(in_dim=dim,
                                         out_dim=dim,
                                         kernels=kernels,
                                         drop_rate=drop_rate)
        self.value = TemporalContextModule(in_dim=dim,
                                           out_dim=dim,
                                           kernels=kernels,
                                           drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output