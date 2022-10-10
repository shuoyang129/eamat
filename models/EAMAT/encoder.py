import torch
import torch.nn as nn

from .operation import Conv1D


class LSTMEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_layers,
                 bi_direction=False,
                 drop_rate=0.0):
        super(LSTMEncoder, self).__init__()

        self.layers_norm1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.layers_norm2 = nn.LayerNorm(out_dim, eps=1e-6)

        self.dropout = nn.Dropout(p=drop_rate)
        self.encoder = nn.LSTM(in_dim,
                               out_dim // 2 if bi_direction else out_dim,
                               num_layers=num_layers,
                               bidirectional=bi_direction,
                               dropout=drop_rate,
                               batch_first=True)

        self.linear = Conv1D(in_dim=out_dim,
                             out_dim=out_dim,
                             kernel_size=1,
                             stride=1,
                             bias=True,
                             padding=0)

    def forward(self, input_feature):
        input_feature = self.layers_norm1(input_feature)
        output, _ = self.encoder(input_feature)
        output = self.layers_norm2(output)
        output = self.dropout(output)
        output = self.linear(output)
        return output


class MultiStepLSTMEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_layers,
                 num_step=1,
                 bi_direction=False,
                 drop_rate=0.0):
        super(MultiStepLSTMEncoder, self).__init__()

        self.num_step = num_step
        self.out_dim = out_dim
        self.layers_norm = nn.LayerNorm(in_dim, eps=1e-6)

        self.dropout = nn.Dropout(p=drop_rate)

        self.encoder = nn.ModuleList([
            nn.LSTM(in_dim,
                    out_dim // 2 if bi_direction else out_dim,
                    num_layers=num_layers,
                    bidirectional=bi_direction,
                    dropout=drop_rate,
                    batch_first=True) for _ in range(num_step)
        ])
        self.linear = Conv1D(in_dim=int(num_step * out_dim),
                             out_dim=out_dim,
                             kernel_size=1,
                             stride=1,
                             bias=True,
                             padding=0)

    def forward(self, input_feature):
        input_feature = self.layers_norm(input_feature)
        B, seq_len, _ = input_feature.shape
        # assert seq_len // self.num_step == 0, "length of sequence({}) must be devided by num_step({})".format(
        #     seq_len, self.num_step)
        output = []
        for i in range(self.num_step):
            encoder_i = self.encoder[i]
            output_i = input_feature.new_zeros([B, seq_len, self.out_dim])
            input_i_len = (seq_len // (i + 1)) * (i + 1)
            for j in range(i + 1):
                input_j = input_feature[:, j:input_i_len:(i + 1), :]
                output_j, _ = encoder_i(input_j)
                output_i[:, j:input_i_len:(i + 1), :] = output_j
            output_i = self.dropout(output_i)
            output.append(output_i)
        output = torch.cat(output, dim=2)
        output = self.linear(output)
        return output

class TemporalContextModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernels=[3], drop_rate=0.):
        super(TemporalContextModule, self).__init__()
        self.dropout = nn.Dropout(p=drop_rate)
        self.temporal_convs = nn.ModuleList([
            Conv1D(in_dim=in_dim,
                   out_dim=out_dim,
                   kernel_size=s,
                   stride=1,
                   padding=s // 2,
                   bias=True) for s in kernels
        ])
        self.out_layer = Conv1D(in_dim=out_dim * len(kernels),
                                out_dim=out_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)

    def forward(self, input_feature):
        intermediate = []
        for layer in self.temporal_convs:
            intermediate.append(layer(input_feature))
        intermediate = torch.cat(intermediate, dim=-1)
        out = self.out_layer(intermediate)
        return out
