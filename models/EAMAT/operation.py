import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim,
                                out_channels=out_dim,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)