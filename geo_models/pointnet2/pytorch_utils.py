# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, mlp_spec, bn=True):
        super(SharedMLP, self).__init__()
        
        self.mlp = nn.ModuleList()
        for i in range(len(mlp_spec) - 1):
            self.mlp.append(nn.Conv2d(mlp_spec[i], mlp_spec[i + 1], kernel_size=1, bias=not bn))
            if bn:
                self.mlp.append(nn.BatchNorm2d(mlp_spec[i + 1]))
            self.mlp.append(nn.ReLU())

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


def feature_dropout_no_scaling(x, dropout_prob, training, inplace=False):
    """
    Dropout that doesn't scale the features during training.
    """
    if not training or dropout_prob == 0:
        return x
    
    if inplace:
        mask = torch.bernoulli(x.new_empty(x.size()).fill_(1 - dropout_prob))
        x.mul_(mask)
        return x
    else:
        mask = torch.bernoulli(x.new_empty(x.size()).fill_(1 - dropout_prob))
        return x * mask
