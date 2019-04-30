# This file is the pytorch implementation of https://github.com/duanyq14/DAML/blob/master/lib/models/modified_googlenet.py

import torch
import torch.nn as nn

from models.google_net import googlenet


class ModifiedGoogLeNet(nn.Module):
    def __init__(self, out_dims=64, normalize_output=False):
        super(ModifiedGoogLeNet, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        self.googlenet.fc = nn.Linear(in_features=1024, out_features=out_dims)
        self.normalize_output = normalize_output

    def forward(self, x):
        if self.training and self.googlenet.aux_logits:
            *_, y = self.googlenet(x)
        else:
            y = self.googlenet(x)
        if self.normalize_output:
            y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
            y = y / y_norm.expand_as(y)
        return y
