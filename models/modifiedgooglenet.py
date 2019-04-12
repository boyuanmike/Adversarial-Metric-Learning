# This file is the pytorch implementation of https://github.com/duanyq14/DAML/blob/master/lib/models/modified_googlenet.py

from models.google_net import *
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class ModifiedGoogLeNet(GoogLeNet):
    def __init__(self, out_dims=64, normalize_output=False):
        super(ModifiedGoogLeNet, self).__init__()
        # add new layers
        self.bn_fc = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(in_features=1024, out_features=out_dims)

        self.local_response_normal = nn.LocalResponseNorm(size=5, k=1, alpha=1e-4 / 5)

        image_mean = torch.tensor([123.0, 117.0, 104.0])  # RGB
        self._image_mean = image_mean[None, :, None, None]
        self.normalize_output = normalize_output

    def forward(self, x):
        # subtract image mean
        x = x - self._image_mean
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 3, 2)
        h = self.local_response_normal(h)

        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.local_response_normal(h)
        h = F.max_pool2d(h, 3, 2)

        h = self.inception3a(h)
        h = self.inception3b(h)
        h = F.max_pool2d(h, 3, 2)

        h = self.inception4a(h)
        h = self.inception4b(h)
        h = self.inception4c(h)
        h = self.inception4d(h)
        h = self.inception4e(h)
        h = F.max_pool2d(h, 3, 2)

        h = self.inception5a(h)
        h = self.inception5b(h)  # [120, 1024, 6, 6]

        h = F.adaptive_avg_pool2d(h, 1)
        # h = F.avg_pool2d(h, 7, 1)

        h = self.bn_fc(h)
        y = self.fc(h.reshape(*h.size()[:2]))
        if self.normalize_output:
            y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
            y = y / y_norm.expand_as(y)
        return y
