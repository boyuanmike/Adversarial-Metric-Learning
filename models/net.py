# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, out_dim=512):
        super(Generator, self).__init__()
        self.out_dim = out_dim

        self.l0 = nn.Linear(self.out_dim * 3, self.out_dim)
        # self.l0.weight.data.copy_(???)
        # w = chainer.initializers.Normal(0.02)

        self.l1 = nn.Linear(self.out_dim, self.out_dim)
        # copy weight to l1
        # self.l1.weight.data.copy_(???)

    def forward(self, x):
        h = self.l0(x)
        h1 = nn.functional.relu(h)
        h2 = self.l1(h1)
        h3 = nn.functional.relu(h2)
        return h3


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.l0 = nn.Linear(self.in_dim, self.out_dim)
        if self.in_dim == self.out_dim:
            self.l0.weight.data.copy_(torch.eye(self.out_dim))
        self.l1 = nn.Linear(self.out_dim, self.out_dim)
        # copy weight to l1
        self.l1.weight.data.copy_(torch.eye(self.out_dim))

    def forward(self, x):
        h = self.l0(x)
        h1 = nn.functional.relu(h)
        return self.l1(h1)
