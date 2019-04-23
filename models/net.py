# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch


# class Generator(nn.Module):
#     def __init__(self, out_dim=512, normalize_output=False):
#         super(Generator, self).__init__()
#         self.out_dim = out_dim
#         self.normalize_output = normalize_output
#
#         self.fc1 = nn.Linear(self.out_dim * 3, self.out_dim)
#         nn.init.normal_(self.fc1.weight, std=0.02)
#         # self.l0.weight.data.copy_(???)
#         # w = chainer.initializers.Normal(0.02)
#
#         self.fc2 = nn.Linear(self.out_dim, self.out_dim)
#         nn.init.normal_(self.fc2.weight, std=0.02)
#         # copy weight to l1
#         # self.l1.weight.data.copy_(???)
#
#     def forward(self, x):
#         fc1_out = self.fc1(x)
#         fc2_out = self.fc2(F.relu(fc1_out))
#         if self.normalize_output:
#             fc2_out_norm = torch.norm(fc2_out, p=2, dim=1, keepdim=True)
#             fc2_out = fc2_out / fc2_out_norm.expand_as(fc2_out)
#         return fc2_out

class Generator(nn.Module):
    def __init__(self, out_dim=512, normalize_output=False):
        super(Generator, self).__init__()
        self.out_dim = out_dim
        self.normalize_output = normalize_output

        self.fc1 = nn.Linear(self.out_dim * 3, 1024)
        nn.init.normal_(self.fc1.weight, std=0.02)
        # self.l0.weight.data.copy_(???)
        # w = chainer.initializers.Normal(0.02)

        self.fc2 = nn.Linear(1024, 1024)
        nn.init.normal_(self.fc2.weight, std=0.02)
        # copy weight to l1
        # self.l1.weight.data.copy_(???)
        self.fc3 = nn.Linear(1024, self.out_dim)
        nn.init.normal_(self.fc3.weight, std=0.02)

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(F.relu(fc1_out))
        fc3_out = self.fc3(F.relu(fc2_out))
        if self.normalize_output:
            fc3_out_norm = torch.norm(fc3_out, dim=1, keepdim=True)
            fc3_out = fc3_out / fc3_out_norm.expand_as(fc3_out)
        return fc3_out


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, normalize_output=True):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize_output = normalize_output

        self.fc1 = nn.Linear(self.in_dim, self.out_dim)
        nn.init.eye_(self.fc1.weight)
        # if self.in_dim == self.out_dim:
        #    self.l0.weight.data.copy_(torch.eye(self.out_dim))
        self.fc2 = nn.Linear(self.out_dim, self.out_dim)
        nn.init.eye_(self.fc2.weight)
        # copy weight to l1
        # self.l1.weight.data.copy_(torch.eye(self.out_dim))

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(F.relu(fc1_out))
        if self.normalize_output:
            fc2_out_norm = torch.norm(fc2_out, p=2, dim=1, keepdim=True)
            fc2_out = fc2_out / fc2_out_norm.expand_as(fc2_out)
        return fc2_out
