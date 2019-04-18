import colorama
import os

import torch.nn.functional as F
from sklearn.model_selection import ParameterSampler
import numpy as np
from common.utils import (UniformDistribution,
                          LogUniformDistribution, load_params, lossfun_one_batch)
from common.train_eval import train
import torch
colorama.init()


def main():
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    random_state = None
    num_runs = 100
    save_distance_matrix = False
    param_distributions = dict(
        learning_rate=LogUniformDistribution(low=1e-5, high=1e-4),
    )
    static_params = dict(
        n_samples = 5,
        num_epochs=100,
        num_batches_per_epoch=60,
        batch_size=32,
        out_dim=512,
        crop_size=224,
        normalize_output=True,
        normalize_bn=True,
        optimizer='Adam',
        distance_type='euclidean',
        dataset='car196',
        method='triplet',
        loss='triplet',
        tradeoff=1.0,
        l2_weight_decay=1e-3,
        alpha=1.0,
        learning_rate = 1e-3
    )
    # sampler = ParameterSampler(param_distributions, num_runs, random_state)
    train(__file__, lossfun_one_batch,
          static_params, save_distance_matrix, '/disk-main/car196')
    # for random_params in sampler:
    #     params = {}
    #     params.update(random_params)
    #     params.update(static_params)
    #
    #     stop = train(__file__, lossfun_one_batch,
    #                  params, save_distance_matrix, '/disk-main/car196')


if __name__ == '__main__':
    main()
