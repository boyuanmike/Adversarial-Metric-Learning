import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import six
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder

from common.utils import *
from common.evaluation import evaluate_recall_asym
from common.evaluation import evaluate_recall
from datasets.dataset import *
from models.modifiedgooglenet import ModifiedGoogLeNet
from models.net import Generator, Discriminator
from torch.utils import model_zoo


# def get_optimizer(model):
#     optimizer = optim.Adam(model.())
#     return optimizer


def train(main_script_path, func_train_one_batch, param_dict, savev_distance_matrix=False, path=None):


    dis_loss = []
    gen_loss = []
    script_filename = os.path.splitext(os.path.basename(main_script_path))[0]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config_parser = six.moves.configparser.ConfigParser()
    config_parser.read('config')
    log_dir_path = "/disk-main/logs/"

    p = Logger(log_dir_path, **param_dict)

    # load data base
    if p.dataset is 'car196':
        data = Car196(root=path)
    else:
        data = CUB_200_2011(root=path)

    sampler = BalancedBatchSampler(data.train.labels, n_samples=p.n_samples, n_classes= 98)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    # train_loader = torch.utils.data.DataLoader(data.train, batch_size=p.batch_size)
    train_loader = torch.utils.data.DataLoader(data.train, batch_sampler = sampler, **kwargs)
    # train_it = iter(train_loader)
    test_loader = torch.utils.data.DataLoader(data.test, batch_size=p.batch_size)
    # construct the model

    model = ModifiedGoogLeNet(p.out_dim, p.normalize_output).to(device)

    model_gen = Generator().to(device)
    model_dis = Discriminator(p.out_dim, p.out_dim).to(device)

    model_optimizer = optim.Adam(model.parameters(),lr= p.learning_rate/1000,weight_decay=p.l2_weight_decay)
    gen_optimizer = optim.Adam(model_gen.parameters(), lr =p.gen_learning_rate/10,weight_decay=p.l2_weight_decay)
    dis_optimizer = optim.Adam(model_dis.parameters(), lr =p.dis_learning_rate,weight_decay=p.l2_weight_decay)
    model_feat_optimizer = optim.Adam(model.parameters(),lr= p.learning_rate,weight_decay=p.l2_weight_decay)


    stop = False
    logger = Logger(log_dir_path)
    logger.soft_test_best = [0]
    time_origin = time.time()
    best_nmi_1 = 0.
    best_f1_1 = 0.
    best_nmi_2 = 0.
    best_f1_2 = 0.

    for epoch in range(p.num_epochs):
        time_begin = time.time()
        epoch_loss_gen = []
        epoch_loss_dis = []
        # loss = 0
        # t = tqdm(range(p.num_batches_per_epoch))
        # for i in t:
        #     t.set_description(desc='# {}'.format(epoch))
        for batch in tqdm(train_loader, desc='# {}'.format(epoch)):


            triplet_batch = generate_random_triplets_from_batch(batch,n_samples= p.n_samples, n_class= 98 )


            loss_gen_list, loss_dis_list = func_train_one_batch(device, model, model_gen, model_dis,
                                                                model_optimizer, model_feat_optimizer, gen_optimizer,
                                                                dis_optimizer, p, triplet_batch,
                                                                epoch)
            # epoch_loss_gen.append(loss_gen.item())
            # epoch_loss_dis.append(loss_dis.item())
            for loss_gen in loss_gen_list:
                epoch_loss_gen.append(loss_gen.item())
            for loss_dis in loss_dis_list:
                epoch_loss_dis.append(loss_dis.item())




        loss_average_gen = sum(epoch_loss_gen) / float(len(epoch_loss_gen))
        loss_average_dis = sum(epoch_loss_dis) / float(len(epoch_loss_dis))

        dis_loss.append(loss_average_dis)
        gen_loss.append(loss_average_gen)

        D = [0]
        soft = [0]
        hard = [0]
        retrieval = [0]

        nmi, f1 = evaluate(device, model, model_dis, test_loader, p.distance_type,
                           return_distance_matrix=savev_distance_matrix, epoch=epoch)
        if nmi > best_nmi_1:
            best_nmi_1 = nmi
            best_f1_1 = f1
            torch.save(model, "/disk-main/models/model.pt")
            torch.save(model_gen, '/disk-main/models/model_gen.pt')
            torch.save(model_dis, '/disk-main/models/model_dis.pt')
        if f1 > best_f1_2:
            best_nmi_2 = nmi
            best_f1_2 = f1

        time_end = time.time()
        epoch_time = time_end - time_begin
        total_time = time_end - time_origin

        logger.epoch = epoch
        logger.total_time = total_time
        logger.gen_loss_log.append(loss_average_gen)
        logger.dis_loss_log.append(loss_average_dis)
        logger.train_log.append([soft[0], hard[0], retrieval[0]])
        print("#", epoch)
        print("time: {} ({})".format(epoch_time, total_time))
        print("[train] loss gen:", loss_average_gen)
        print("[train] loss dis:", loss_average_dis)
        print("[test]  nmi:", nmi)
        print("[test]  f1:", f1)
        print("[test]  nmi:", best_nmi_1, "  f1:", best_f1_1, "for max nmi")
        print("[test]  nmi:", best_nmi_2, "  f1:", best_f1_2, "for max f1")
        print(p)

    dir_name = "-".join([p.dataset, script_filename,
                         time.strftime("%Y%m%d%H%M%S"), str(logger.soft_test_best[0])])

    logger.save(dir_name)
    p.save(dir_name)

    plt.plot(dis_loss)
    plt.ylabel("dis_loss")
    plt.savefig('dis_loss.png')

    plt.plot(gen_loss)
    plt.ylabel("gen_loss")
    plt.savefig('gen_loss.png')
    # print("total epochs: {} ({} [s])".format(logger.epoch, logger.total_time))
    # print("best test score (at # {})".format(logger.epoch_best))
    # print("[test]  soft:", logger.soft_test_best)
    # print("[test]  hard:", logger.hard_test_best)
    # print("[test]  retr:", logger.retrieval_test_best)
    # print(str(p).replace(', ', '\n'))
    # print()

    return stop
