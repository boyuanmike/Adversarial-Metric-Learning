import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils import Logger, evaluate
from datasets.dataset import BalancedBatchSampler, generate_random_triplets_from_batch
from datasets.dataset import Car196, CUB_200_2011
from models.modifiedgooglenet import ModifiedGoogLeNet
from models.net import Generator, Discriminator
import os


def train(func_train_one_batch, param_dict, path, log_dir_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    p = Logger(log_dir_path, **param_dict)

    # load data base
    if p.dataset is 'car196':
        data = Car196(root=path)
    else:
        data = CUB_200_2011(root=path)

    sampler = BalancedBatchSampler(data.train.label_to_indices, n_samples=p.n_samples, n_classes=p.n_classes)
    kwargs = {'num_workers': 6, 'pin_memory': True}

    train_loader = DataLoader(data.train, batch_sampler=sampler, **kwargs)  # (5 * 98, 3, 224, 224)

    # train_iter = iter(train_loader)
    # batch = next(train_iter)
    # generate_random_triplets_from_batch(batch, p.n_samples, p.n_classes)

    test_loader = DataLoader(data.test, batch_size=p.batch_size)

    # construct the model
    model = ModifiedGoogLeNet(p.out_dim, p.normalize_hidden).to(device)
    model_gen = Generator(p.out_dim, p.normalize_hidden).to(device)
    model_dis = Discriminator(p.out_dim, p.out_dim, p.normalize_output).to(device)

    model_optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)
    gen_optimizer = optim.Adam(model_gen.parameters(), lr=p.learning_rate)
    dis_optimizer = optim.Adam(model_dis.parameters(), lr=p.learning_rate)
    model_feat_optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)

    time_origin = time.time()
    best_nmi_1 = 0.
    best_f1_1 = 0.
    best_nmi_2 = 0.
    best_f1_2 = 0.

    dis_loss = []
    gen_loss = []

    for epoch in range(p.num_epochs):
        time_begin = time.time()
        epoch_loss_gen = 0
        epoch_loss_dis = 0
        total = 0
        for batch in tqdm(train_loader, desc='# {}'.format(epoch)):
            triplet_batch = generate_random_triplets_from_batch(batch, n_samples=p.n_samples, n_class=p.n_classes)
            loss_gen, loss_dis = func_train_one_batch(device, model, model_gen, model_dis,
                                                      model_optimizer, model_feat_optimizer, gen_optimizer,
                                                      dis_optimizer, p, triplet_batch,
                                                      epoch)

            epoch_loss_gen += loss_gen
            epoch_loss_dis += loss_dis
            total += triplet_batch[0].size(0)

        loss_average_gen = epoch_loss_gen / total
        loss_average_dis = epoch_loss_dis / total

        dis_loss.append(loss_average_dis)
        gen_loss.append(loss_average_gen)

        nmi, f1 = evaluate(device, model, model_dis, test_loader, epoch,
                           n_classes=p.n_classes,
                           distance=p.distance_type,
                           normalize=p.normalize_output,
                           neg_gen_epoch=p.neg_gen_epoch)
        if nmi > best_nmi_1:
            best_nmi_1 = nmi
            best_f1_1 = f1
            torch.save(model, os.path.join(p.model_save_path, "model.pt"))
            torch.save(model_gen, os.path.join(p.model_save_path, "model_gen.pt"))
            torch.save(model_dis, os.path.join(p.model_save_path, "model_dis.pt"))
        if f1 > best_f1_2:
            best_nmi_2 = nmi
            best_f1_2 = f1

        time_end = time.time()
        epoch_time = time_end - time_begin
        total_time = time_end - time_origin

        print("#", epoch)
        print("time: {} ({})".format(epoch_time, total_time))
        print("[train] loss gen:", loss_average_gen)
        print("[train] loss dis:", loss_average_dis)
        print("[test]  nmi:", nmi)
        print("[test]  f1:", f1)
        print("[test]  nmi:", best_nmi_1, "  f1:", best_f1_1, "for max nmi")
        print("[test]  nmi:", best_nmi_2, "  f1:", best_f1_2, "for max f1")
        print(p)

    plt.figure(0)
    plt.plot(dis_loss)
    plt.ylabel("dis_loss")
    plt.savefig('dis_loss.png')

    plt.figure(1)
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
