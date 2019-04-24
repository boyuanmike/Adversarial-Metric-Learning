import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils import Logger, evaluate
from datasets.dataset import BalancedBatchSampler, gen_triplets_from_batch
from datasets.dataset import Car196, CUB_200_2011
from models.modifiedgooglenet import ModifiedGoogLeNet
import os


def train(train_one_batch, param_dict, path, log_dir_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    p = Logger(log_dir_path, **param_dict)

    # load data base
    if p.dataset is 'car196':
        data = Car196(root=path)
    else:
        data = CUB_200_2011(root=path)

    sampler = BalancedBatchSampler(data.train.label_to_indices, n_samples=p.n_samples, n_classes=p.n_classes)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_loader = DataLoader(data.train, batch_sampler=sampler, **kwargs)  # (5 * 98, 3, 224, 224)

    # train_iter = iter(train_loader)
    # batch = next(train_iter)
    # generate_random_triplets_from_batch(batch, p.n_samples, p.n_classes)

    test_loader = DataLoader(data.test, batch_size=p.batch_size)

    # construct the model
    model_a = ModifiedGoogLeNet(p.out_dim, p.normalize_output).to(device)
    model_b = ModifiedGoogLeNet(p.out_dim, p.normalize_output).to(device)

    model_a_opt = optim.Adam(model_a.parameters(), lr=p.learning_rate)
    model_b_opt = optim.Adam(model_b.parameters(), lr=p.learning_rate)

    time_origin = time.time()
    model_a_best_nmi_1 = 0.
    model_a_best_f1_1 = 0.
    model_a_best_nmi_2 = 0.
    model_a_best_f1_2 = 0.

    model_b_best_nmi_1 = 0.
    model_b_best_f1_1 = 0.
    model_b_best_nmi_2 = 0.
    model_b_best_f1_2 = 0.

    total_adv_loss = []
    total_stu_loss = []

    for epoch in range(p.num_epochs):
        time_begin = time.time()
        epoch_adv_loss = 0
        epoch_stu_loss = 0
        total = 0
        for data_batch in tqdm(train_loader, desc='# {}'.format(epoch)):
            triplet_batch = gen_triplets_from_batch(data_batch,
                                                    n_samples=p.n_samples,
                                                    n_class=p.n_classes)
            adv_loss, stu_loss = train_one_batch(device, model_a, model_b,
                                                 model_a_opt, p, triplet_batch)

            epoch_adv_loss += adv_loss
            epoch_stu_loss += stu_loss
            total += triplet_batch[0].size(0)

            triplet_batch = gen_triplets_from_batch(data_batch,
                                                    n_samples=p.n_samples,
                                                    n_class=p.n_classes)
            adv_loss, stu_loss = train_one_batch(device, model_b, model_a,
                                                 model_b_opt, p, triplet_batch)
            epoch_adv_loss += adv_loss
            epoch_stu_loss += stu_loss
            total += triplet_batch[0].size(0)

        average_adv_loss = epoch_adv_loss / total
        average_stu_loss = epoch_stu_loss / total

        total_adv_loss.append(average_adv_loss)
        total_stu_loss.append(average_stu_loss)

        model_a_nmi, model_a_f1 = evaluate(device, model_a, test_loader,
                                           n_classes=p.n_classes,
                                           normalize=p.normalize_output)

        model_b_nmi, model_b_f1 = evaluate(device, model_b, test_loader,
                                           n_classes=p.n_classes,
                                           normalize=p.normalize_output)

        if model_a_nmi > model_a_best_nmi_1:
            model_a_best_nmi_1 = model_a_nmi
            model_a_best_f1_1 = model_a_f1
            torch.save(model_a, os.path.join(p.model_save_path, "model_a.pt"))
        if model_a_f1 > model_a_best_f1_2:
            model_a_best_nmi_2 = model_a_nmi
            model_a_best_f1_2 = model_a_f1

        if model_b_nmi > model_b_best_nmi_1:
            model_b_best_nmi_1 = model_b_nmi
            model_b_best_f1_1 = model_b_f1
            torch.save(model_a, os.path.join(p.model_save_path, "model_b.pt"))
        if model_b_f1 > model_b_best_f1_2:
            model_b_best_nmi_2 = model_b_nmi
            model_b_best_f1_2 = model_b_f1

        time_end = time.time()
        epoch_time = time_end - time_begin
        total_time = time_end - time_origin

        print("#", epoch)
        print("time: {} ({})".format(epoch_time, total_time))
        print("[train] loss adv:", average_adv_loss)
        print("[train] loss stu:", average_stu_loss)

        print("[test]  model A nmi:", model_a_nmi)
        print("[test]  model A f1:", model_a_f1)
        print("[test]  nmi:", model_a_best_nmi_1, "  f1:", model_a_best_f1_1, "for model A max nmi")
        print("[test]  nmi:", model_a_best_nmi_2, "  f1:", model_a_best_f1_2, "for model A max f1")

        print("[test]  model B nmi:", model_b_nmi)
        print("[test]  model B f1:", model_b_f1)
        print("[test]  nmi:", model_b_best_nmi_1, "  f1:", model_b_best_f1_1, "for model B max nmi")
        print("[test]  nmi:", model_b_best_nmi_2, "  f1:", model_b_best_f1_2, "for model B max f1")

        print(p)

    plt.figure(0)
    plt.plot(total_adv_loss)
    plt.ylabel("adv_loss")
    plt.savefig('adv_loss.png')

    plt.figure(1)
    plt.plot(total_stu_loss)
    plt.ylabel("stu_loss")
    plt.savefig('stu_loss.png')

    # print("total epochs: {} ({} [s])".format(logger.epoch, logger.total_time))
    # print("best test score (at # {})".format(logger.epoch_best))
    # print("[test]  soft:", logger.soft_test_best)
    # print("[test]  hard:", logger.hard_test_best)
    # print("[test]  retr:", logger.retrieval_test_best)
    # print(str(p).replace(', ', '\n'))
    # print()
