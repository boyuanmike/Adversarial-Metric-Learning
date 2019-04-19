import torch.optim as optim
from torch.utils import model_zoo

from common.utils import *
from datasets.dataset import *
from models.modifiedgooglenet import ModifiedGoogLeNet
from models.net import Generator, Discriminator
import time
import matplotlib.pyplot as plt


def train(func_train_one_batch, param_dict, path):
    dis_loss = []
    gen_loss = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_dir_path = "/disk-main/logs/"

    p = Logger(log_dir_path, **param_dict)

    # load data base
    if p.dataset is 'car196':
        data = Car196(root=path)
    else:
        data = CUB_200_2011(root=path)

    sampler = BalancedBatchSampler(data.train.labels, n_samples=p.n_samples, n_classes=98)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    # train_loader = torch.utils.data.DataLoader(data.train, batch_size=p.batch_size)
    train_loader = torch.utils.data.DataLoader(data.train, batch_sampler=sampler, **kwargs)
    # train_it = iter(train_loader)
    test_loader = torch.utils.data.DataLoader(data.test, batch_size=p.batch_size)

    # construct the model
    model = ModifiedGoogLeNet(p.out_dim, p.normalize_output).to(device)
    model_gen = Generator().to(device)
    model_dis = Discriminator(p.out_dim, p.out_dim).to(device)

    model_optimizer = optim.Adam(model.parameters(), lr=p.learning_rate / 1e3, weight_decay=p.l2_weight_decay)
    gen_optimizer = optim.Adam(model_gen.parameters(), lr=p.learning_rate * 10)
    dis_optimizer = optim.Adam(model_dis.parameters(), lr=p.learning_rate, weight_decay=p.l2_weight_decay)
    model_feat_optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)

    time_origin = time.time()
    best_nmi_1 = 0.
    best_f1_1 = 0.
    best_nmi_2 = 0.
    best_f1_2 = 0.

    for epoch in range(p.num_epochs):
        time_begin = time.time()
        epoch_loss_gen = []
        epoch_loss_dis = []

        for batch in tqdm(train_loader, desc='# {}'.format(epoch)):
            triplet_batch = generate_random_triplets_from_batch(batch, n_samples=p.n_samples, n_class=98)
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

        nmi, f1 = evaluate(device, model, model_dis, test_loader, epoch,
                           distance=p.distance_type, normalize=p.normalize_output)
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

        print("#", epoch)
        print("time: {} ({})".format(epoch_time, total_time))
        print("[train] loss gen:", loss_average_gen)
        print("[train] loss dis:", loss_average_dis)
        print("[test]  nmi:", nmi)
        print("[test]  f1:", f1)
        print("[test]  nmi:", best_nmi_1, "  f1:", best_f1_1, "for max nmi")
        print("[test]  nmi:", best_nmi_2, "  f1:", best_f1_2, "for max f1")
        print(p)

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
