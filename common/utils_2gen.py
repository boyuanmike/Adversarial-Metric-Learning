# -*- coding: utf-8 -*-


import itertools

import torch.nn.functional as F
import yaml
from tqdm import tqdm

from common.evaluation import evaluate_cluster
from datasets.dataset import *
from functions.triplet_loss import triplet_loss as F_tloss


def load_params(filename):
    with open(filename) as f:
        params = yaml.load(f)
    return params


def make_positive_pairs(num_classes, num_examples_per_class, repetition=1):
    c = num_classes
    n = num_examples_per_class
    num_pairs_per_class = n * (n - 1) // 2

    pairs_posi_class0 = np.array(list(itertools.combinations(range(n), 2)))
    offsets = n * np.repeat(np.arange(c), num_pairs_per_class)[:, None]
    pairs_posi = np.tile(pairs_posi_class0, (c, 1)) + offsets
    return np.tile(pairs_posi, (repetition, 1))


def iter_combinatorial_pairs(queue, num_examples, batch_size, interval,
                             num_classes, augment_positive=False):
    num_examples_per_class = num_examples // num_classes
    pairs = np.array(list(itertools.combinations(range(num_examples), 2)))

    if augment_positive:
        additional_positive_pairs = make_positive_pairs(
            num_classes, num_examples_per_class, num_classes - 1)
        pairs = np.concatenate((pairs, additional_positive_pairs))

    num_pairs = len(pairs)
    num_batches = num_pairs // batch_size
    perm = np.random.permutation(num_pairs)
    for i, batch_indexes in enumerate(np.array_split(perm, num_batches)):
        if i % interval == 0:
            x, c = queue.get()
            x = x.astype(np.float32) / 255.0
            c = c.ravel()
        indexes0, indexes1 = pairs[batch_indexes].T
        x0, x1, c0, c1 = x[indexes0], x[indexes1], c[indexes0], c[indexes1]
        t = np.int32(c0 == c1)  # 1 if x0 and x1 are same class, 0 otherwise
        yield x0, x1, t


class NPairMCIndexMaker(object):
    def __init__(self, batch_size, num_classes, num_per_class):
        self.batch_size = batch_size  # number of examples in a batch
        self.num_classes = num_classes  # number of classes
        self.num_per_class = num_per_class  # number of examples per class

    def get_epoch_indexes(self):
        B = self.batch_size
        K = self.num_classes
        M = self.num_per_class
        N = K * M  # number of total examples
        num_batches = M * int(K // B)  # number of batches per epoch

        indexes = np.arange(N, dtype=np.int32).reshape(K, M)
        epoch_indexes = []
        for m in range(M):
            perm = np.random.permutation(K)
            c_batches = np.array_split(perm, num_batches // M)
            for c_batch in c_batches:
                b = len(c_batch)  # actual number of examples of this batch
                indexes_anchor = M * c_batch + m

                positive_candidates = np.delete(indexes[c_batch], m, axis=1)
                indexes_positive = positive_candidates[
                    range(b), np.random.choice(M - 1, size=b)]

                epoch_indexes.append((indexes_anchor, indexes_positive))

        return epoch_indexes


class Logger(defaultdict):
    def __init__(self, root_dir_path, **kwargs):
        super(Logger, self).__init__(list, kwargs)
        if not os.path.exists(root_dir_path):
            os.makedirs(root_dir_path)
        self._root_dir_path = root_dir_path

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __str__(self):
        keys = filter(lambda key: not key.startswith('_'), self)
        return ", ".join(["{}:{}".format(key, self[key]) for key in keys])

    def save(self, dir_name):
        dir_path = os.path.join(self._root_dir_path, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        others = []
        for key, value in self.items():
            if key.startswith('_'):
                continue

            if isinstance(value, (np.ndarray, list)):
                np.save(os.path.join(dir_path, key + ".npy"), value)
            else:
                others.append("{}: {}".format(key, value))

        with open(os.path.join(dir_path, "log.txt"), "a") as f:
            text = "\n".join(others) + "\n"
            f.write(text)


def iterate_forward(device, model, dis_model, test_loader, epoch, normalize=False):
    y_batches = []
    c_batches = []
    for anchors, labels in tqdm(test_loader):
        anchors = anchors.to(device)
        labels = labels.to(device)
        y_batch = model(anchors)

        if epoch >= 0:
            y_batch = dis_model(y_batch)
        if normalize:
            y_norm = torch.norm(y_batch, p=2, dim=1, keepdim=True)
            y_norm = y_norm.expand_as(y_batch)
            y_batch_data = y_batch / y_norm
        else:
            y_batch_data = y_batch
        y_batches.append(y_batch_data)
        c_batches.append(labels)

    y_data = torch.cat(tuple(y_batches), dim=0).cpu().numpy()
    c_data = torch.cat(tuple(c_batches), dim=0).cpu().numpy()
    return y_data, c_data


def compute_soft_hard_retrieval(distance_matrix, labels, label_batch=None):
    softs = []
    hards = []
    retrievals = []

    if label_batch is None:
        label_batch = labels
    # distance_matrix = cuda.to_cpu(distance_matrix)
    # labels = cuda.to_cpu(labels)
    # label_batch = cuda.to_cpu(label_batch)

    K = 11  # "K" for top-K
    for d_i, label_i in zip(distance_matrix, label_batch):
        top_k_indexes = np.argpartition(d_i, K)[:K]
        sorted_top_k_indexes = top_k_indexes[np.argsort(d_i[top_k_indexes])]
        ranked_labels = labels[sorted_top_k_indexes]
        # 0th entry is excluded since it is always 0
        ranked_hits = ranked_labels[1:] == label_i

        # soft top-k, k = 1, 2, 5, 10
        soft = [np.any(ranked_hits[:k]) for k in [1, 2, 5, 10]]
        softs.append(soft)
        # hard top-k, k = 2, 3, 4
        hard = [np.all(ranked_hits[:k]) for k in [2, 3, 4]]
        hards.append(hard)
        # retrieval top-k, k = 2, 3, 4
        retrieval = [np.mean(ranked_hits[:k]) for k in [2, 3, 4]]
        retrievals.append(retrieval)

    average_soft = np.array(softs).mean(axis=0)
    average_hard = np.array(hards).mean(axis=0)
    average_retrieval = np.array(retrievals).mean(axis=0)
    return average_soft, average_hard, average_retrieval


def lossfun_one_batch(device, model, model_pos_gen, model_neg_gen, dis_model, opt, fea_opt, opt_pos_gen, opt_neg_gen, opt_dis, params, batch, epoch):
    # the first half of a batch are the anchors and the latters
    # are the positive examples corresponding to each anchor
    lambda1 = 1.0
    lambda2 = 1.0
    n_epoch_ = 0
    model.train()

    ancs, poss, negs = batch
    ancs = ancs.split(params.batch_size, dim=0)
    poss = poss.split(params.batch_size, dim=0)
    negs = negs.split(params.batch_size, dim=0)

    total_loss_gen_pos = []
    total_loss_gen_neg = []
    total_loss_m = []


    for i in range(len(ancs)):
        anc = ancs[i].to(device)
        pos = poss[i].to(device)
        neg = negs[i].to(device)

        anc_out = model(anc)  # (N, 512)
        pos_out = model(pos)  # (N, 512)
        neg_out = model(neg)  # (N, 512)

        t_loss = F_tloss(anc_out, pos_out, neg_out, params.alpha)

        # Train model and dis_model
        batch_concat = torch.cat((anc_out, pos_out, neg_out), dim=1)
        g_pos = model_pos_gen(batch_concat)

        batch_concat_g = torch.cat((anc_out, g_pos, neg_out), dim=1)
        fake = model_neg_gen(batch_concat_g) # (N, 512)

        batch_fake = torch.cat((anc_out, pos_out, fake), dim=0)
        #pos_out or g_pos?
        embedding_fake = dis_model(batch_fake)  # (3 * N, 512)
        loss_m = triplet_loss(embedding_fake)
        total_loss_m.append(loss_m)

        if epoch < n_epoch_:
            t_loss.backward()
            fea_opt.step()
        else:
            loss_m.backward()
            opt.step()
            opt_dis.step()

        # train gen_model
        ###############################################
        #neg gen
        batch_concat_g = torch.cat((anc_out.detach(), g_pos.detach(), neg_out.detach()), dim=1)
        fake = model_neg_gen(batch_concat_g)  # (N, 512)
        loss_hard = torch.mean(F.pairwise_distance(fake, anc_out, p=2))
        loss_reg = torch.mean(F.pairwise_distance(fake, neg_out, p=2)) # batch -> neg_out

        batch_fake = torch.cat((anc_out, g_pos, fake), dim=0)
        #pos_out or g_pos?
        embedding_fake = dis_model(batch_fake)  # (3 * N, 512)
        loss_adv = adv_loss(embedding_fake)
        loss_gen = loss_hard + lambda1 * loss_reg + lambda2 * loss_adv

        total_loss_gen_neg.append(loss_gen)

        if epoch >= n_epoch_:
            loss_gen.backward()
            opt_gen.step()

        ##############################################
        #pos gen
        batch_concat = torch.cat((anc_out.detach(), pos_out.detach(), neg_out.detach()), dim=1)
        g_pos = model_pos_gen(batch_concat)
        loss_hard_pos = l2(g_pos, anc_out)
        loss_reg_pos = l2(g_pos, pos_out)
        loss_gen_pos = loss_hard_pos + lambda1 * loss_reg_pos
        total_loss_gen_pos.append(loss_gen_pos)

        if epoch >= n_epoch_:
            loss_gen_pos.backward()
            opt_gen.step()


        model.zero_grad()
        model_pos_gen.zero_grad()
        model_neg_gen.zero_grad()
        dis_model.zero_grad()

    return total_loss_gen_pos, total_loss_gen_neg, total_loss_m


def evaluate(device, model, dis_model, test_loader, epoch, distance='euclidean', normalize=False):
    if distance not in ('cosine', 'euclidean'):
        raise ValueError("distance must be 'euclidean' or 'cosine'.")
    model.eval()
    with torch.no_grad():
        y_data, c_data = iterate_forward(device,
                                         model, dis_model, test_loader, epoch,
                                         normalize=normalize)
    nmi, f1 = evaluate_cluster(y_data, c_data, 98)
    return nmi, f1


def triplet_loss(y, alpha=1.0):
    a, p, n = split_to_three(y)

    return F.triplet_margin_loss(a, p, n, margin=alpha)

    # distance = torch.sum((a - p) ** 2.0, dim=1) - torch.sum((a - n) ** 2.0, dim=1) + alpha
    # return torch.mean(F.relu(distance)) / 2


def adv_loss(y, alpha=1.0):
    return -triplet_loss(y, alpha)
    # a, p, n = split_to_three(y)
    # distance = -torch.sum((a - p) ** 2.0, dim=1) + torch.sum((a - n) ** 2.0, dim=1) - alpha
    #
    # return torch.mean(F.relu(distance)) / 2


# def l2_norm(fake, neg_out):
#     _, _, fake_n = split_to_three(fake)
#     l2 = torch.sum((fake_n - neg_out) ** 2.0, dim=1)
#     return torch.mean(l2)


# def l2_hard(fake, anc_out):
#     _, _, fake_n = split_to_three(fake)
#     l2 = torch.sum((fake_n - anc_out) ** 2.0, dim=1)  # a -> anc_out
#     return torch.mean(l2)


def split_to_three(y, dim=0):
    # split along dim=0 into three pieces
    d = y.size(dim)
    a, p, n = torch.split(y, d // 3, dim=dim)
    return a, p, n