from __future__ import absolute_import, print_function
from torchvision.transforms import *

import torch
import torch.utils.data as data
from PIL import Image
import os
from collections import defaultdict
from datasets import transforms
import numpy as np
from itertools import combinations


def default_loader(path):
    return Image.open(path).convert('RGB')


def Generate_transform_Dict(origin_width=256, width=224, ratio=0.16):
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])

    transform_dict = dict()

    transform_dict['rand-crop'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(origin_width),
            transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['center-crop'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(origin_width),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['resize'] = \
        transforms.Compose([
            transforms.CovertBGR(),
            transforms.Resize(width),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ])
    return transform_dict


class MyData(data.Dataset):
    def __init__(self, root, label_txt=None,
                 transform=None, loader=default_loader, triplet=True):

        self.root = root
        # default behavior
        if label_txt is None:
            label_txt = os.path.join(root, 'train.txt')

        if transform is None:
            transform_dict = Generate_transform_Dict()['rand-crop']

        # read txt get image path and labels
        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:
            # img_anon = img_anon.replace(' ', '\t')

            [img, label] = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader
        self.triplet = triplet

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # if self.triplet == False:
    #     fn, label = self.images[index], self.labels[index]
    #     fn = os.path.join(self.root, fn)
    #     img = self.loader(fn)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img, label
    # else:
    #     target_class = self.labels[index]
    #
    #     # pool to choose n_idx
    #     pool = self.classes
    #     pool.remove(target_class)
    #     n_class = np.random.choice(pool)
    #     pool.append(target_class)
    #
    #     # p_idx should not be the same as index
    #
    #     p_idx = np.random.choice(self.Index[target_class])
    #     while p_idx == index:
    #         p_idx = np.random.choice(self.Index[target_class])
    #
    #     n_idx = np.random.choice(self.Index[n_class])
    #     anchor_fn = os.path.join(self.root, self.images[index])
    #     pos_fn = os.path.join(self.root, self.images[p_idx])
    #     neg_fn = os.path.join(self.root, self.images[n_idx])
    #     anchor_img = self.loader(anchor_fn)
    #     pos_img = self.loader(pos_fn)
    #     neg_img = self.loader(neg_fn)
    #     if self.transform is not None:
    #         return self.transform(anchor_img), self.transform(pos_img), self.transform(neg_img)
    #     return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.images)


# Example of using this class
# data = CUB_200_2011(root = path_of_my_data)
# train_loader = torch.utils.data.DataLoader(
#         data.train, batch_size=batch_size,
#         sampler=some_sampler,
#         drop_last=True, pin_memory=True, num_workers=nThreads)
class CUB_200_2011:
    def __init__(self, width=224, origin_width=256, ratio=0.16, root=None, transform=None):
        # Data loading code
        # print('ratio is {}'.format(ratio))
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)

        # root should be the directory with images and train.txt, text.txt
        # example of train.txt can be found at datasets/example
        if root is None:
            root = "/Users/Mike/Desktop/EECS498/Project/data/CUB_200_2011"

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        self.test = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'], triplet=False)


# Example of using this class
# data = Car196(root = path_of_my_data)
# train_loader = torch.utils.data.DataLoader(
#         data.train, batch_size=batch_size,
#         sampler=some_sampler,
#         drop_last=True, pin_memory=True, num_workers=nThreads)
class Car196:
    def __init__(self, root=None, origin_width=256, width=224, ratio=0.16, transform=None):
        # Data loading code

        if transform is None:
            transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)

        # root should be the directory with images and train.txt, text.txt
        # example of train.txt can be found at datasets/example
        if root is None:
            root = '/Users/Mike/Desktop/EECS498/Project/data/car196'

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        self.test = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'], triplet=False)


def testCar196():
    data = Car196()
    print(len(data.test))
    print(len(data.train))
    print(data.train[1])


def testCUB_200_2011():
    print(CUB_200_2011.__name__)
    data = CUB_200_2011()
    print(len(data.test))
    print(len(data.train))
    print(data.train[1])


# loader = torch.utils.data.DataLoader(my_data.train, batch_sampler=sampler)
class BalancedBatchSampler(data.BatchSampler):
    """
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {}
        for l in self.labels_set:
            self.label_to_indices[l] = []
        for i in range(len(labels)):
            l = labels[i]
            self.label_to_indices[l].append(i)

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:

            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                pair = self.label_to_indices[class_][
                       self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                 class_] + self.n_samples]
                indices.extend(pair)
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def generate_random_triplets_from_batch(batch, n_samples, n_class):
    # batch [batch_size,3,244,244]
    batch_size = batch[0].shape[0]
    image_clusters = batch[0].split(n_samples)
    triplets = []
    for i in range(len(image_clusters)):
        anchor_positives = list(combinations(image_clusters[i], 2))
        n_idx = np.random.randint(n_class)
        while n_idx == i:
            n_idx = np.random.randint(n_class)
        negs = image_clusters[n_idx]
        for anchor_positive in anchor_positives:
            r_index = np.random.randint(n_samples)
            triplets.append(anchor_positive + (negs[r_index],))

    anc = []
    pos = []
    neg = []
    for triplet in triplets:
        anc.append(triplet[0])
        pos.append(triplet[1])
        neg.append(triplet[2])

    anc_tensor = torch.stack(anc, 0)
    pos_tensor = torch.stack(pos, 0)
    neg_tensor = torch.stack(neg, 0)

    return anc_tensor, pos_tensor, neg_tensor
