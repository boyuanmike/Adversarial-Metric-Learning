from __future__ import absolute_import, print_function
from torchvision.transforms import transforms

import torch
import torch.utils.data as data
from PIL import Image
import os
from collections import defaultdict
import numpy as np
from itertools import combinations


def default_loader(path):
    return Image.open(path).convert('RGB')


class CovertBGR(object):
    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img


def generate_transform_dict(origin_width=256, width=224, ratio=0.16):
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])

    transform_dict = dict()

    transform_dict['rand-crop'] = \
        transforms.Compose([
            CovertBGR(),
            transforms.Resize(origin_width),
            transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['center-crop'] = \
        transforms.Compose([
            CovertBGR(),
            transforms.Resize(origin_width),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['resize'] = \
        transforms.Compose([
            CovertBGR(),
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
            transform = generate_transform_dict()['rand-crop']

        # read txt get image path and labels
        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:
            # img_anon = img_anon.replace(' ', '\t')

            img, label = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_indices[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.label_to_indices = label_to_indices
        self.loader = loader
        self.triplet = triplet

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        fn = os.path.join(self.root, fn)
        img = self.loader(fn)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

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


# Example of using this class
# data = CUB_200_2011(root = path_of_my_data)
# train_loader = torch.utils.data.DataLoader(
#         data.train, batch_size=batch_size,
#         sampler=some_sampler,
#         drop_last=True, pin_memory=True, num_workers=nThreads)
class CUB_200_2011:
    def __init__(self, root, width=224, origin_width=256, ratio=0.16):
        # Data loading code
        # print('ratio is {}'.format(ratio))
        transform_dict = generate_transform_dict(origin_width=origin_width, width=width, ratio=ratio)

        # root should be the directory with images and train.txt, text.txt
        # example of train.txt can be found at datasets/example

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform_dict['rand-crop'])
        self.test = MyData(root, label_txt=test_txt, transform=transform_dict['center-crop'], triplet=False)


# Example of using this class
# data = Car196(root = path_of_my_data)
# train_loader = torch.utils.data.DataLoader(
#         data.train, batch_size=batch_size,
#         sampler=some_sampler,
#         drop_last=True, pin_memory=True, num_workers=nThreads)
class Car196:
    def __init__(self, root, origin_width=256, width=224, ratio=0.16):
        transform_dict = generate_transform_dict(origin_width=origin_width, width=width, ratio=ratio)

        # root should be the directory with images and train.txt, text.txt
        # example of train.txt can be found at datasets/example

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        self.train = MyData(root, label_txt=train_txt, transform=transform_dict['rand-crop'])
        self.test = MyData(root, label_txt=test_txt, transform=transform_dict['center-crop'], triplet=False)


# def testCar196():
#     data = Car196()
#     print(len(data.test))
#     print(len(data.train))
#     print(data.train[1])
#
#
# def testCUB_200_2011():
#     print(CUB_200_2011.__name__)
#     data = CUB_200_2011()
#     print(len(data.test))
#     print(len(data.train))
#     print(data.train[1])


# loader = torch.utils.data.DataLoader(my_data.train, batch_sampler=sampler)
class BalancedBatchSampler:
    """
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, label_to_indices, n_classes, n_samples):
        self.label_to_indices = label_to_indices
        for label in self.label_to_indices:
            np.random.shuffle(self.label_to_indices[label])

        self.used_label_indices_count = defaultdict(int)
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = sum((len(x) for x in self.label_to_indices.values()))
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(list(self.label_to_indices.keys()), self.n_classes, replace=False)
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
    images, labels = batch
    image_clusters = torch.chunk(images, n_class)  # tuple
    # unique_labels = torch.unique(torch.stack(torch.chunk(labels, n_class)), dim=1).reshape(-1)

    triplets = []
    for index, images in enumerate(image_clusters):
        for anc_pos_pair in combinations(images, 2):
            neg_class = np.random.randint(n_class)
            while neg_class == index:
                neg_class = np.random.randint(n_class)
            neg_idx = np.random.randint(n_samples)
            triplets.append(anc_pos_pair + (image_clusters[neg_class][neg_idx],))

    triplet_batch = list(zip(*triplets))

    anchors, positives, negtives = tuple(map(torch.stack, triplet_batch))  # torch.Size([980, 3, 224, 224])
    return anchors, positives, negtives
