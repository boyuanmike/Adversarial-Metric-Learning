import copy
import os

import matplotlin.pyplot as plt
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
from ..datasets.dataset import *
from ..models.modified_googlenet import ModifiedGoogLeNet
from ..models.net import Generator, Discriminator

def train(main_script_path, func_train_one_batch, param_dict, savev_distance_matrix=False,):
	script_filename = os.path.splitext(os.path.basename(main_script_path))[0]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config_parser = six.moves.configparser.ConfigParser()
	config_parser.read('config')
	log_dir_path = os.path.expanduser(config_parser.get('logs', 'dir_path'))

	p = utils.Logger(log_dir_path, **param_dict)

	# load data base

	streams = data_provider.get_streams(p.batch_size, dataset = p.dataset, 
		method = p.method, crop_size = p.crop_size)
	stream_train, stream_train_eval, stream_test = streams
	iter_train = stream_train.get_epoch_iterator()

	# construct the model

	model = ModifiedGoogLeNet(p.out_dim, p.normalize_output)
	model_gen = Generator()
	model_dis = Discriminator(512, 512)
	model.