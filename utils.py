import random
import torch
import os
import time

import numpy as np
import pprint as pprint

from torch.utils.data import DataLoader, Sampler
import random
from tqdm import tqdm
from models import *
from collections import OrderedDict, defaultdict
from copy import deepcopy
import itertools


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == -1:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.v = 0
        self.acc_v = 0
        self.n = 0

    def add(self, x, current_n=1):
        # self.acc_v += x * current_n
        self.acc_v += x
        self.n += current_n
        self.v = self.acc_v / self.n

    def item(self):
        try:
            return self.acc_v / self.n
        except ZeroDivisionError:
            return 0.

class Averager_Loss():

    def __init__(self):
        self.v = 0
        self.acc_v = 0
        self.n = 0

    def add(self, x, current_n=1):
        self.acc_v += x * current_n
        # self.acc_v += x
        self.n += current_n
        self.v = self.acc_v / self.n

    def item(self):
        return self.v



class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).sum().item()
    else: # only for gpu setup
        pass


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


class BatchSampler(Sampler):
    def __init__(self, dataset, num_iterations, batch_size):
        super().__init__(None)
        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)), self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations

def batch_iterator(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch
