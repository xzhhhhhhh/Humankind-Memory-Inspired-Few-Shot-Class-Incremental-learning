import argparse
import importlib
from utils import *
import torch

MODEL_DIR=None
DATA_DIR = './data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='mini_imagenet', choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=252)
    parser.add_argument('-epochs_base_ft', type=int, default=300)
    parser.add_argument('-epochs_new', type=int, default=30)
    parser.add_argument('-epochs_new_ft', type=int, default=5)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.05)
    parser.add_argument('-schedule', type=str, default='Step', choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=80)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=1)
    parser.add_argument('-base_mode', type=str, default='ft_cos', choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos', choices=['ft_dot', 'ft_cos', 'avg_cos', 'ft_comb', 'ft_euc']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier
    parser.add_argument('-features', type=int, default=512)
    
    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-fraction_to_keep', type=float, default=0.1)
    
    parser.add_argument('-mix_times', type=int, default=4)
    parser.add_argument('-beta', type=float, default=0.5)

    return parser

if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = get_command_line_parser()
    args = parser.parse_args()

    # For incremental learning, get same random +seed that has been used during pretraining step
    # if args.model_dir is not None:
    #    args.seed = int(args.model_dir.split('/')[-2][-9])

    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()