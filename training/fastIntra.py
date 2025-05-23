import argparse
import os
from solver import Solver
from data_loader1 import get_loader
from torch.backends import cudnn
import random
import time
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

def main(config):
    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    epoch = 5
    decay_ratio = 0.2
    decay_epoch = int(epoch*decay_ratio)

    config.num_epochs = epoch
    config.lr = 0.0001  #0.0001
    config.num_epochs_decay = decay_epoch
    solver = Solver(config, [0])
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam 
    parser.add_argument('--model_path', type=str, default='./trained_models/10-17-models')
    config = parser.parse_args()
    main(config)
