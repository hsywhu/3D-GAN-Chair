from PIL import Image
import argparse
import os
import numpy as np
import torch
# from datasets import MaskFaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from glob import glob
import pdb
import random
from optimModel import optim
from threeDgan import Generator, Discriminator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--generator',
                         type=str,
                         help='Pretrained generator',
                         default='./models/gen_95.pt' )
    parser.add_argument( '--discriminator',
                         type=str,
                         help='Pretrained discriminator',
                         default='./models/dis_95.pt' )
    parser.add_argument( '--batch_size',
                         type=int,
                         default=64 )
    parser.add_argument( '--per_iter_step',
                         type=int,
                         default=1500,
                         help='number of steps per iteration' )
    parser.add_argument('--mesh',
                        type=str,
                        default='./dataset/3d/0.npy',
                        help='path to the input mesh')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    target = np.load(args.mesh)
    target /= 255
    args.batch_size = 1
    o = optim(args)
    target = torch.from_numpy(target)
    target = torch.tensor(target, dtype=torch.float32)
    target = torch.unsqueeze(target, 0)
    target = torch.unsqueeze(target, 0)
    completed = o.doOptim(target)

if __name__ == '__main__':
    main()
