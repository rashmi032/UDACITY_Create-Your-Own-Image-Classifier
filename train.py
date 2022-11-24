''' 1. Train
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
''' 

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

import json

import time

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from torch.autograd import Variable

import argparse

import functions_train
import functions_predict

parser = argparse.ArgumentParser(description = 'Train.py')

parser.add_argument('data_dir', nargs = '*', action = 'store', default = './flowers/')
parser.add_argument('--gpu', dest = 'gpu', action = 'store', default = 'gpu')
parser.add_argument('--save_dir', dest = 'save_dir', action='store', default = './checkpoint.pth')
parser.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.01)
#parser.add_argument('--dropout', dest = 'dropout', action = 'store', default = 0.5)
parser.add_argument('--epochs', dest = 'epochs', action = 'store', type = int, default = 10)
parser.add_argument('--arch', dest = 'arch', action = 'store', default = 'vgg16', type = str)
parser.add_argument('--hidden_units', type = int, dest = 'hidden_units', action = 'store', default = 120)

parser = parser.parse_args()
data_dir = parser.data_dir
path = parser.save_dir
lr = parser.learning_rate
arch = parser.arch
#dropout_probability = parser.dropout
middle_features = parser.hidden_units
use = parser.gpu
epochs = parser.epochs


train_loader, validate_loader, test_loader, train_data = functions_train.load_data(data_dir)
# def load_data(data_dir = 'flowers'):

model, criterion, optimizer = functions_train.build_model(arch, middle_features, lr, use)
# def build_model(arch = 'vgg16', middle_features = 1024, learning_rate = 0.01, device = 'gpu'):

functions_train.train_model(model, criterion, optimizer, validate_loader, train_loader, use, epochs)
# def train_model(model, criterion, optimizer, validate_loader, train_loader, use = 'gpu', epochs = 10):

functions_train.save_checkpoint(model, optimizer, train_data, arch, path, middle_features, lr, epochs)
# def save_checkpoint(model, optimizer, train_data, arch = 'vgg16', path = 'checkpoint.pth', input_features = 25088, middle_features = 1024, output_number = 102, lr = 0.01, epochs = 10, batch_size = 64):

print("Model is trained") 