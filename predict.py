'''2. Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
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

parser = argparse.ArgumentParser(description = 'predict-file')

parser.add_argument('path_to_image', default = 'paind-project/flowers/test/99/image_07871.jpg', nargs = '*', type = str)
parser.add_argument('checkpoint', default = '/home/workspace/ImageClassifier/checkpoint.pth', nargs = '*', type = str)
parser.add_argument('--top_k', default = 5, dest = "top_k", type=int)
parser.add_argument('--category_names', dest = "category_names", default = 'cat_to_name.json')
parser.add_argument('--gpu', default = "gpu", dest = "gpu")

parser = parser.parse_args()
path_to_image = parser.path_to_image
path_to_checkpoint = parser.checkpoint
topk = parser.top_k
use = parser.gpu


# input_img = parser.path_to_image
# path_to_checkpoint = parser.checkpoint

train_loader, validate_loader, test_loader, train_data = functions_train.load_data()
# def load_data(data_dir = 'flowers'):

model = functions_predict.load_checkpoint(path_to_checkpoint)
# def load_checkpoint(path = 'checkpoint.pth'):


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    # What about process_image?

probabilities = functions_predict.predict(path_to_image, model, topk, use)
# def predict(path_to_image, model, topk = 5, use = 'gpu'):

#print(probabilities)

labels = [cat_to_name[index] for index in probabilities[1]]
probability = np.array(probabilities[0])

#print(labels)
#print(probability)

index = 0
while index < topk:
    print("There is a {} % chance that this photo shows a {}.".format(probability[index] * 100, labels[index]))
    index += 1

#print("End")