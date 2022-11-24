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


def load_checkpoint(path = '~/workspace/ImageClassifier/checkpoint.pth'):
    checkpoint = torch.load(path)
    
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['network'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    cropped_size = 224
    resized_size = 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_transforms = transforms.Compose([transforms.Resize(resized_size),
                                     transforms.CenterCrop(cropped_size), 
                                     transforms.ToTensor()])
    
    transformed_image = image_transforms(image).float()
    numpy_image = np.array(transformed_image)    
    
    mean = np.array(means)
    std = np.array(stds)
    numpy_image = (np.transpose(numpy_image, (1, 2, 0)) - mean) / std    
    numpy_image = np.transpose(numpy_image, (2, 0, 1))
            
    return numpy_image



def predict(path_to_image, model, topk = 5, use = 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    cuda = torch.cuda.is_available()
    if cuda and use == 'gpu':
        model.cuda()
        #print("GPU")
    else:
        model.cpu()
        #print("CPU")
    
    model.eval()
    #print(path_to_image[0])
    image_to_predict = Image.open(path_to_image[0])
    image = process_image(image_to_predict)
    image = torch.from_numpy(np.array([image])).float()
    #image = Variable(image)
    
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    probs = torch.topk(probabilities, topk)[0].tolist()[0] 
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return probs, label