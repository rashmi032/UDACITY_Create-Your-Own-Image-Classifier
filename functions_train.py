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

# You may have one function for data pre-processing which takes a path as an argument and returns all the loaders.
def load_data(data_dir = './flowers'):
    
    data_dir = str(data_dir).strip('[]').strip("'")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    
    # For all three sets you'll need to normalize the means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225] 
    # data_transforms = 
    
    cropped_size = 224
    resized_size = 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    # random scaling, cropping, and flipping, resized to 224x224 pixels
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(cropped_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stds)])
    
    # resize then crop the images to the appropriate size
    validate_transforms = transforms.Compose([transforms.RandomResizedCrop(cropped_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    
    # resize then crop the images to the appropriate size
    test_transforms = transforms.Compose([transforms.Resize(resized_size), # Why 255 pixels? 
                                          transforms.CenterCrop(cropped_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])
    
    
    # TODO: Load the datasets with ImageFolder
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform = validate_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    image_data = [train_data, validate_data, test_data] 
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    
    batch_size = 60
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    validate_loader = torch.utils.data.DataLoader(validate_data,  batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data)
    
    # train_loader
    
    dataloaders = [train_loader, validate_loader, test_loader]
    return train_loader, validate_loader, test_loader, train_data


# One function to load the pre-trained model, build the classifier and define the optimizer. The function will take the architecture name, hidden units etc.
def build_model(arch = 'vgg16', middle_features = 1024, learning_rate = 0.01, device = 'gpu'):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model. Did you mean vgg16, densenet121 or alexnet?".format(arch))

    for param in model.parameters():
        param.requires_grad = False

    input_features = 25088
    # middle_features = 1024
    output_number = 102
    dropout_probability = 0.5
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_features, middle_features)),
                              ('drop', nn.Dropout(p = dropout_probability)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(middle_features, output_number)),
                              ('output', nn.LogSoftmax(dim = 1))
                               ]))
       
    model.classifier = classifier
                                  
    # learning_rate = 0.01
    # epochs = 10
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr = learning_rate)

    if torch.cuda.is_available() and device == 'gpu':
            model.cuda()
            
    return model, criterion, optimizer



# One function to train the model
def train_model(model, criterion, optimizer, validate_loader, train_loader, use = 'gpu', epochs = 10):
    # TODO: Using the image datasets and the trainforms, define the dataloaders

    #batch_size = 60

    #train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    #validate_loader = torch.utils.data.DataLoader(validate_data,  batch_size = batch_size, shuffle = True)
    #test_loader = torch.utils.data.DataLoader(test_data)

    # train_loader

    #dataloaders = [train_loader, validate_loader, test_loader]
    running_loss = 0
    steps = 0
    #model.to(device)
        
    validation = True
    
    device = torch.device('cuda' if torch.cuda.is_available() and use == 'gpu' else 'cpu')
    model.to(device)
    
    start_time = time.time()
    print('Training starts')

    for epoch in range(epochs):
        training_loss = 0
    
        for images, labels in train_loader:
            
            # Move data tensors to the default device (gpu or cpu)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_probabilities = model(images)
            loss = criterion(log_probabilities, labels)
            loss.backward()
            optimizer.step()
    
            training_loss += loss.item()
        
        print('Epoch {} of {} ///'.format(epoch + 1, epochs), 'Training loss {:.3f} ///'.format(training_loss / len(train_loader)))
        
        if validation == True:
            validation_loss = 0
            validation_accuracy = 0
            model.eval()
    
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in validate_loader:
                    # Move data tensors to the default device (gpu or cpu)
                    images, labels = images.to(device), labels.to(device)
                    
                    log_probabilities = model(images)
                    loss = criterion(log_probabilities, labels)
            
                    validation_loss += loss.item()
           
                    # accuracy calculation
                    probabilities = torch.exp(log_probabilities)
                    top_probability, top_class = probabilities.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
           
            model.train()
            
            print("Validation loss {:.3f} ///".format(validation_loss / len(validate_loader)), "Validation accuracy {:.3f} ///".format(validation_accuracy / len(validate_loader)))

    end_time = time.time()
    print('Training ends')

    training_time = end_time - start_time
    print('Training time {:.0f}m {:.0f}s'.format(training_time / 60, training_time % 60))    
    
    return model



# one function to save the checkpoint file
def save_checkpoint(model, optimizer, train_data, arch = 'vgg16', path = 'checkpoint.pth', input_features = 25088, middle_features = 1024, output_number = 102, lr = 0.01, epochs = 10, batch_size = 64):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'network': 'vgg16',
                'input': input_features,
                'output': output_number,
                'learning_rate': lr,       
                'batch_size': batch_size,
                'classifier' : model.classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}
    
    torch.save(checkpoint, path)
    
