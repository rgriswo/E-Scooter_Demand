# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:06:13 2023

@author: ryang
"""
import sys
sys.path.insert(0, 'C:\\Users\\ryang\\Desktop\\E_Scooter_Demand\\E-Scooter_Demand\Code') # location of src 
import pickle
import scipy.sparse as sp
import random, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import ConvAutoencoder as conv
from torch.utils.data import Dataset

SCOOTER_DATA = "C:\\Users\\ryang\\Desktop\\E_Scooter_Demand\\E-Scooter_Demand\\Code\\test2.pkl"
MODEL_FILE = 'autoencoder-model-v2.pt'
BATCH_SIZE = 10
EPOCHS = 1

def processor_info(device):
    print(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Number of Devics:', torch.cuda.device_count())
        print('Allocated Memory:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached Memory   :', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    return 

def initiate_loader(dataset, batchsize): 
    transform = transforms.ToTensor()  
    
    #create mask to get %80 of data
    mask1 = np.random.rand(len(dataset)) < 0.8
    
    train_data = scooter_dataset([])
    test_data = scooter_dataset([])
    
    #split dataset into training and test
    for i, mask in enumerate(mask1):
        if mask == True:
            train_data.append(dataset[i])
        else:
            test_data.append(dataset[i])

    # to prepare training loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, num_workers=0)
    # to prepare test loader
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batchsize, num_workers=0)
    
    del train_data
    del test_data
    
    return train_loader, test_loader

class scooter_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):

        return self.data[index].toarray()
    
    def __len__(self):
        return len(self.data)
    
    def append(self, data):
        self.data.append(data)
 
def read_pickle_file(file):
    #read pick file
    objects = []
    with (open(file, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    #retun pickle data
    return objects[0]

if __name__ == "__main__":
    raw_data = read_pickle_file(SCOOTER_DATA)
    
    # to prepare training loader
    train_loader, test_loader = initiate_loader(raw_data, BATCH_SIZE)
    
    #get device
    device = conv.get_device()
    
    processor_info(device)
    
    #create model
    model = conv.ConvAutoencoder(MODEL_FILE,device,raw_data[0].shape)
   
    #release data
    del raw_data

    model.train_model(train_loader, BATCH_SIZE, EPOCHS)


    

    
   
    