# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:40:16 2023

@author: ryang
"""

import sys
sys.path.insert(0, 'C:\\Users\\ryang\\Desktop\\E_Scooter_Demand\\E-Scooter_Demand\Code') # location of src 
import numpy as np
import torch
import LSTM_model
from torch.utils.data import Dataset
import pickle

FILENAME = "C:\\Users\\ryang\\Desktop\\E_Scooter_Demand\\E-Scooter_Demand\\Code\\outputs.pkl"
BATCHSIZE = 32
EPOCH = 1
WINDOWSIZE = 117
HIDDENSIZE = 50
NUMLAYERS = 2
FUTURESIZE = 5
    
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
    return objects

def split_data(dataset, percent = 0.8):
    #create mask to get %80 of data
    mask1 = np.random.rand(len(dataset)) < percent
    
    train_data = []
    test_data = []
    
    #split dataset into training and test
    for i, mask in enumerate(mask1):
        if mask == True:
            train_data.append(dataset[i])
        else:
            test_data.append(dataset[i])
    
    return train_data, test_data

if __name__ == "__main__":
    raw_data = read_pickle_file(FILENAME)
    
    train_data, test_data = split_data(raw_data)
        
    
    model = LSTM_model.PredictModel(hidden_size=HIDDENSIZE, 
                                    num_layers=NUMLAYERS,
                                    input_size=len(train_data[0]),
                                    future_predict = FUTURESIZE,
                                    dev=None)
    
    model.train_model(train_data,EPOCH,BATCHSIZE,WINDOWSIZE)
    
    model.eval_model(test_data,20)
    