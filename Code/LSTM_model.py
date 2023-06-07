# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:39:50 2023

@author: ryang
"""

#!/usr/bin/env python
import sys
import os
import torch
#import torch.nn as nn
#import torch.optim as optim
# import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def gpu_info(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        freemem, totalmem = torch.cuda.mem_get_info(0)
        allocated = torch.cuda.memory_allocated(0)
        reserved  = torch.cuda.memory_reserved(0)
        print("\tfree:      %7.3f GB" % (freemem /   1024**3))
        print("\ttotal:     %7.3f GB" % (totalmem /  1024**3))
        print("\tallocated: %7.3f GB" % (allocated / 1024**3))
        print("\tcached:    %7.3f GB" % (reserved  / 1024**3))
        print(torch.cuda.memory_summary())
    return 

class scooter_Dataset(Dataset):
    def __init__(self, data, window=20, future = 1):
        self.window =window
        self.data = data
        self.future = future
        
        if len(self.data) < window:
            print("Window size is to big for data")
            sys.exit()
        
    def __len__(self):
       #subtrace window size and 1 for the last input which will not have a label
       return len(self.data) - self.window - self.future
        
    def __getitem__(self, idx):  
        features = torch.Tensor(self.data[idx:idx+self.window])
        label = torch.Tensor(self.data[(idx+self.window)+1:idx+self.window+self.future+1])
        return features, label
             
class PredictModel(torch.nn.Module):
    def __init__(self, model_file=None, hidden_size=51, num_layers=2, input_size=19, future_predict = 1, dev=None):
        super(PredictModel, self).__init__()
        
        self.model_file = model_file if model_file is not None else 'LSTM-v100.pt'
        self.future_predict = future_predict
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.input_size = input_size
        
        #x -> batch_size, seq, input_size
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                                  batch_first=True)
        
        self.linear = torch.nn.Linear(self.hidden_size, self.input_size)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, 
                                          weight_decay=0)   
        self.loss_list = []
        
        #load gpu
        if dev is not None:
            self.device = torch.device(dev)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")           
        
        self.to(self.device)
        
        if self.model_file is not None:
            self.load_model(self.model_file)
            
        return 
    
    def forward(self, x, future=1):
        n_samples = x.size(0)
        
        h1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
        c1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)

        hidden_state = (h1, c1)  
            
        output, hidden_state = self.lstm(x, hidden_state)
        output = self.linear(output)
        
        #get LSTM output
        output = output[:,-1:]
        
        #create new feat using output
        feat = torch.cat([x[:,1:], output[:,-1:]], dim=1)
        
        # the last output is fed as an input to the prediction             
        for i in range(future-1):
            pred, hidden_state = self.lstm(feat, hidden_state)
            pred = self.linear(pred)
            #update with new last output
            feat = torch.cat([feat[:,1:], pred[:,-1:]], dim=1)
          
        return feat[:,-future:], hidden_state
        #return output
    
    def load_model(self, modelfile):
        if os.path.isfile(modelfile): 
            checkpoint = torch.load(modelfile)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.loss_list = checkpoint['loss']
        else:
            print("File not found: %s" % modelfile, file=sys.stderr)
            print("Starting a new training", file=sys.stderr, flush=True)
        return
    
    def save_model(self, modelfile):
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss_list}, 
                    modelfile)
        return
    
    def train_model(self, data, epochs=1, batchsize=1, window_size = 20):
        self.train()
        print("device:", self.device.type)
        #turn data into data set with window size
        ds = scooter_Dataset(data, window_size, self.future_predict)
        #turn dataset into data loader with batch size
        loader = torch.utils.data.DataLoader(ds, batch_size=batchsize)
        
        epochs_done = len(self.loss_list)
        
        epochs_togo = epochs_done + epochs
        
        epoch_start = len(self.loss_list)
        
        WAIT = "Wait..."
        for epoch in range(epochs):
            for i, (features, labels) in tqdm(enumerate(loader),unit="batch", total=len(loader)):
               
                #push feature and label to gpu
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                #x -> batch_size, seq, input_size
                pred, _ = self(features, future=self.future_predict)
                
                #before -> batch_size, seq, input_size
                #reshaped ->batch_size, input_size
                #pred = pred.reshape(pred.shape[0],pred.shape[2])
                
                #calculate loss
                loss = self.criterion(pred, labels)
                
                #zero at grad
                self.optimizer.zero_grad()
                
                #update weights 
                loss.backward()
                self.optimizer.step()
                
                
            print(' Epoch: {}/{} \tLoss: {:.6f}'.format(epoch+1+epochs_done, epochs_togo, loss.item()), flush=True)
            self.loss_list.append(loss.item()) 
        
        #save model 
        self.save_model(self.model_file)
        #turn off training
        self.train(False)
        return
    
    def eval_model(self, data, window_size=20, pos=0):
        self.eval()
        batchsize = 1
        ds = scooter_Dataset(data, window_size, self.future_predict)
        loader = torch.utils.data.DataLoader(ds, batch_size=batchsize)
        
        pos = pos % len(ds)
        with torch.no_grad():    # disabled gradient calculation
            for i, (feature, label) in enumerate(loader):
                #push feature and label to gpu
                feature = feature.to(self.device)
                
                label   = (label).to(self.device)

                
                pred, _ = self(feature, future=self.future_predict)
                    
                
                loss = self.criterion(pred, label)
                
            print("test loss =", loss.item())
            title = "Predict Pos %d at Epoch %d" % (pos, len(self.loss_list))

            
        return 



def main(epochs=1, device_name=None):
    import time
    t0 = time.time()
    model_file = 'sine-model-v4-101.pt'
    data_file  = 'sine101.pt'    
         
    predictor = PredictModel(ni=100, nwin=150, dev=device_name) 
    predictor.load_model(model_file)    
    predictor.train_model(data_file, epochs, batchsize=10, stride=1)
    predictor.save_model(model_file)
    predictor.eval_model(data_file,125)
#    sineplot.plot_loss(model_file)

    t1 = time.time()
    print("mean exe time: %s sec" % ((t1-t0)/epochs))
          
    return

if __name__ == "__main__":
    import sys
    cnt = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    # main(cnt)
    main(1, "cuda")

# cuda
# mean exe time: 12.345476869742075 sec