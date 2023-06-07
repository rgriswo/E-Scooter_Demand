# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:44:39 2023

@author: ryang
"""
import torch
import torch.nn as nn
import os, sys
from tqdm import tqdm

def get_device(default="cuda"):
    if torch.cuda.is_available() and default=="cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device: ", device)
    return device

class ConvAutoencoder(nn.Module):
    def __init__(self, model_file=None, device=None, grid_size=None):
        super(ConvAutoencoder, self).__init__()
        self.grid_size = grid_size
        self.model_file = model_file if model_file is not None else 'autoencoder-v100.pt'
        self.device = device if device is not None else get_device()
        in_channels = 1
        channel1 = 16
        enc_padding = 2
        enc_channels = 32
        enc_kernels = 3
        dec_kernels = 2
        pool_kernels = 15
        pool_stide = 15
        learning_rate = 0.0001
        # Encoder
        self.encode = nn.Sequential (
            nn.Conv2d(in_channels, channel1, enc_kernels, padding=enc_padding), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stide),                   # kernel_size, stride
            nn.Conv2d(channel1, enc_channels, enc_kernels, padding=enc_padding),# in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stide),                    # kernel_size, stride
            torch.nn.Flatten()                                          #flatten model
        )
        #Decoder
        self.decode = nn.Sequential(
            torch.nn.Unflatten(1, (32, 314, 314)),                              #unflatten model
            nn.ConvTranspose2d(enc_channels, channel1, dec_kernels, stride=pool_stide, padding=1, output_padding=1), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.ConvTranspose2d(channel1, in_channels, dec_kernels, stride=pool_stide, padding=1, output_padding=1),  # in_channels, out_channels, kernel_size
            nn.Sigmoid()
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_list = [] 
        self.to(self.device)
        
        if self.model_file is not None:
            self.load_model(self.model_file)
        return 
        
        
    def forward(self, x):
           # input:    32x1x2591x2591    
        x = self.encode(x)          # encoded:  32x1x13478432
        print(x.shape)
        x = self.decode(x)          # decoded:  32x1x13478432
        return x  

    def load_model(self, modelfile, device=None):
        if os.path.isfile(modelfile): 
            checkpoint = torch.load(modelfile)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_list = checkpoint['loss']
        else:
            print("File not found: %s" % modelfile, file=sys.stderr)
            print("Starting a new training", file=sys.stderr, flush=True)
        return
    
    def save_model(self, modelfile=None):
        if modelfile is None:
            modelfile = self.model_file
            
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss_list}, 
                    modelfile)
        return
    
    def train_model(self, loader, batch_size, epochs):   
        self.train()
        epochs_done = len(self.loss_list)
        epochs_togo = epochs_done + epochs
        
        for epoch in range(epochs):
            # monitor training loss
            train_loss = 0.0
            #Training
            for i, batch in tqdm(enumerate(loader),unit="batch", total=len(loader)):
                torch.cuda.empty_cache()
                features = batch

                #reshape input from (1,batch size ,input len, input lenth)
                #to (batch size,1,input len, input lenth)
                features = torch.reshape(features,(features.shape[0], 1,
                                         features.shape[1], 
                                         features.shape[2]))
                #print(features.shape)
                #set freatures to device
                features = features.to(self.device)
                
                #Get out put of model
                outputs = self(features.float())
                
                #calculate loss
                loss = self.criterion(outputs, features.float())
                
                #zero at grad
                self.optimizer.zero_grad()
                
                #adjust weigths
                loss.backward()
                
                self.optimizer.step()
                
                train_loss += loss.item()*features.size(0)
                  
            train_loss = train_loss/len(loader)
            self.loss_list.append(train_loss)
            print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch+1+epochs_done, epochs_togo, train_loss),
                  flush=True)
            
        self.save_model(self.model_file)
        
        self.train(False)
        return