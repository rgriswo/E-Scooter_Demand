# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:24:24 2023

@author: ryang
"""
import sys
sys.path.insert(0, 'C:\\Users\\ryang\\Desktop\\E_Scooter_Demand\\E-Scooter_Demand\Code') # location of src 
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm



SCOOTER_DATA = "C:\\Users\\ryang\\Desktop\\E_Scooter_Demand\\E-Scooter_Demand\\Code\\e_scooter_high_demand.pkl"
MODEL_FILE = 'escooter_model-v2.pt'
BATCH_SIZE = 11
EPOCHS = 100
WINDOWSIZE = 12
HIDDENSIZE = 500
NUMLAYERS = 2
FUTURESIZE = 2
TRAINING_SIZE = .8
INPUTSIZE = 1800

def get_device(default="cuda"):
    if torch.cuda.is_available() and default=="cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device: ", device)
    return device

def processor_info(device):
    print(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Number of Devics:', torch.cuda.device_count())
        print('Allocated Memory:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached Memory   :', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    return 


class e_scootermodel(nn.Module):
    def __init__(self, model_file=None, device=None, hidden_size=51, num_layers=2, input_size=19, future_predict = 1):
        super(e_scootermodel, self).__init__()
        self.model_file = model_file if model_file is not None else 'escooter_model-v100.pt'
        self.device = device if device is not None else get_device()
        
        self.future_predict = future_predict
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.input_size = input_size
        
        #auto encode and decode settings
        in_channels = 1
        channel1 = 16
        enc_padding = 3
        enc_channels = 2
        enc_kernels = 3
        dec_kernels = 2
        pool_kernels = 2
        pool_stride = 4
        learning_rate = 0.0001
        
        # Encoder
        self.encode = nn.Sequential (
            nn.Conv2d(in_channels, channel1, enc_kernels, padding=enc_padding), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stride),                   # kernel_size, stride
            nn.Conv2d(channel1, enc_channels, enc_kernels, padding=enc_padding),# in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stride),                    # kernel_size, stride
            torch.nn.Flatten()                                          #flatten model
        )
        #Decoder
        self.decode = nn.Sequential(
            torch.nn.Unflatten(1, (2, 30, 30)),                              #unflatten model
            nn.ConvTranspose2d(enc_channels, channel1, dec_kernels, stride=pool_stride, padding=3, output_padding=3), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.ConvTranspose2d(channel1, in_channels, dec_kernels, stride=pool_stride, padding=3, output_padding=0),  # in_channels, out_channels, kernel_size
            nn.Sigmoid()
        )
        
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                                  batch_first=True)
        
        self.linear = torch.nn.Linear(self.hidden_size, self.input_size)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_list = [] 
        self.to(self.device)
        
        if self.model_file is not None:
            self.load_model(self.model_file)
        return 
    
    def forward(self, x, future=1):  
        n_samples = x.size(0)
        
        h1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
        c1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
        #LSTM hidden states init 0
        
        hidden_state = (h1, c1)  
         
        # input:    batchsizexinputizexgridsizexgridsize
        #           11x12x452x452
        batch_size = x.shape[0]
        input_size = x.shape[1]
        data_size_x =  x.shape[2]
        data_size_y =  x.shape[3]
        
        #reshape: 60x1x452x452
        x = torch.reshape(x,((batch_size*input_size),
                              1,
                              data_size_x,
                              data_size_y))
        
        x = self.encode(x)          # encoded:  132x1800
        
        encoded_size = x.shape[1]
        
        #reshape: 11x12x1800
        x = torch.reshape(x,(batch_size,
                             input_size,
                             encoded_size))
        
        
        
        output, hidden_state = self.lstm(x, hidden_state) #predict: 11x12x50
        
        output = self.linear(output)                      #reshape: 11x12x5202       
        
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
        
        
        #get predicted outputs 11x2x50
        x = feat[:,-future:]
        
        del pred, feat, output
        
        #reshape: 22x1800
        x = torch.reshape(x,((batch_size*future),
                              encoded_size))
        
        x = self.decode(x)          # decoded:  22x1x452x452
        
        #reshape: 11x2x452x452
        x = torch.reshape(x,(batch_size,
                             future,
                             data_size_x,
                             data_size_y))
        
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
            for i, (features, labels) in tqdm(enumerate(loader),unit="batch", total=len(loader)):
                size = features.size(0)
                #push feature and label to gpu
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                #Get prediction from model
                pred = self(features.float(),self.future_predict)
                
                #calculate loss
                loss = self.criterion(pred, labels)
                del features, labels, pred
                #empty memory
                torch.cuda.empty_cache()
                #zero at grad
                self.optimizer.zero_grad()
                
                #adjust weigths
                loss.backward()  
                self.optimizer.step()
                
                train_loss += loss.item()*size
            
            train_loss = train_loss/len(loader)
            self.loss_list.append(train_loss)
            print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch+1+epochs_done, epochs_togo, train_loss),
                  flush=True)
            
            self.save_model(self.model_file)
        
        self.train(False)
        return   

    def eval_model(self, data, window_size=20, pos=0):
        self.eval()
        
        
        with torch.no_grad():    # disabled gradient calculation
        
            itr = iter(data)
            
            feature, labels = next(itr)
            
            input_data = torch.reshape(feature[0],   (1,
                                                      feature.shape[1],
                                                      feature.shape[2],
                                                      feature.shape[2]))
            
            input_data = (input_data).to(self.device)
            
            pred = self(input_data, future=self.future_predict)
            
            pred = pred[-1][-1]
            pred = pred.to('cpu')
            labels = labels[0][-1]
                
            plt.imshow(labels, cmap='hot', interpolation='nearest')
            plt.show()
        
            plt.imshow(pred, cmap='hot', interpolation='nearest')
            plt.show()
            for i, (feature, label) in tqdm(enumerate(data),unit="batch", total=len(data)):
                #push feature and label to gpu
                feature = feature.to(self.device)
                
                label   = (label).to(self.device)
    
                
                pred = self(feature, future=self.future_predict)
                    
                
                loss = self.criterion(pred, label)
                
            print("test loss =", loss.item())
            
            
            
            
        return 
    
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

# class scooter_Dataset(Dataset):
#     def __init__(self, data, window=20, future = 1):
#         self.window =window
#         self.data = data
#         self.future = future
        
#         if len(self.data) < window:
#             print("Window size is to big for data")
#             sys.exit()
        
#     def __len__(self):
#        #subtrace window size and 1 for the last input which will not have a label
#        return len(self.data) - self.window - self.future
        
#     def __getitem__(self, idx):  
        
#         #Turn csr matrix notation into array notation for torch tensor
#         #get window number of inputs starting from index
#         features = torch.Tensor(np.array(list(self.data[i].toarray() for i in range(idx, idx+self.window))))
        
#         #get future number of inputs starting from after features end
#         label = torch.Tensor(np.array(list(self.data[i].toarray() for i in range((idx+self.window),idx+self.window+self.future))))
        
#         return features, label

class scooter_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        #subtrace window size and 1 for the last input which will not have a label
        return len(self.data)
        
    def __getitem__(self, idx):  
        features, labels = self.data[idx]
        #Turn csr matrix notation into array notation for torch tensor
        #get window number of inputs starting from index
        features = torch.Tensor(np.array(list(i.toarray() for i in features)))
        
        #get future number of inputs starting from after features end
        label = torch.Tensor(np.array(list(i.toarray() for i in labels)))
        
        return features, label

class raw_Dataset(Dataset):
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
        
    def __getitem__(self, idx, to_array=False):  
        
        #Turn csr matrix notation into array notation for torch tensor
        #get window number of inputs starting from index
        features = list(self.data[i] for i in range(idx, idx+self.window))
        
        #get future number of inputs starting from after features end
        label = list(self.data[i]for i in range((idx+self.window),idx+self.window+self.future))
        
        return features, label
    
        

    
def initiate_loader(file, batchsize, window, train_size): 
    raw_data = read_pickle_file(file)
    
    raw_dataset = raw_Dataset(raw_data, WINDOWSIZE, FUTURESIZE)

    train_size = int(train_size * len(raw_dataset))
    test_size = len(raw_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(raw_dataset, [train_size, test_size])
    
    train_dataset = scooter_Dataset(train_dataset)
    test_dataset  = scooter_Dataset(test_dataset)
    
    # to prepare training loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=2)
    # to prepare test loader
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batchsize, num_workers=2)
    
    del train_dataset, test_dataset, raw_data

    
    return train_loader, test_loader


if __name__ == "__main__":
    #empty memory
    torch.cuda.empty_cache()
    
    train_loader, test_loader = initiate_loader(SCOOTER_DATA, BATCH_SIZE, WINDOWSIZE, TRAINING_SIZE)
    
    
    #get device
    device = get_device()
    
    processor_info(device)
    
    #create model
    model = e_scootermodel(model_file     = MODEL_FILE,
                           device         = device,
                           hidden_size    = HIDDENSIZE,
                           num_layers     = NUMLAYERS,
                           input_size     = INPUTSIZE,
                           future_predict = FUTURESIZE)
   

    #model.train_model(train_loader, BATCH_SIZE, EPOCHS)
    
    model.eval_model(test_loader, BATCH_SIZE)