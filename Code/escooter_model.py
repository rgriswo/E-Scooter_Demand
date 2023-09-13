# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 18:14:52 2023

@author: ryang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:24:24 2023

@author: ryang
"""

import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy import sparse
import pandas as pd
import seaborn as sns
from datetime import datetime
import random
from math import prod
import sys

config = None
TRAINING_SIZE = .9
MIN = 0
MAX = 0

def unnormalize_grid(grid, MIN, MAX):

    grid = torch.round((grid * (MAX - MIN)) + MIN)
    
    return grid
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
    def __init__(self, model_config, input_size = 0, model_file=None, device=None, len_extra_features = 0):

        super(e_scootermodel, self).__init__()
        self.model_file = model_file if model_file is not None else 'escooter_model-v100.pt'
        self.device = device if device is not None else get_device()
        
        
        self.future_predict = model_config["future_size"]
        self.hidden_size = model_config["hidden_size"]
        self.num_layers = model_config["num_layers"] 
        self.input_size = model_config["input_size"]
        self.rnn_type = model_config["rnn_type"]
        
        #auto encode and decode settings
        grid_size = model_config["grid_size"]
        in_channels = model_config["in_channels"]
        channel1 = model_config["channel1"]
        enc_padding = model_config["enc_padding"]
        enc_channels = model_config["enc_channels"]
        enc_kernels = model_config["enc_kernels"]
        dec_kernels = model_config["dec_kernels"]
        pool_kernels = model_config["pool_kernels"]
        pool_stride = model_config["pool_stride"]
        learning_rate = model_config["pool_stride"]
        unflatten_dim = model_config["unflatten_dim"]
        output_padding1 = model_config["output_padding1"]
        output_padding2 = model_config["output_padding2"]
        
        
        #calculate the total number of features that need to be outputed from linear layer
        number_cells = prod(grid_size)
        
        #because of decode size it currenly assiens 2 extra collums when decoding
        output_grid = grid_size.copy()
        output_grid[2] +=  2
        
        output_cells = prod(output_grid)
        
        # Encoder
        self.encode = nn.Sequential (
            torch.nn.Flatten() , #flatten the input to be inputed into the linear layer
            torch.nn.Linear(input_size , number_cells), #shrink the number of cell for new grid size
            torch.nn.Unflatten(1, grid_size), # unflatten cells into grid size
            nn.Conv2d(in_channels, channel1, enc_kernels, padding=enc_padding), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stride),                   # kernel_size, stride
            nn.Conv2d(channel1, enc_channels, enc_kernels, padding=enc_padding),# in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stride),                    # kernel_size, stride
            torch.nn.Flatten()                                          #flatten model
        )
        #Decoder
        self.decode = nn.Sequential(                            #unflatten model
            torch.nn.Unflatten(1, unflatten_dim), 
            nn.ConvTranspose2d(enc_channels, channel1, dec_kernels, stride=pool_stride, padding=enc_padding, output_padding=output_padding1), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.ConvTranspose2d(channel1, in_channels, dec_kernels, stride=pool_stride, padding=enc_padding, output_padding=output_padding2),  # in_channels, out_channels, kernel_size
            torch.nn.Flatten(), #flatten output to be inputed into linear layer
            torch.nn.Linear(output_cells, input_size), #reconstruct original grid size

        )
        
        if(self.rnn_type == "LSTM"):
            #input to lSTM will be the number of laten featurese plus size of extra features such as time of day and so on
            self.rnn = torch.nn.LSTM(self.input_size + len_extra_features, self.hidden_size, num_layers=self.num_layers,
                                      batch_first=True)
        elif (self.rnn_type == "GRU"):
            self.rnn = torch.nn.GRU(self.input_size + len_extra_features, self.hidden_size, num_layers=self.num_layers,
                                      batch_first=True)
        elif (self.rnn_type == "RNN"):
            self.rnn = torch.nn.RNN(self.input_size + len_extra_features, self.hidden_size, num_layers=self.num_layers,
                                      batch_first=True)
        else:
            raise Exception("RNN Type not Suported please try LSTM, GRU, or RNN")
            
        
        #Linear layer to reconstruct output of LSTM into input size for auto encoder 
        self.linear = torch.nn.Linear(self.hidden_size, self.input_size)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_list = [] 
        self.to(self.device)
        
        if self.model_file is not None:
            self.load_model(self.model_file)
        return 
    
    def forward(self, x, time_features, future=1):  
        n_samples = len(x)
        
        
        if(self.rnn_type == "LSTM"):
            #LSTM hidden states init 0
            h1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
            c1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
            hidden_state = (h1, c1)  
        else:
            #RNN and GRU hidden states init 0
            hidden_state = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
        
        
        
         
        # input:    batchsizexinputizexgridsizexgridsize
        batch_size = x.shape[0]
        window_size = x.shape[1]
        channel_size = x.shape[2]
        
        data_size_x =   x.shape[3]
        data_size_y =   x.shape[4]
        
        #reshape input from LSTM window size into array for autoencoder
        x = torch.reshape(x,(batch_size * window_size,
                             channel_size,
                             data_size_x,
                             data_size_y))
        
        x = self.encode(x)

        #reshape:
        x = torch.reshape(x,(batch_size,
                             window_size,
                             self.input_size))

        #append the time features 
        x = torch.cat((x,time_features[:,:window_size]),-1)
        
        
        output, hidden_state = self.rnn(x, hidden_state) #predict: 
        
        output = self.linear(output)                      #reshape:       
        
        #get LSTM output
        output = output[:,-1:]
        
        #get time fearures for output 
        output_time_features = time_features[:,len(time_features[0])-future]
        
        #reshape to match output
        output_time_features = torch.reshape(output_time_features, (batch_size, 
                                                                    1 ,
                                                                    len(time_features[0][0])))
                      
        #add time features to output
        output_time_features = torch.cat((output_time_features , output),-1)
        #create new feat using output
        x = torch.cat([x[:,1:], output_time_features[:,-1:]], dim=1)
        
        
        # the last output is fed as an input to the prediction             
        for i in range(1,future):
            pred, hidden_state = self.rnn(x, hidden_state)
            
            pred = self.linear(pred)
            
            #store newly predicted output
            output = torch.cat((output, pred[:,-1:]), dim=1)
            
            #get the timefeatures for next predicted output
            output_time_features = time_features[:,len(time_features[0])-future+i]
            
            #reshape to match output
            output_time_features = torch.reshape(output_time_features, (batch_size, 
                                                                        1 ,
                                                                        len(time_features[0][0])))
            
            #add the time features to the new predicted out put
            output_and_timefeatures = torch.cat((output_time_features , pred[:,-1:]),-1)
            
            #update input for next prediction with new output
            x = torch.cat((x[:,1:], output_and_timefeatures), dim=1)

        
        
        #get predicted outputs
        x = output
        
        
        
        #reshape: output to inputsXinputsize
        x = torch.reshape(x,((batch_size*future),
                              self.input_size))
        
        x = self.decode(x)

        #reshape: output after being decoded to BatchsizeXFutureXChannelXData_size_xXData_size_Z
        x = torch.reshape(x,(batch_size,
                             future,
                             channel_size,
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
            for i, (features, labels, time_features) in tqdm(enumerate(loader),unit="batch", total=len(loader)):
             
                size = len(features)
                
                #push features/time_features and label to gpu
                labels = labels.to(self.device)
    
                features = features.to(self.device)
                
                time_features = time_features.to(self.device)
                
                #Get prediction from model
                pred = self(features, time_features, self.future_predict)
                
                #calculate loss
                loss = self.criterion(pred, labels)
                
                del labels, pred
                
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
            
            z = 0
            
            for input_data, labels, time_features in itr:
                if z == 20:
                    break
                else:
                    z +=1
                    
            time_features = time_features.to(self.device)
            input_data = input_data.to(self.device)
            
            
            pred = self(input_data, time_features, future=self.future_predict)
            
            
            pred = pred.to('cpu')
            
            pred = torch.round(pred)
            
            end_pred = pred[0][0][1]
            end_labels = labels[0][0][1]
            
            pred = pred[0][0][0]
            labels = labels[0][0][0]
            
            print("predicted")
            print(sparse.csr_matrix(pred))
            
            print("actual")
            print(sparse.csr_matrix(labels))
            
            
            create_heatmap(labels, config.CITY + " Start Actual (First Hour)")
            
            create_heatmap(pred, config.CITY + " Start Pred (First Hour)")
            
            create_heatmap(end_labels, config.CITY + " End Actual (First Hour)")
            
            create_heatmap(end_pred, config.CITY + " End Pred (First Hour)")
            
            plt.figure()
            plt.title("Loss")
            plt.ylabel("loss")
            plt.xlabel("Epoch")
            plt.plot(self.loss_list)
            
            start_pred_demand = []
            start_label_demand = []
            
            end_pred_demand = []
            end_label_demand = []
            
            start_loss_plt = {}
            end_loss_plt = {}
            
            for i in range(self.future_predict):
                start_loss_plt[i] = 0
                end_loss_plt[i] = 0

            for i, (feature, label, time_features) in tqdm(enumerate(data),unit="batch", total=len(data)):
                #push feature and label to gpu
                
                label   = label.to(self.device)
                
                feature = feature.to(self.device)
                
                time_features = time_features.to(self.device)
                
                pred = self(feature, time_features, future=self.future_predict)
                    
                
                loss = self.criterion(pred, label)
                
                pred = pred.to('cpu')
                label = label.to('cpu')
                
                #go through each future prediciton and calculate the loss
                for j in range(self.future_predict):

                    start_loss_plt[j] += self.criterion(pred[:,j,0,:,:], label[:,j,0,:,:])
                    end_loss_plt[j] += self.criterion(pred[:,j,1,:,:], label[:,j,1,:,:])
                
                #round the predictions
                pred = torch.round(pred)
                
                #grab the first hour predicted
                start_pred = pred[0][0][0]
                start_label = label[0][0][0]
                
                end_pred = pred[0][0][1]
                end_label = label[0][0][1]
                
                #append the sum of the first hour of prediction to array to be 
                #used for graphing later
                start_pred_demand.append(sparse.csr_matrix(start_pred).sum())
                start_label_demand.append(sparse.csr_matrix(start_label).sum())
                
                end_pred_demand.append(sparse.csr_matrix(end_pred).sum())
                end_label_demand.append(sparse.csr_matrix(end_label).sum())
            
            
            create_total_demand_chart(start_pred_demand, 
                                      start_label_demand,  
                                      config.CITY + " Total Start Demand Actual Vs Prediced (Only Predicting First Hour)")
            

            create_total_demand_chart(end_pred_demand, 
                                      end_label_demand,  
                                     config.CITY + " Total End Demand Actual Vs Prediced (Only Predicting First Hour)")
            
            
            #calculate the averge loss for each future preiction
            average_start_loss =  np.array(list(start_loss_plt.values())) / len(data)
            
            average_end_loss = np.array(list(end_loss_plt.values())) / len(data)
            
            create_loss_chart(start_loss_plt.keys(),average_start_loss,
                              config.CITY + " Average Start Loss")

            create_loss_chart(end_loss_plt.keys(),average_end_loss,
                              config.CITY + " Average End Loss")
            
            create_loss_chart(start_loss_plt.keys(),start_loss_plt.values(),
                              config.CITY + " Total Start Loss (" + str(len(data)) +" samples)")
            
            create_loss_chart(end_loss_plt.keys(),end_loss_plt.values(),
                              config.CITY + " Total End Loss (" + str(len(data)) +" samples)")

            print("test loss =", loss.item())
            
            
            
            
        return 


def create_heatmap(data, title):
    plt.figure()
    g = sns.heatmap(data, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title(title)
    g.axes.set_ylim(0,max(data.shape))
    plt.savefig(title)


def create_total_demand_chart(pred_data, label_data,  title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Hours')
    plt.ylabel('Demand')
    plt.plot(pred_data)
    plt.plot(label_data)
    plt.legend(['Pred','Actual'])
    plt.savefig(title)


def create_loss_chart(xlabel, ylabel, title):
    plt.figure()
    plt.title(title)
    plt.xlabel('Hours In the Future')
    plt.ylabel('Total MSE Loss')
    plt.bar(xlabel, ylabel)
    plt.savefig(title)
    
    
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




class scooter_Dataset(Dataset):
    def __init__(self, data, data_format):
        self.data = data
        self.data_format = data_format
    def __len__(self):
        #subtrace window size and 1 for the last input which will not have a label
        return len(self.data)
        
    def __getitem__(self, idx):  
        #get data returned by raw_dataset
        features, labels, times_label = self.data[idx]

        #get future number of inputs starting from after features end
        labels = labels

        #get time_features for future number of inputs starting from after features end
        time_features = [extract_time_features(i,self.data_format) for i in times_label]
        
        return features, labels, time_features
    
    def get_label_times(self, idx):
        features, labels, times_label  = self.data[idx]
        
        return times_label

#this function controls how data is extraced from the scooter_Dataset
def csr_collate(batch):
    #create an empty list for features, labels, and time_features
    features = []
    labels = []
    time_features = []
    
    #iterate though the batch each batch contains the information returned
    #by __getitem__ in scooter_Dataset

    for i in range(len(batch)):
        features.append(batch[i][0])
        labels.append(batch[i][1])
        time_features.append(batch[i][2])
    return torch.Tensor(np.array(features,dtype = np.single)).float(), torch.Tensor(np.array(labels,dtype = np.single)), torch.Tensor(np.array(time_features))

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
        
    def __getitem__(self, idx):  
        
        #Turn csr matrix notation into array notation for torch tensor
        #get window number of inputs starting from index
        features = list(np.stack((self.data[i][0].toarray(), self.data[i][1].toarray())) for i in range(idx, idx+self.window))
        
        #get future number of inputs starting from after features end
        label = list(np.stack((self.data[i][0].toarray(), self.data[i][1].toarray())) for i in range((idx+self.window),idx+self.window+self.future))
        
        #get time_features for the output to be predicted
        times_features = list(self.data[i][2][0] for i in range(idx, idx+self.window+self.future))
        
        return features, label, times_features
    
        

    
def initiate_loader(file, batchsize, window, furure_size, train_size): 
    raw_data = read_pickle_file(file)
    
    #get training dataset size 
    train_size = int(train_size * len(raw_data))
    
    #create dataset for raw data 
    train_raw_dataset = raw_Dataset(raw_data[:train_size], window, furure_size)
    test_raw_dataset = raw_Dataset(raw_data[train_size:], window, furure_size)
    
    
    #tranform raw dataset to scooter dataset 
    #the scooter dataset handles the data is diffrently since the data is 
    #no longer in order after the split
    train_dataset = scooter_Dataset(train_raw_dataset, config.data_format)
    test_dataset  = scooter_Dataset(test_raw_dataset, config.data_format)
    
    # to prepare training loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=4, collate_fn=csr_collate, shuffle = True)
    # to prepare test loader
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1, num_workers=2, collate_fn=csr_collate)
    
    del train_dataset, test_dataset, raw_data

    
    return train_loader, test_loader

def extract_time_features(str_date, data_format):

    #create datetime from string %Y-%m-%dT%H:%M:%SZ"
    #"%Y-%m-%dT%H:%M"
    
    date = datetime.strptime(str_date, data_format)
    
    #return the week days in one hot encode same with month
    #also return the time of the data which is normalized 
    #combine all of the information into one list to be fed into the 
    #model
    return (list(createOneHotEncoding(range(7),date.weekday())) +
            list(createOneHotEncoding(range(1,13),date.month)) +
            [normalize_data(date.time().hour,0,23)])

def find_outliers(demand_matrixs):
    demands = []
    
    #create list of all cells that contain a demand value
    for matrix in demand_matrixs:
        for cell in matrix.data:
            demands.append(cell)
    
    #sort demands list
    demands = np.sort(demands)
    
    #get the first 99.99% demand points
    filtered_demands = demands[:round(len(demands)*.9999)]
    
    #find outlire values by compareing diffrences in unique values 
    outliers = list(set(np.unique(demands)) - set(np.unique(filtered_demands)))
    
    print("outliers: " + str(np.sort(outliers)))
    
    
    return outliers
    
def remove_outliers(data):
    #find outlires in the data from the matrixs
    outliers = find_outliers([i[0] for i in data])
    
    #iterate through each matrix
    for matrix, time_features in data:
        
        #get the cells that contain an outlier
        mask = [i in outliers for i in matrix.data]
        
        #remove the outlier by assigning it 0 
        matrix.data[mask] = 0  
        
    return data

def normalize_data(data, MIN, MAX):
    
    data = (data - MIN)/(MAX - MIN)
    
    return data

def createOneHotEncoding(uniqueValues,data):
    #create a tuple for each value filled with 0s
    arr = np.zeros(len(uniqueValues),dtype = int)
    #assign 1 for that value
    arr[uniqueValues.index(data)] = 1
    return arr

def normalize_trip_db(db): 
    global MAX, MIN
    
    #find max of all matrixs
    for matrix, i in db:
        if len(matrix.data) > 0 and MAX < max(matrix.data):
            MAX = max(matrix.data)
            
    #normalize the cells in every matrix
    for matrix, i in db:
        matrix.data = normalize_data(matrix.data, MIN, MAX)

def import_config(config_name):
    global config
    config = __import__(config_name)
    return config
    
if __name__ == "__main__":   
    def main():
        
        argumentList = sys.argv[1:]
        
        for i in argumentList:
            
            #import config file
            import_config(i)
            
            #get device
            device = get_device()
            
            #empty memory
            torch.cuda.empty_cache()
            train_loader, test_loader = initiate_loader(config.SCOOTER_DATA, 
                                                        config.BATCH_SIZE, 
                                                        config.WINDOWSIZE, 
                                                        config.MODEL_CONFIG["future_size"],
                                                        TRAINING_SIZE)
            
            
            processor_info(device)
            
            #get the size of the input data
            input_size = np.prod(next(iter(train_loader))[0][0][0].shape)
            
            #create model
            model = e_scootermodel(model_file     = config.MODEL_FILE,
                                   model_config = config.MODEL_CONFIG,
                                   input_size = input_size,
                                    device         = device,
                                    len_extra_features = 20)
               
            
            model.train_model(train_loader, config.BATCH_SIZE, config.EPOCHS)
            
            model.eval_model(test_loader, config.BATCH_SIZE)
    main()
    
    
   