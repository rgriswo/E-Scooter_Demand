# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:24:24 2023

@author: ryang
"""
import sys
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
from escooter_config import SCOOTER_DATA, MODEL_FILE, GRID_DICT
from escooter_config import MODEL_CONFIG, BATCH_SIZE, EPOCHS, WINDOWSIZE
from datetime import datetime

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
    def __init__(self, model_config, model_file=None, device=None, len_extra_features = 0):

        super(e_scootermodel, self).__init__()
        self.model_file = model_file if model_file is not None else 'escooter_model-v100.pt'
        self.device = device if device is not None else get_device()
        
        
        self.future_predict = model_config["future_size"]
        self.hidden_size = model_config["hidden_size"]
        self.num_layers = model_config["num_layers"] 
        self.input_size = model_config["input_size"]
    
        #auto encode and decode settings
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
        self.decode = nn.Sequential(                            #unflatten model
            torch.nn.Unflatten(1, unflatten_dim), 
            nn.ConvTranspose2d(enc_channels, channel1, dec_kernels, stride=pool_stride, padding=enc_padding, output_padding=output_padding1), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.ConvTranspose2d(channel1, in_channels, dec_kernels, stride=pool_stride, padding=enc_padding, output_padding=output_padding2),  # in_channels, out_channels, kernel_size
            nn.Sigmoid()
        )
        
        self.lstm = torch.nn.LSTM(self.input_size + len_extra_features, self.hidden_size, num_layers=self.num_layers,
                                  batch_first=True)
        
        self.linear = torch.nn.Linear(self.hidden_size, self.input_size)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_list = [] 
        self.to(self.device)
        
        if self.model_file is not None:
            self.load_model(self.model_file)
        return 
    
    def forward(self, x, time_features, future=1):  
        n_samples = len(x)
        
        h1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
        c1 = torch.zeros(self.num_layers, n_samples, self.hidden_size, dtype=torch.float32, device=self.device)
        #LSTM hidden states init 0
        
        hidden_state = (h1, c1)  
         
        # input:    batchsizexinputizexgridsizexgridsize
        #           11x12x452x452
        batch_size = len(x)
        input_size = len(x[0])
        data_size_x =  x[0][0].shape[0]
        data_size_y =  data_size_x
        
        encoded_x = []
        
        #iterate though all csr matrix and encode them
        for i in range(batch_size):
            for j in range(input_size):
                
                #convert csr matrx to 2D array
                x_array = torch.Tensor(x[i][j].toarray()).float()
                
                #reashpe data into batchsize, 1dim, then size of grid
                x_array = torch.reshape(x_array,(1,1,data_size_x,data_size_x))
                
                #send data to device 
                x_array = x_array.to(self.device)
                
                #encode data
                encoded_x.append(self.encode(x_array)) # encoded:  132x1800

        #convert from a list of tensors to a tensor of tensors 
        x = torch.stack(encoded_x)
        
        del encoded_x
        
        #reshape: 11x12x1800
        x = torch.reshape(x,(batch_size,
                             input_size,
                             self.input_size))
        
        #append the time features 
        x = torch.cat((x,time_features[:,:len(x[i])]),-1)
        
        output, hidden_state = self.lstm(x, hidden_state) #predict: 11x12x50
        
        output = self.linear(output)                      #reshape: 11x12x5202       
        
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
            pred, hidden_state = self.lstm(x, hidden_state)
            
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

        
        
        #get predicted outputs 11x2x50
        x = output
        
        
        
        #reshape: 22x1800
        x = torch.reshape(x,((batch_size*future),
                              self.input_size))
        
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
            for i, (features, labels, time_features) in tqdm(enumerate(loader),unit="batch", total=len(loader)):
             
                size = len(features)
                #push feature and label to gpu
                labels = labels.to(self.device)
                
                time_features = time_features.to(self.device)
                
                #Get prediction from model
                pred = self(features,time_features,self.future_predict)
                
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
        
        #read grid dict
        df = pd.read_csv(GRID_DICT,sep=',')
        grid_dict = {}
        
        for i, row in df.iterrows():
            grid_dict[row[df.columns[0]]]= [row[df.columns[1]]]
            
        with torch.no_grad():    # disabled gradient calculation
        
            itr = iter(data)
            
            input_data, labels, time_features = next(itr)
            
            time_features = time_features.to(self.device)
            
            pred = self(input_data, time_features, future=self.future_predict)
            
            pred = pred[-1][0]
            pred = pred.to('cpu')
            labels = labels[-1][0]
            
            
            time = train_loader.dataset.get_label_times(0)
            
            pred_csr = sparse.csr_matrix(unnormalize_grid(pred,MIN,MAX))
            labels_csr = sparse.csr_matrix(unnormalize_grid(labels,MIN,MAX))
            pred_dict = dict(pred_csr.todok().items())
            labels_dict = dict(labels_csr.todok().items())
            
            pred_grid_locations_x = []
            pred_grid_locations_y = []
            
            for i in pred_dict.keys():
                x = list(grid_dict.keys())[list(grid_dict.values()).index(i[0])]
                y = list(grid_dict.keys())[list(grid_dict.values()).index(i[1])]
                pred_grid_locations_x.append(x)
                pred_grid_locations_y.append(y)
                    
            pred_data = {'Grid Location': pred_dict.keys(), 'Demand': pred_dict.values(),
                    'Start Location': pred_grid_locations_x,
                    'End Location': pred_grid_locations_y}
            
            pd.DataFrame.from_dict(pred_data).to_csv("pred_value.csv", index = False)
            
                
            labels_grid_locations_x = []
            labels_grid_locations_y = []
            
            for i in labels_dict.keys():
                x = list(grid_dict.keys())[list(grid_dict.values()).index(i[0])]
                y = list(grid_dict.keys())[list(grid_dict.values()).index(i[1])]
                labels_grid_locations_x.append(x)
                labels_grid_locations_y.append(y)

            
            label_data = {'Grid Location': labels_dict.keys(), 'Demand': labels_dict.values(),
                    'Start Location': labels_grid_locations_x,
                    'End Location': labels_grid_locations_y}
            
            pd.DataFrame.from_dict(label_data).to_csv("label_data.csv", index = False)
            
            print("predicted")
            print(sparse.csr_matrix(unnormalize_grid(pred,MIN,MAX)))
            
            print("actual")
            print(sparse.csr_matrix(unnormalize_grid(labels,MIN,MAX)))
            
            labels = unnormalize_grid(labels,MIN,MAX)
            pred = unnormalize_grid(pred,MIN,MAX)
            
            plt.figure()
            sns.heatmap(labels, cmap=sns.cubehelix_palette(as_cmap=True), vmax=4)
            #plt.imshow(labels, cmap='hot', interpolation='nearest')
            plt.title("Actual: "+ time[0])
            plt.show()
            
            plt.figure()
            sns.heatmap(pred, cmap=sns.cubehelix_palette(as_cmap=True))
            #plt.imshow(pred, cmap='hot', interpolation='nearest')
            plt.title("Pred: "+ time[0])
            plt.show()
            
            plt.figure()
            plt.title("Loss")
            plt.ylabel("loss")
            plt.xlabel("Epoch")
            plt.plot(self.loss_list)
            
            pred_demand = []
            label_demand = []
            
            for i, (feature, label, time_features) in tqdm(enumerate(data),unit="batch", total=len(data)):
                #push feature and label to gpu
                
                label   = (label).to(self.device)
                
                time_features = time_features.to(self.device)
                
                pred = self(feature, time_features, future=self.future_predict)
                    
                
                loss = self.criterion(pred, label)
                
                pred = unnormalize_grid(pred[0][0],MIN,MAX)
                label = unnormalize_grid(label[0][0],MIN,MAX)
                
                pred = pred.to('cpu')
                label = label.to('cpu')
                
                pred_demand.append(sparse.csr_matrix(pred).sum())
                label_demand.append(sparse.csr_matrix(label).sum())
            
            plt.figure()
            plt.title("Total Demand Prediced")
            plt.plot(pred_demand)
            plt.figure()
            plt.title("Total Demand Actual: "+ time[0])
            plt.plot(label_demand)
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




class scooter_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        #subtrace window size and 1 for the last input which will not have a label
        return len(self.data)
        
    def __getitem__(self, idx):  
        #get data returned by raw_dataset
        features, labels, times_label = self.data[idx]

        #get future number of inputs starting from after features end
        labels = np.array(list(i.toarray() for i in labels))
        
        #get time_features for future number of inputs starting from after features end
        time_features = [extract_time_features(i) for i in times_label]
        
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
    return features, torch.Tensor(np.array(labels)), torch.Tensor(np.array(time_features))

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
        features = list(self.data[i][0] for i in range(idx, idx+self.window))
        
        #get future number of inputs starting from after features end
        label = list(self.data[i][0] for i in range((idx+self.window),idx+self.window+self.future))
        
        #get time_features for the output to be predicted
        times_features = list(self.data[i][1][0] for i in range(idx, idx+self.window+self.future))
        
        return features, label, times_features
    
        

    
def initiate_loader(file, batchsize, window, furure_size, train_size): 
    raw_data = read_pickle_file(file)
    
    #remove outliers from data base 
    raw_data = remove_outliers(raw_data)
    
    #normalize data
    normalize_trip_db(raw_data)
    
    #create dataset for raw data 
    raw_dataset = raw_Dataset(raw_data, window, furure_size)

    #get training dataset size 
    train_size = int(train_size * len(raw_dataset))
    
    #get test dataset size
    test_size = len(raw_dataset) - train_size
    
    #split data set into training and test data set
    train_dataset, test_dataset = torch.utils.data.random_split(raw_dataset, [train_size, test_size])
    
    #tranform raw dataset to scooter dataset 
    #the scooter dataset handles the data is diffrently since the data is 
    #no longer in order after the split
    train_dataset = scooter_Dataset(train_dataset)
    test_dataset  = scooter_Dataset(test_dataset)
    
    # to prepare training loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, num_workers=2, collate_fn=csr_collate)
    # to prepare test loader
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=1, num_workers=2, collate_fn=csr_collate)
    
    del train_dataset, test_dataset, raw_data

    
    return train_loader, test_loader

def extract_time_features(str_date):
    #create datetime from string
    date = datetime.strptime(str_date, "%Y-%m-%dT%H:%M:%SZ")
    
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


    
if __name__ == "__main__":    
    #get device
    device = get_device()
    
    #empty memory
    torch.cuda.empty_cache()
    train_loader, test_loader = initiate_loader(SCOOTER_DATA, 
                                                BATCH_SIZE, 
                                                WINDOWSIZE, 
                                                MODEL_CONFIG["future_size"],
                                                TRAINING_SIZE)
    
    
    processor_info(device)
    
    #create model
    model = e_scootermodel(model_file     = MODEL_FILE,
                           model_config = MODEL_CONFIG,
                            device         = device,
                            len_extra_features = 20)
   

    model.train_model(train_loader, BATCH_SIZE, EPOCHS)
    
    model.eval_model(test_loader, BATCH_SIZE)