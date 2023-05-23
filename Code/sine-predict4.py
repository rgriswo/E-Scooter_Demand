#!/usr/bin/env python

import os
import torch
#import torch.nn as nn
#import torch.optim as optim
# import numpy as np
import sineplot
from torch.utils.data import Dataset, DataLoader

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
    
class SineDataset(Dataset):
    def __init__(self, filename, ni=19, nwin=20, stride=1, frac=0.8, train=True, offset=1):
        self.filename = filename
        self.frace = frac
        self.train = train
        self.offset = offset
        self.ni = ni
        self.nwin = nwin
        self.stride = stride
        data = torch.load(self.filename)
        if data.ndim == 1:
            data = data.reshape([1,-1])
        self.channels = data.shape[0]
        
        cnt  = int(data.shape[1] * frac)            # slen for train data
        if cnt < self.nwin:
            print("Too short sequence length of train data.  The minimum is %d, but now it is %d" % (self.nwin, cnt))
            sys.exit()
            
        tcnt = data.shape[1] - cnt                  # slen for test data
        if tcnt < self.nwin: 
            print("Too short sequence length of test data.  The minimum is %d, but now it is %d" % (self.nwin, tcnt))
            sys.exit()
            
        self.data = data[:,:cnt] if train else data[:,cnt:]   
      
        self.num_per_channel = (self.data.shape[1] - self.nwin) // self.stride
        # print("self.data shape = %s, win=%d" % (self.data.shape, self.nwin))        
        return
        
    def __len__(self):
       return self.channels * self.num_per_channel
        
    def __getitem__(self, idx):     
        ridx = idx // self.num_per_channel
        cbeg = (idx % self.num_per_channel) * self.stride
        inputs = torch.from_numpy(self.data[ridx, cbeg:cbeg+self.ni]).type(torch.float32)
        labels = torch.from_numpy(self.data[ridx, cbeg+self.ni:cbeg+self.nwin]).type(torch.float32)
        return inputs, labels
               
class PredictModel(torch.nn.Module):
    def __init__(self, n_hidden=51, n_layers=2, ni=19, nwin=20, dev=None):
        super(PredictModel, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers 
        self.ni = ni
        self.nwin = nwin   
        self.lstm = torch.nn.LSTM(self.ni, self.n_hidden, num_layers=self.n_layers,
                                  batch_first=True)
        self.linear = torch.nn.Linear(self.n_hidden, self.ni)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, 
                                          weight_decay=0)   
        self.loss_list = []
        
        if dev is not None:
            self.device = torch.device(dev)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")           
        
        self.to(self.device)
        return 
    
    def forward(self, x, future=1):
        h1 = torch.zeros(self.n_layers, self.n_hidden, dtype=torch.float32, device=self.device)
        c1 = torch.zeros(self.n_layers, self.n_hidden, dtype=torch.float32, device=self.device)

        hidden_state = (h1, c1)  
            
        output, hidden_state = self.lstm(x, hidden_state)
        output = self.linear(output)
        
        output = output[:,-1:]
        feat = torch.cat([x[:,1:], output[:,-1:]], dim=1)
        
        # the last output is fed as an input to the prediction             
        for i in range(future-1):
            pred, hidden_state = self.lstm(feat, hidden_state)
            feat = torch.cat([feat[:,1:], pred[:,-1:]], dim=1)
            
        return feat[:,-future:], hidden_state
    
    def load_model(self, modelfile):
        if os.path.isfile(modelfile): 
            checkpoint = torch.load(modelfile)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # if self.device.type == "cuda":
            #     for state in self.optimizer.state.values():
            #         for k, v in state.items():
            #             if torch.is_tensor(v):
            #                 state[k] = v.cuda()
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
    
    def train_model(self, datafile, epochs=1, batchsize=1, stride=1):
        self.train()
        self.stride = stride
        print("device:", self.device.type)
        ds = SineDataset(datafile, ni=self.ni, nwin=self.nwin, stride=self.stride)
        loader = DataLoader(ds, batch_size=batchsize)
        epoch_start = len(self.loss_list)
        
        WAIT = "Wait..."
        for epoch in range(epochs):
            print("Epoch %d/%d: %s" % (epoch+epoch_start, epochs+epoch_start, WAIT), 
                  end='', flush=True)
            self.optimizer.zero_grad()
            for i, (features, labels) in enumerate(loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                pred, hidden = self(features, future=self.nwin-self.ni)
                loss = self.criterion(pred, labels)
                loss.backward()
                self.optimizer.step()
                
            print("\b"*len(WAIT) + "loss (%f)" % loss.item())
            self.loss_list.append(loss.item()) 

#        x0 = features[0:1,:].detach().numpy()
#        y0 = labels[0:1,:].detach().numpy()
#        yp = pred[0:1,:].detach().numpy()
#        sineplot.draw_predict(x0, y0, yp,
#                              title="Predict at Epoch %d" % epochs)   
        return
    
    def eval_model(self, datafile, pos=0):
        self.eval()
        batchsize = 1
        ds = SineDataset(datafile, train=False, 
                         ni=self.ni, nwin=self.nwin)
        pos = pos % len(ds)
        with torch.no_grad():    # disabled gradient calculation
            feature = torch.reshape(ds[pos][0], (1,-1)).to(self.device)
            label   = torch.reshape(ds[pos][1], (1,-1)).to(self.device)
            future  = self.nwin-self.ni
            pred, _ = self(feature, future=future)
            loss = self.criterion(pred, label)
            print("test loss =", loss.item())
            title = "Predict Pos %d at Epoch %d" % (pos, len(self.loss_list))

            x = feature.cpu().numpy()
            y = label.cpu().numpy()
            yp = pred.cpu().numpy()
                
            sineplot.draw_predict(x, y, yp, title=title)   
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