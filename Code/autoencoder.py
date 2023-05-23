#!/usr/bin/env python3
# author: Stephen Kim (dskim@iu.edu)

import random, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def processor_info(device):
    print(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Number of Devics:', torch.cuda.device_count())
        print('Allocated Memory:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached Memory   :', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    return 

def get_device(default="cuda"):
    if torch.cuda.is_available() and default=="cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device: ", device)
    return device

def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0))) 

def show_images(loader, classes): 
    #Obtain one batch of training images
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images.numpy() # convert images to numpy for display
    
    #Plot the images
    fig = plt.figure(figsize=(8, 8))
    # display 20 images
    for idx in np.arange(9):
        ax = fig.add_subplot(3, 3, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])
             
def compare_images(original, reconstructed, labels, classes, choice): 
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    #Original Images
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for i, idx in enumerate(choice):
        ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
        imshow(original[idx])
        ax.set_title(classes[labels[idx]])
    fig.suptitle("Original Images")        
    plt.show()
    
    #Reconstructed Images
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for i, idx in enumerate(choice):
        ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
        imshow(reconstructed[idx])
        ax.set_title(classes[labels[idx]])
    fig.suptitle('Reconstructed Images')
    plt.show()             
    
def plot_loss(modelfile):
    checkpoint = torch.load(modelfile)
    loss = checkpoint['loss']
    plt.plot(loss)
    return    
    
#Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, model_file=None, device=None):
        super(ConvAutoencoder, self).__init__()
        self.model_file = model_file if model_file is not None else 'autoencoder-v100.pt'
        self.device = device if device is not None else get_device()
        in_channels = 3
        channel1 = 16
        enc_padding = 1
        enc_channels = 32
        enc_kernels = 3
        dec_kernels = 2
        pool_kernels = 2
        pool_stide = 2
        learning_rate = 0.0001
        # Encoder
        self.encode = nn.Sequential (
            nn.Conv2d(in_channels, channel1, enc_kernels, padding=enc_padding), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stide),                   # kernel_size, stride
            nn.Conv2d(channel1, enc_channels, enc_kernels, padding=enc_padding),# in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(pool_kernels, pool_stide)                    # kernel_size, stride
        )
        #Decoder
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(enc_channels, channel1, dec_kernels, stride=pool_stide), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.ConvTranspose2d(channel1, in_channels, dec_kernels, stride=pool_stide),  # in_channels, out_channels, kernel_size
            nn.Sigmoid()
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_list = [] 
        self.to(self.device)
        if self.model_file is not None:
            self.load_model(self.model_file)
        return 
        

    def forward(self, x):           # input:    32x3x32x32    
        x = self.encode(x)          # encoded:  32x4x8x8
        x = self.decode(x)          # decoded:  32x3x32x32
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
            for data in loader:
                images, _ = data
                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()*images.size(0)
                  
            train_loss = train_loss/len(loader)
            self.loss_list.append(train_loss)
            print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch+1+epochs_done, epochs_togo, train_loss),
                  flush=True)
            
        self.save_model(self.model_file)
        
        self.train(False)
        return
    
def initiate_loader(batchsize): 
    transform = transforms.ToTensor()    
    
    # to prepare training loader
    train_data = datasets.CIFAR10(root='data', train=True,  download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, num_workers=0)
    # to prepare test loader
    test_data  = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batchsize, num_workers=0)
    return train_loader, test_loader

def main(model_file, batchsize, epochs):    
    device = get_device()
    train_loader, test_loader = initiate_loader(batchsize)
    model = ConvAutoencoder(model_file, device)
    model.train_model(train_loader, batchsize, epochs)    
    
    # test and evaluation
    image_chosen = random.choices(list(range(batchsize)), k=5)
    #Define the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    #Batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)    
    reconstructed = model(images)
    compare_images(images, reconstructed.detach(), labels, classes, image_chosen)
    
    return

if __name__ == '__main__':
    batch_size = 32
    model_file = 'autoencoder-model-v2.pt'

#    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    epochs = 1
    
    t0 = time.time()
    main(model_file, batch_size, epochs)
    t1 = time.time()
    print("mean exe time: ", (t1-t0)/epochs)
