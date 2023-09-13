# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:54:12 2023

@author: ryang
"""

SCOOTER_DATA = "e_scooter_250_lousiville.pkl"
MODEL_FILE = 'e_scooter_lousiville_GRU_v1.pt'
GRID_DICT = 'grid_dict_250.csv'
FILE_PATH = "lousiville-escooter-2018-2019.csv"

BATCH_SIZE = 10
EPOCHS = 1
WINDOWSIZE = 74
data_format = "%Y-%m-%dT%H:%M"

CITY = "Lousiville"
MODEL_CONFIG = {"grid_size" : [2, 74 , 40], #Grid size the input grid will be transformed into 
                "hidden_size" : 550, #Hidden layer of the LSTM
                "num_layers" : 2, #Number of layers of the LSTM
                "future_size" : 24, #Number of future outputs that the LSTM will predict
                "input_size" : 1377, #Input size of the LSTM (number of latent features created by the encoder)
                "in_channels" : 2, #number in input channels for convolution network
                "channel1" : 18, # number of channes in first convolution channls
                "enc_padding" : 0, #padding for encoder
                "enc_channels" : 9, #number of channels at seconds convution layer
                "enc_kernels" : 2, #Kernal size for encoder
                "dec_kernels" : 3, #Kernal size for decoder
                "pool_kernels" : 2, #pool kernaels
                "pool_stride" : 2, #pool stride
                "learning_rate" : 0.0001, #learning rate
                "output_padding1" : 1, #padding for decoder
                "output_padding2" : 1, #padding for decoder second layer
                "unflatten_dim" : (9, 17, 9), #unflatten dimentions for decoder
                "rnn_type" : "GRU"} #Type of RNN network use LSTM, GRU, RNN
