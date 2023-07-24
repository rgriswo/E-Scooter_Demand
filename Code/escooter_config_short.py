# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:54:12 2023

@author: ryang
"""

SCOOTER_DATA = "e_scooter_demand_short.pkl"
MODEL_FILE = 'escooter_model_short_v2.pt'
GRID_DICT = 'grid_dict_short.csv'
FILE_PATH = "purr_scooter_data.csv"

BATCH_SIZE = 12
EPOCHS = 1
WINDOWSIZE = 10

MODEL_CONFIG = {"hidden_size" : 500,
                "num_layers" : 2,
                "future_size" : 2,
                "input_size" : 1800,
                "in_channels" : 1,
                "channel1" : 16,
                "enc_padding" : 3,
                "enc_channels" : 2,
                "enc_kernels" : 3,
                "dec_kernels" : 2,
                "pool_kernels" : 2,
                "pool_stride" : 4,
                "learning_rate" : 0.00001,
                "output_padding1" : 3,
                "output_padding2" : 0,
                "unflatten_dim" : (2, 30, 30)}
