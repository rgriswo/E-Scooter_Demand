#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:11:52 2024

@author: dskim
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
from readconfig import read_config

config = {}

def get_device(default="cuda"):
    if torch.cuda.is_available() and default=="cuda":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def processor_info(device):
    print(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Number of Devics:', torch.cuda.device_count())
        print('Allocated Memory:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached Memory   :', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    return 

def main(filename):
    config = read_config('escooter.ini')
    device = get_device("cuda")
    torch.cuda.empty_cache()
    processor_info(device)
    
    return 
    

if __name__ == "__main__":
    filename = sys.argv[1:]
#    main(filename)
    main(r'purr_scooter_data.pkl') 
    
