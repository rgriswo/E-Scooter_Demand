#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:35:40 2023

@author: dskim
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages

def next_color():
    color_pool = 'rbgcmy'
    i = 0
    while True: 
        i %= len(color_pool)
        yield color_pool[i]
        i += 1
    return
    
def plot_loss(modelfile):
    checkpoint = torch.load(modelfile)
    loss = checkpoint['loss']
    plt.plot(loss)
    return
                      
def draw_predict(x, y, yp, title=None, filename=None): 
    if filename is not None:
        _, output = os.path.splitext(filename)
        if output == '.pdf':
            pdf = PdfPages(filename)
        else:
            print('unkown file type :', filename)
            return 
        
    plt.figure(figsize=(12,6))
    if title is not None:
        plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ilength = x.shape[1]
    flength = y.shape[1]    
    xinput  = np.arange(ilength)
    xfuture = np.arange(ilength, ilength+flength)
    
    col = next_color()
    
    for i in range(x.shape[0]): 
        colch = next(col)
        plt.plot(xinput,  x[i,:],  colch, linewidth=2.0)
        plt.plot(xfuture, y[i,:],  colch, linewidth=1.0)
        plt.plot(xfuture, yp[i,:], colch+":", linewidth=2.0)
        
    if filename is not None:
        pdf.savefig()  
        pdf.close()
    else:
        plt.show()
        
    plt.close()  
    return

