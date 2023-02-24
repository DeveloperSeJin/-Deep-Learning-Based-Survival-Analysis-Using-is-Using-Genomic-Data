import os
import pandas as pd
import numpy as np
#from Preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Train import train
from DataLoader import *
import torch
from Analysis import analysis

def run() :
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    learning_rate = 0.09
    epochs = 1000
    dtype = torch.FloatTensor
    
    X_train, t_train, e_train = load_data('./train_data.csv', dtype)
    X_test, t_test, e_test = load_data('./test_data.csv', dtype)
    X_val, t_val, e_val = load_data('./val_data.csv', dtype)
    
    train_cost, val_cost, test_cindex = train(X_train, t_train, e_train, X_val, t_val, e_val, X_test, t_test, e_test, learning_rate, epochs)
    
    print(test_cindex)
    
    analysis(train_cost, val_cost)
    
run()