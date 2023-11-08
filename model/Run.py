import os
import pandas as pd
import numpy as np
#from Preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Train import train
from DataLoader import *
import torch
from Analysis import draw_learning
import matplotlib.pyplot as plt

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def run() :    
    learning_rate = [0.9]
    regularization = [0.7]
    act = ["Tanh"]
    dropout_rates = [0.0775]
    optimizers = ["NAdam"]
    
    lr = 0.9
    l2 = 0.7
    a = "Tanh"
    dropout_rate = 0.1
    optimizer = "NADam"
    
    epochs = 1000
    dtype = torch.FloatTensor
    
    X_train, t_train, e_train = load_data('./train_data.csv', dtype)
    X_test, t_test, e_test = load_data('./test_data.csv', dtype)
    X_val, t_val, e_val = load_data('./val_data.csv', dtype)
    
    #for a in act :
        #for lr in learning_rate:
            #for l2 in regularization:
                #for dropout_rate in dropout_rates :
                    #for optimizer in optimizers :
    train_cost, val_cost, test_cindex = train(X_train, t_train, e_train, X_val, t_val, e_val, X_test, t_test, e_test, a, lr, l2, epochs, dropout_rate, optimizer)
    draw_learning(train_cost, val_cost, str(a) + '_' + str(lr) + '_' + str(l2) + '_' + str(dropout_rate) + '_' + str(optimizer))
                        
    return test_cindex