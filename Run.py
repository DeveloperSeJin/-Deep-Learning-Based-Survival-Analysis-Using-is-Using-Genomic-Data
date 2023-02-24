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
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def run() :    
    learning_rate = [0.9]
    regularization = [0.7]
    act = ["Tanh"]
    dropout_rates = [0.1]
    optimizers = ["NAdam"]
    
    epochs = 1000
    dtype = torch.FloatTensor
    
    X_train, t_train, e_train = load_data('./train_data.csv', dtype)
    X_test, t_test, e_test = load_data('./test_data.csv', dtype)
    X_val, t_val, e_val = load_data('./val_data.csv', dtype)
    
    test_cindexes = []
    
    for a in act :
        for lr in learning_rate:
            for l2 in regularization:
                for dropout_rate in dropout_rates :
                    for optimizer in optimizers :
                        train_cost, val_cost, test_cindex = train(X_train, t_train, e_train, X_val, t_val, e_val, X_test, t_test, e_test, a, lr, l2, epochs, dropout_rate, optimizer)
                        analysis(train_cost, val_cost, str(a) + '_' + str(lr) + '_' + str(l2) + '_' + str(dropout_rate) + '_' + str(optimizer))
                        test_cindexes.append(test_cindex)
                        
    plt.boxplot(test_cindexes)
    plt.xlabel('type')
    plt.ylabel('value')   
    plt.savefig("./1_boxplot.png")
    plt.clf()
run()