import os
import pandas as pd
import numpy as np
from Preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Train import train

def run() :
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="4"

    lihc = loadData('./lihc_RNA.txt', './lihc_patient.txt')
    lihc = preprocessing(lihc)
    #lihc_corrFeature = calCorrFeature(lihc, 'lihc')
    #lihc_corrFeature = pd.DataFrame(lihc_corrFeature)
    lihc_corrFeature = pd.read_csv('./lihc_corrFeature.txt', sep = '\t', header = None)
    lihc = selectCorrFeature(lihc, lihc_corrFeature.T)
    
    train, test = train_test_split(lihc, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25,random_state=42)

    train = train.reset_index()
    test = test.reset_index()
    val = val.reset_index()

    t_train = train['t'].to_numpy(dtype = 'float32')
    e_train = train['e'].to_numpy(dtype = 'int32')
    X_train = train.drop(['t', 'e', 'Patient Identifier'], axis = 1)

    t_val = val['t'].to_numpy(dtype = 'float32')
    e_val = val['e'].to_numpy(dtype = 'int32')
    X_val = val.drop(['t', 'e', 'Patient Identifier'], axis = 1)

    t_test = test['t'].to_numpy(dtype = 'float32')
    e_test = test['e'].to_numpy(dtype = 'int32')
    X_test = test.drop(['t', 'e', 'Patient Identifier'], axis = 1)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    X_train = X_train.to_numpy(dtype = 'float32')
    X_test = X_test.to_numpy(dtype = 'float32')
    X_val = X_val.to_numpy(dtype = 'float32')


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    learning_rate = 0.0001
    epochs = 1000
    
    train_cindex, train_cost, val_cindex, val_cost, test_cindex =train(X_train, t_train, e_train, X_val, t_val, e_tval, X_test, t_test, e_test, learning_rate, epochs)