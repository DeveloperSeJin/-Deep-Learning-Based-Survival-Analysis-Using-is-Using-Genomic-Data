from model.Survival_CostFunc_CIndex import *
import torch
from model.DataLoader import load_data
#from model.Model import model
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def draw_learning_rate(train_cost, val_cost, name) :
    #train_cost = train_cost.numpy()
    train_cost = torch.tensor(train_cost).detach().numpy()
    # #val_cost = val_cost.numpy()
    val_cost = torch.tensor(val_cost).detach().numpy()
    
    plt.plot(train_cost)
    plt.plot(val_cost)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(['train', 'val'])    
    plt.savefig('./analysis/' + name + '.png')
    plt.clf()
    
def createCIndexArray() :
    dtype = torch.FloatTensor
    X_test, t_test, e_test = load_data('./test_data.csv', dtype)
    cindexs = []
    
    for (root, directories, files) in os.walk('./models') :
        for file in files :
            file_path = os.path.join(root, file)
            activation = file.split('_')[1]
            drop_out = file.split('_')[-3]
            model = test_model(len(X_test[0]), activation, float(drop_out))
            model.load_state_dict(torch.load('./models/' + file))
            model.eval()
    
            pred = model(X_test)
            test_cindex = c_index(pred, t_test, e_test)
            cindexs.append(test_cindex)
            
    plt.boxplot(cindexs.detach().numpy())
    plt.xlabel('type')
    plt.ylabel('value')   
    plt.savefig("./hidden2_boxplot.png")
    plt.clf()
   
def calculateCIndex(cindexes, name) :
    plt.boxplot(cindexes)
    plt.xlabel('type')
    plt.ylabel('value')   
    plt.savefig('./Analysis/' + name + '_cindex.png')
    plt.clf()