import numpy as np
import matplotlib.pyplot as plt
import torch

def analysis(train_cost, val_cost, name) :
    #train_cost = train_cost.numpy()
    train_cost = torch.tensor(train_cost).detach().numpy()
    # #val_cost = val_cost.numpy()
    val_cost = torch.tensor(val_cost).detach().numpy()
    
    plt.plot(train_cost)
    plt.plot(val_cost)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(['train', 'val'])    
    plt.savefig('LearningCurve1/' + name + '.png')
    plt.clf()