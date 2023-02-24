import numpy as np
import matplotlib.pyplot as plt
import torch

def analysis(train_cost, val_cost) :
    #train_cost = train_cost.numpy()
    train_cost = torch.tensor(train_cost).detach().numpy()
    # #val_cost = val_cost.numpy()
    val_cost = torch.tensor(val_cost).detach().numpy()
    print(train_cost)
    
    plt.plot(train_cost)
    plt.plot(val_cost)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(['train', 'val'])    
    plt.savefig('test.png')