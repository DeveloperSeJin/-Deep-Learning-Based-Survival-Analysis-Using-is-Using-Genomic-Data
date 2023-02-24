from Survival_CostFunc_CIndex import *
import torch
from DataLoader import load_data
from Model import test_model
import os

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
            
createCIndexArray()