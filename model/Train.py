from model.Model import model
from model.Model_add_layer import model_add_layer
import torch.optim as optim
from model.Survival_CostFunc_CIndex import *
import torch
import numpy as np
from tqdm import tqdm
import copy
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

def train(X_train, t_train, e_train, X_val, t_val, e_val, X_test, t_test, e_test,
         a, lr, l2, epochs, dropout_rates, optimizer = 'Adam', model_name = 'model') :
    
    if model_name == 'model_add_layer' :
        net = model_add_layer(len(X_train[0]), a, dropout_rates)
        print('model_add_layer')
    else :
        net = model(len(X_train[0]), a, dropout_rates)
        
    if torch.cuda.is_available():
        net.cuda()
        print('net is operated by cuda')
    
    opt = optim.Adam(net.parameters(), lr = lr, weight_decay = l2)
    
    if optimizer == "NAdam" :
        opt = optim.NAdam(net.parameters(), lr = lr, weight_decay = l2)
        
    train_cost = []
    val_cost = []
    #best_val_cost = np.inf
    
    #best_model = copy.deepcopy(net)

    for epoch in tqdm(range(epochs)) :
        net.train()
        opt.zero_grad()
        pred = net(X_train)
        train_loss = neg_par_log_likelihood(pred, t_train, e_train)
        train_loss.backward()
        opt.step()
        
        if (epoch % 200 == 0) :
            net.train()
            train_pred = net(X_train)
            train_loss = neg_par_log_likelihood(train_pred, t_train, e_train).view(1,)
            train_cost.append(train_loss)
            
            net.eval()
            val_pred = net(X_val)
            val_loss = neg_par_log_likelihood(val_pred, t_val, e_val).view(1,)
            val_cost.append(val_loss)
 
    net.eval()
    test_pred = net(X_test)
    test_cindex = c_index(test_pred, t_test, e_test)
    torch.save(net.state_dict(), './save_model/'  + str(test_cindex.detach().cpu().numpy()) + '_' + model_name)
    
#     best_model.eval()
#     test_pred = best_model(X_test)
#     test_cindex = c_index(test_pred, t_test, e_test)
#     torch.save(best_model.state_dict(), './save_model/'  + str(test_cindex.detach().cpu().numpy()) + '_' + str(a) + '_' + str(lr) + '_' + str(l2) + '_' + str(dropout_rates) + '_' + str(optimizer) + '_' + str(epoch) + '_' + model_name)
    
    return train_cost, val_cost, test_cindex