from Model import model
import torch.optim as optim
from Survival_CostFunc_CIndex import *
import torch
import numpy as np
from tqdm import tqdm
import copy

def train(X_train, t_train, e_train, X_val, t_val, e_val, X_test, t_test, e_test,
         a, lr, l2, epochs, dropout_rates, optimizer) :
    
    net = model(len(X_train[0]), a, dropout_rates)
    if torch.cuda.is_available():
        net.cuda()
    
    if optimizer == "Adam" :
        opt = optim.Adam(net.parameters(), lr = lr, weight_decay = l2)
    
    elif optimizer == "NAdam" :
        opt = optim.NAdam(net.parameters(), lr = lr, weight_decay = l2)
        
    train_cost = []
    val_cost = []
    best_val_cost = np.inf
    
    for epoch in tqdm(range(epochs)) :
        net.train()
        train_pred = net(X_train)
        train_loss = neg_par_log_likelihood(train_pred, t_train, e_train)
        train_cost.append(train_loss)
        train_loss.backward()
        opt.zero_grad()
        opt.step()
            
        if (epochs % 20 == 0) :
            net.eval()
            val_pred = net(X_val)
            val_loss = neg_par_log_likelihood(val_pred, t_val, e_val)
            val_cost.append(val_loss)
        
            if (best_val_cost > val_loss) :
                best_val_cost = val_loss
                val_cindex = c_index(val_pred, t_val, e_val)
                best_model = copy.deepcopy(net)
                torch.save(best_model.state_dict(), './models1/' + str(val_cindex.detach().numpy()) + '_' + str(a) + '_' + str(lr) + '_' + str(l2) + '_' + str(dropout_rates) + '_' + str(optimizer) + '_' + str(epoch))
        # print('train_loss = ', train_loss, 'val_loss = ', val_loss)
    
    best_model.eval()
    test_pred = best_model(X_test)
    test_cindex = c_index(test_pred, t_test, e_test)
    torch.save(best_model.state_dict(), './models1/'  + str(test_cindex.detach().numpy()) + '_' + str(a) + '_' + str(lr) + '_' + str(l2) + '_' + str(dropout_rates) + '_' + str(optimizer) + '_' + str(epoch))
    
    return train_cost, val_cost, test_cindex