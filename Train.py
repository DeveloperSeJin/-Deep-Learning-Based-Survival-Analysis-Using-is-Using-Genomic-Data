from Model import test_model
import torch.optim as optim
from Survival_CostFunc_CIndex import *

def train(X_train, t_train, e_train, X_val, t_val, e_tval, X_test, t_test, e_test,
         learning_rate, epochs) :
    
    net = test_model(len(X_train[0]))
    opt = optim.Adam(net.parameters(), lr = learning_rate)
    train_cost = []
    val_cost = []
    train_cindex = []
    val_cindex = []
    
    for epoch in range(epochs) :
        net.train()
        train_pred = net(X_train)
        train_loss = neg_par_log_likelihood(train_pred, t_train, e_train)
        train_cost.append(train_loss)
        train_loss.backward()
        opt.zero_grad()
        opt.step()
        train_cindex.append(c_index(train_pred, t_train, e_train))
            
        if (epochs % 20 == 0) :
            net.eval()
            val_pred = net(X_val)
            val_loss = neg_par_log_likelihood(val_pred, t_val, e_val)
            val_cost.append(val_loss)
            val_cindex.append(c_index(val_pred, t_val, e_val))
        
            if (min(val_cost) > val_cost) :
                torch.save(model.state_dict(), './models/' + str(epoch))
        print('train_loss = ', train_loss, 'val_loss = ', val_loss)
    
    net.eval()
    test_pred = net(X_test)
    test_cindex = c_index(test_pred, t_test, e_test)
    
    return train_cindex, train_cost, val_cindex, val_cost, test_cindex