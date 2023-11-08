import torch
import torch.nn as nn

class model_add_layer(nn.Module) :
    def __init__(self, input_layer, activation, dropout_rates) :
        super().__init__()
        
        hidden_node1 = 143
        hidden_node2 = 143
        output_node = 1
        
        self.hidden1 = nn.Linear(input_layer, hidden_node1)
        self.hidden2 = nn.Linear(hidden_node1, hidden_node2)
        self.output = nn.Linear(hidden_node2, output_node, bias = False)
        
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Tanh" :
            self.activation = nn.Tanh()
            
        self.dropout = nn.Dropout(dropout_rates)
        
        
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.dropout(x)
        x = self.activation(self.hidden2(x))
        x = self.dropout(x)
        x_result = self.output(x)
        
        return x_result