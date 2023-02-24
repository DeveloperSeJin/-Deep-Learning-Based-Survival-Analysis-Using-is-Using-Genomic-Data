import torch
import torch.nn as nn

class test_model(nn.Module) :
    def __init__(self, input_layer) :
        super().__init__()
        
        hidden_node = 3000
        output_node = 500
        
        self.hidden1 = nn.Linear(input_layer, hidden_node)
        self.hidden2 = nn.Linear(hidden_node, output_node)
        self.output = nn.Linear(output_node, 1, bias = False)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.hidden1(x))
        
        x = self.tanh(self.hidden2(x))
        
        x = self.output(x)
        
        return x