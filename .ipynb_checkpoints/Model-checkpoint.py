import torch
import torch.nn as nn

class test_model(nn.Module) :
    def __init__(self, input_layer) :
        super().__init__()
        
        self.hidden1 = nn.Linear(input_layer, input_layer / 2)
        self.hidden2 = nn.Linear(input_layer / 2, input_layer / 4)
        self.output = nn.Linear(input_layer / 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x) :
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x