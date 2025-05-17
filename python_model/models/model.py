import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError("This model is not implemented yet.")
    