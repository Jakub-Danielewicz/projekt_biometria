import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model

class HandwritingMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64, 512)
        self.batchnorm = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.30)

    def forward(self, x):
        x = self.flatten(x)           # [B, 1, 64, 64] -> [B, 4096]
        x = F.relu(self.fc1(x))
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x