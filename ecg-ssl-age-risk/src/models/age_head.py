import torch
import torch.nn as nn

class AgeHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.2):
        super(AgeHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # Single output for age regression
        )

    def forward(self, x):
        return self.net(x)
