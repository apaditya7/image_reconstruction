import torch
import torch.nn as nn

class SimpleFMRIEncoder(nn.Module):
    def __init__(self, num_voxels, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_voxels, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, fmri):
        return self.encoder(fmri)