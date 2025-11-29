import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, map_channels=3, map_height=10, map_width=10, num_actions=4):
        super(DQN, self).__init__()

        # --- Convolutional Layers for Map Processing ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # --- Combined FC layers ---
        self.fc_combined = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # --- Output layer ---
        self.output = nn.Linear(32, num_actions)

    def forward(self, x):
        if(len(x.shape) == 3):
            x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = self.pool(x)   
        x = x.view(x.size(0), -1)
        combined = self.fc_combined(x)
        return self.output(combined)