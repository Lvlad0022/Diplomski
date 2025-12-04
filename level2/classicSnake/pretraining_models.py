import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    def __init__(self, in_channel=3,  out_channel=4):
        super(Head, self).__init__()

        # --- Output layer ---
        self.output = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        output = self.output(x)
        return output
    

class backbone_model(nn.Module):
    def __init__(self, map_channels=3, map_height=10, map_width=10, num_actions=4):
        super(backbone_model, self).__init__()

        # --- Convolutional Layers ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc_layer = nn.Linear(128, 128)


    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return self.fc_layer(x)


class MultiHeadModel(nn.Module):
    def __init__(self, output_shapes):
        super().__init__()
        self.backbone = backbone_model(map_channels=4)
        self.heads = nn.ModuleList([Head(in_channel=128, out_channel=i) 
                                    for i in output_shapes])

    def forward(self, x):
        feats = self.backbone(x)
        return [head(feats) for head in self.heads]