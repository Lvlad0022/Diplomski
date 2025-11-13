import torch.nn as nn
import os
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdvancedSimpleSnakeNN(nn.Module):
    def __init__(self, map_channels=2, map_height=25, map_width=60,
                 metadata_dim=7, num_actions=4):
        super(AdvancedSimpleSnakeNN, self).__init__()

        # --- 🔹 Convolutional Feature Extractor ---
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (12x30)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (6x15)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Optional downsample for compression
            nn.MaxPool2d(2),  # (3x7)
            nn.Dropout2d(0.2)
        )

        # --- compute flattened size dynamically ---
        dummy = torch.randn(1, map_channels, map_height, map_width)
        with torch.no_grad():
            conv_out = self.conv_layers(dummy)
            self._conv_out_size = conv_out.view(1, -1).shape[1]

        # --- 🔸 Metadata MLP ---
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        metadata_out = 256

        print(metadata_out ,self._conv_out_size)
        # --- 🔹 Combined MLP ---
        self.fc_combined = nn.Sequential(
            nn.Linear(self._conv_out_size + metadata_out, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # --- 🔸 Output Layer ---
        self.output = nn.Linear(32, num_actions)

    def forward(self, map_input, metadata_input):
        map_input = map_input.to(DEVICE)
        metadata_input = metadata_input.to(DEVICE)

        x = self.conv_layers(map_input)
        x = x.view(x.size(0), -1)
        m = self.metadata_fc(metadata_input)
        combined = torch.cat([x, m], dim=1)
        combined = self.fc_combined(combined)
        return self.output(combined)


class SimpleSnakeNN(nn.Module):
    def __init__(self, map_channels=2, map_height=25, map_width=60,
                 metadata_dim=7, num_actions=4):
        super(SimpleSnakeNN, self).__init__()

        # --- Convolutional Layers for Map Processing ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (12x30)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (6x15)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculate flattened size dynamically
        dummy_input = torch.randn(1, map_channels, map_height, map_width)
        with torch.no_grad():
            conv_out = self.conv_layers(dummy_input)
            self._conv_out_size = conv_out.view(1, -1).shape[1]

        # --- Metadata fully-connected layers ---
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        metadata_out = 128

        # --- Combined FC layers ---
        self.fc_combined = nn.Sequential(
            nn.Linear(self._conv_out_size + metadata_out, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # --- Output layer ---
        self.output = nn.Linear(128, num_actions)

    def forward(self, map_input, metadata_input):
        map_input = map_input.to(DEVICE)
        metadata_input = metadata_input.to(DEVICE)

        x = self.conv_layers(map_input)
        x = x.view(x.size(0), -1)
        m = self.metadata_fc(metadata_input)
        combined = torch.cat([x, m], dim=1)
        combined = self.fc_combined(combined)
        return self.output(combined)



class DuelingSimpleSnakeNN(nn.Module):
    def __init__(self, map_channels=2, map_height=25, map_width=60,
                 metadata_dim=7, num_actions=4):
        super(DuelingSimpleSnakeNN, self).__init__()

        # --- Convolutional Layers for Map Processing ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (12x30)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (6x15)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate flattened size dynamically
        self._conv_out_size = self._get_conv_out(map_channels, map_height, map_width)

        # --- Metadata fully-connected layers ---
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        metadata_out = 128

        # --- Combined FC layers ---
        self.fc_value = nn.Sequential(
            nn.Linear(self._conv_out_size + metadata_out, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(self._conv_out_size + metadata_out, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

        # --- Output layer ---
    def _get_conv_out(self, c, h, w):
        was_training = self.conv_layers.training
        self.conv_layers.eval()
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)  # samo za shape
            out = self.conv_layers(x)
            flat = out.view(1, -1).shape[1]
        if was_training:
            self.conv_layers.train()
        return flat

    def forward(self, map_input, metadata_input):
        map_input = map_input.to(DEVICE)
        metadata_input = metadata_input.to(DEVICE)

        x = self.conv_layers(map_input)
        x = x.view(x.size(0), -1)
        m = self.metadata_fc(metadata_input)
        combined = torch.cat([x, m], dim=1)
        value = self.fc_value(combined)
        advantage = self.fc_advantage(combined)

        Q =  value + (advantage - advantage.mean(dim=1, keepdim=True))
        return Q


from torchvision.models import resnet18, ResNet18_Weights

class ResnetSnakeNN_Small(nn.Module):
    def __init__(self, map_channels=2, metadata_dim=7, num_actions=4):
        super(ResnetSnakeNN_Small, self).__init__()

        # --- Backbone (ResNet18) ---
        resnet = resnet18(weights=None)  # no pretrained, 2 channels
        resnet.conv1 = nn.Conv2d(map_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer
        self.backbone_out = resnet.fc.in_features  # usually 512

        # Resize input to 224x224 to match ResNet structure
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # --- Metadata ---
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        metadata_out = 128

        # --- Combined ---
        self.fc_combined = nn.Sequential(
            nn.Linear(self.backbone_out + metadata_out, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.output = nn.Linear(128, num_actions)

    def forward(self, map_input, metadata_input):
        map_input = self.resize(map_input)
        x = self.backbone(map_input).view(map_input.size(0), -1)
        m = self.metadata_fc(metadata_input)
        combined = torch.cat([x, m], dim=1)
        combined = self.fc_combined(combined)
        return self.output(combined)
