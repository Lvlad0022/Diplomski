import torch.nn as nn
import torch
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedSnakeNN(nn.Module):
    def __init__(self,map_height = 25,
                        map_width = 60,
                        metadata_dim = 38,
                        num_actions = 4,
                        map_channels = 3,):
        super(AdvancedSnakeNN, self).__init__()

        

        # --- Convolutional Layers for Map Processing ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Dynamically calculate flattened CNN output size
        dummy_input = torch.randn(1, map_channels, map_height, map_width)
        self._conv_out_size = self._get_conv_out_size(dummy_input)

        # --- Metadata FC layers ---
        self.metadata_fc_layers = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self._metadata_fc_out_size = 256

        # --- Combined FC layers ---
        self._combined_input_size = self._conv_out_size + self._metadata_fc_out_size
        self.combined_fc_layers = nn.Sequential(
            nn.Linear(self._combined_input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self._combined_fc_out_size = 64

        # --- Output layer ---
        self.output_layer = nn.Linear(self._combined_fc_out_size+1, num_actions)

    def _get_conv_out_size(self, x):
        out = self.conv_layers(x)
        return out.view(out.size(0), -1).size(1)

    def forward(self, map_input, metadata_input):
        map_input = map_input.to(DEVICE)
        metadata_input = metadata_input.to(DEVICE)
        round_no =metadata_input[:,-1:]

        conv_features = self.conv_layers(map_input)
        conv_features_flat = conv_features.view(conv_features.size(0), -1)
        metadata_features = self.metadata_fc_layers(metadata_input) 
        combined_features = torch.cat((conv_features_flat, metadata_features), dim=1) 
        combined_fc_out = self.combined_fc_layers(combined_features)
        combined_fc_out = torch.cat((combined_fc_out,round_no), dim=1) 
        output = self.output_layer(combined_fc_out) 
        return output


class SnakeNN(nn.Module):
    def __init__(self,map_height = 25,
                        map_width = 60,
                        metadata_dim = 38,
                        num_actions = 4,
                        map_channels = 3):
        super(SnakeNN, self).__init__()

        

        # --- Convolutional Layers for Map Processing ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Dynamically calculate flattened CNN output size
        dummy_input = torch.randn(1, map_channels, map_height, map_width)
        self._conv_out_size = self._get_conv_out_size(dummy_input)

        # --- Metadata FC layers ---
        self.metadata_fc_layers = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self._metadata_fc_out_size = 128

        # --- Combined FC layers ---
        self._combined_input_size = self._conv_out_size + self._metadata_fc_out_size
        self.combined_fc_layers = nn.Sequential(
            nn.Linear(self._combined_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self._combined_fc_out_size = 32

        # --- Output layer ---
        self.output_layer = nn.Linear(self._combined_fc_out_size, num_actions)

    def _get_conv_out_size(self, x):
        out = self.conv_layers(x)
        return out.view(out.size(0), -1).size(1)

    def forward(self, map_input, metadata_input):
        map_input = map_input.to(DEVICE)
        metadata_input = metadata_input.to(DEVICE)

        conv_features = self.conv_layers(map_input)
        conv_features_flat = conv_features.view(conv_features.size(0), -1)
        metadata_features = self.metadata_fc_layers(metadata_input) 
        combined_features = torch.cat((conv_features_flat, metadata_features), dim=1) 
        combined_fc_out = self.combined_fc_layers(combined_features)
        output = self.output_layer(combined_fc_out) 
        return output


class DuelingSnakeNN(nn.Module):
    def __init__(self,map_height = 25,
                        map_width = 60,
                        metadata_dim = 38,
                        num_actions = 4,
                        map_channels = 3):
        super(DuelingSnakeNN, self).__init__()

        

        # --- Convolutional Layers for Map Processing ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Dynamically calculate flattened CNN output size
        dummy_input = torch.randn(1, map_channels, map_height, map_width)
        self._conv_out_size = self._get_conv_out_size(dummy_input)

        # --- Metadata FC layers ---
        self.metadata_fc_layers = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        metadata_out = 128

        self._metadata_fc_out_size = 128
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

    def _get_conv_out_size(self, x):
        out = self.conv_layers(x)
        return out.view(out.size(0), -1).size(1)

    def forward(self, map_input, metadata_input):
        map_input = map_input.to(DEVICE)
        metadata_input = metadata_input.to(DEVICE)

        conv_features = self.conv_layers(map_input)
        conv_features_flat = conv_features.view(conv_features.size(0), -1)
        metadata_features = self.metadata_fc_layers(metadata_input) 
        combined = torch.cat((conv_features_flat, metadata_features), dim=1)
        value = self.fc_value(combined)
        advantage = self.fc_advantage(combined)

        Q =  value + (advantage - advantage.mean(dim=1, keepdim=True))
        return Q
