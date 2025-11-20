import torch.nn as nn
import torch
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedSnakeNN(nn.Module):
    def __init__(self):
        super(AdvancedSnakeNN, self).__init__()

        map_height = 25
        map_width = 60
        metadata_dim = 38
        num_actions = 4
        map_channels = 3

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
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self._combined_fc_out_size = 128

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



from torchvision.models import resnet18, ResNet18_Weights
class ResnetSnakeNN(nn.Module):
    def __init__(self):
        super(ResnetSnakeNN, self).__init__()

        map_height = 25
        map_width = 60
        metadata_dim = 38
        num_actions = 4

        # Pretrained ResNet18 (remove final FC layer)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
        self.backbone_output_size = resnet.fc.in_features  # Usually 512

        # Resize your map input to 3x224x224 for ResNet
        self.input_resizer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # Metadata processing
        self.metadata_fc_layers = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        metadata_out = 256

        # Combined layers
        self.combined_fc_layers = nn.Sequential(
            nn.Linear(self.backbone_output_size + metadata_out, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(128, num_actions)

    def forward(self, map_input, metadata_input):
        map_input = self.input_resizer(map_input)  # Resize for ResNet
        x = self.backbone(map_input).view(map_input.size(0), -1)
        metadata = self.metadata_fc_layers(metadata_input)
        combined = torch.cat([x, metadata], dim=1)
        x = self.combined_fc_layers(combined)
        return self.output_layer(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def freeze_backbone(self, freeze=True):
        """
        Freezes or unfreezes the EfficientNet backbone.

        Args:
            freeze (bool): If True, freeze all backbone layers (no gradient update).
                           If False, unfreeze (allow training).
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze