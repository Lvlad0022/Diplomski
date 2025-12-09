import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))

        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        # Buffers must be defined ONLY in __init__
        self.register_buffer("eps_in", torch.zeros(1, in_features))
        self.register_buffer("eps_out", torch.zeros(out_features, 1))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))


        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)

        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.out_features))

    def f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        # Create noise ON THE SAME DEVICE AS PARAMETERS
        device = self.mu_weight.device

        eps_in = torch.randn(1, self.in_features, device=device)
        eps_out = torch.randn(self.out_features, 1, device=device)

        # Store factorized noise
        self.eps_in.copy_(self.f(eps_in))
        self.eps_out.copy_(self.f(eps_out))

        # Store final epsilon
        self.weight_epsilon.copy_(self.eps_out @ self.eps_in)
        self.bias_epsilon.copy_(self.eps_out.squeeze())

    def forward(self, x, training, return_ratio=False):

        # deterministic part
        y_mu = F.linear(x, self.mu_weight, self.mu_bias)

        if training:
            # noisy part
            w_noisy = self.sigma_weight * self.weight_epsilon
            b_noisy = self.sigma_bias * self.bias_epsilon
            y_sigma = F.linear(x, w_noisy, b_noisy)
            
            # full output
            y = y_mu + y_sigma
        else:
            y_sigma = 0
            y = y_mu

        if return_ratio:
            # Avoid division by zero
            eps = 1e-8
            ratio = torch.abs(y_sigma) / (torch.abs(y_mu) + eps)
            return y, ratio

        return y



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
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
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


class DQNnoisy(nn.Module):
    def __init__(self,is_training, map_channels=3, map_height=10, map_width=10, num_actions=4):
        super(DQNnoisy, self).__init__()

        self.is_training = is_training
        self.ratios = False

        # --- Convolutional Layers ---
        self.backbone = backbone_model(map_channels=map_channels)


        self.noisy1 = NoisyLinear(128, 64)
        self.noisy2 = NoisyLinear(64, 32)
        self.noisy_output = NoisyLinear(32, num_actions)
        
        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.backbone(x)

        if self.ratios:
            x,ratio1 = self.noisy1(x,self.is_training, self.ratios)
            x = self.relu(x)
            x, ratio2 = self.noisy2(x,self.is_training, self.ratios)
            x = self.relu(x)
            x, ratio3 = self.noisy_output(x,self.is_training, self.ratios)

            return x, (ratio1, ratio2, ratio3)
        
        
        
        x = self.noisy1(x,self.is_training, self.ratios)
        x = self.relu(x)
        x = self.noisy2(x,self.is_training, self.ratios)
        x = self.relu(x)
        x = self.noisy_output(x,self.is_training, self.ratios)

        return x

    def reset_noise(self):
        """Call this once per environment step (training mode)."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class DQNnoisy_metadata(nn.Module):
    def __init__(self,is_training, map_channels=3, metadata_dim = 1,map_height=10, map_width=10, num_actions=4):
        super(DQNnoisy_metadata, self).__init__()

        self.is_training = is_training
        self.ratios = False

        # --- Convolutional Layers ---
        self.backbone = backbone_model(map_channels=map_channels)

        self.linear = nn.Linear(128 + metadata_dim, 128)

        self.noisy1 = NoisyLinear(128, 64)
        self.noisy2 = NoisyLinear(64, 32)
        self.noisy_output = NoisyLinear(32, num_actions)
        
        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x, metadata):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(metadata.shape) == 1:
            metadata = metadata.unsqueeze(0)

        x = self.backbone(x)

        x = torch.cat((x, metadata), dim=1)

        x = self.relu(self.linear(x))

        if self.ratios:
            x,ratio1 = self.noisy1(x,self.is_training, self.ratios)
            x = self.relu(x)
            x, ratio2 = self.noisy2(x,self.is_training, self.ratios)
            x = self.relu(x)
            x, ratio3 = self.noisy_output(x,self.is_training, self.ratios)

            return x, (ratio1, ratio2, ratio3)
        
        
        
        x = self.noisy1(x,self.is_training, self.ratios)
        x = self.relu(x)
        x = self.noisy2(x,self.is_training, self.ratios)
        x = self.relu(x)
        x = self.noisy_output(x,self.is_training, self.ratios)

        return x

    def reset_noise(self):
        """Call this once per environment step (training mode)."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class DQNnoisy_metadata_dueling(nn.Module):
    def __init__(self, is_training, 
                 map_channels=3, metadata_dim=1,
                 map_height=10, map_width=10, num_actions=4):

        super(DQNnoisy_metadata_dueling, self).__init__()

        self.is_training = is_training
        self.ratios = False  # keeps your existing ratio system

        # Backbone (unchanged)
        self.backbone = backbone_model(map_channels=map_channels)

        # Metadata fusion layer
        self.linear = nn.Linear(128 + metadata_dim, 128)

        # -------------------------------
        # Dueling: Value stream
        # -------------------------------
        self.val_fc1 = nn.Linear(128, 64)
        self.val_fc2 = nn.Linear(64, 1)

        # -------------------------------
        # Dueling: Advantage stream
        # -------------------------------
        self.adv_fc1 = NoisyLinear(128, 64, sigma_init=0.02)
        self.adv_fc2 = NoisyLinear(64, num_actions, sigma_init=0.02)

        self.relu = nn.ReLU()

    def forward(self, x, metadata):

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(metadata.shape) == 1:
            metadata = metadata.unsqueeze(0)

        # Extract features
        x = self.backbone(x)

        # Add metadata
        x = torch.cat((x, metadata), dim=1)
        x = self.relu(self.linear(x))

        # === Value stream ===
        V = self.relu(self.val_fc1(x, self.is_training, False))
        V = self.val_fc2(V, self.is_training, False)  # shape: [B, 1]

        # === Advantage stream ===
        A = self.relu(self.adv_fc1(x, self.is_training, False))
        A = self.adv_fc2(A, self.is_training, False)  # shape: [B, num_actions]

        # Combine for dueling Q-values:
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        A_mean = A.mean(dim=1, keepdim=True)
        Q = V + (A - A_mean)

        if self.ratios:
            return Q, None
        return Q  # Adjust return as needed for ratios

    def reset_noise(self):
        """Call this once per environment step (training mode)."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


def load_backbone_only(model, checkpoint_path, strict=False):
    """Load state_dict only for the backbone"""
    checkpoint = torch.load(checkpoint_path)
    
    # Filter keys to only include backbone parameters
    backbone_state_dict = checkpoint["model_state_dict"]
    
    # Load only backbone parameters
    model.load_state_dict(backbone_state_dict, strict=False)
    print(f"Loaded {len(backbone_state_dict)} backbone parameters from checkpoint")
    
    return model
