import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicConvBackbone(nn.Module):
    def __init__(self, map_channels=3, output_dim=128):
        super().__init__()
        self.output_dim = output_dim

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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(128, output_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)


class ResNeXtSnakeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, groups=4):
        super().__init__()
        hidden_channels = out_channels * expansion
        hidden_groups = min(groups, hidden_channels)

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=hidden_groups,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=hidden_groups,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


class ResNeXtSnakeBackbone(nn.Module):
    def __init__(self, map_channels=3, output_dim=128):
        super().__init__()
        self.output_dim = output_dim

        self.stem = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.blocks_10x10 = nn.Sequential(
            ResNeXtSnakeBlock(32, 64),
            ResNeXtSnakeBlock(64, 64),
        )
        self.pool = nn.MaxPool2d(2)
        self.blocks_5x5 = nn.Sequential(
            ResNeXtSnakeBlock(64, 128),
            ResNeXtSnakeBlock(128, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(128, output_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.stem(x)
        x = self.blocks_10x10(x)
        x = self.pool(x)
        x = self.blocks_5x5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

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
        device = self.mu_weight.device
        eps_in = torch.randn(1, self.in_features, device=device)
        eps_out = torch.randn(self.out_features, 1, device=device)

        self.eps_in.copy_(self.f(eps_in))
        self.eps_out.copy_(self.f(eps_out))
        self.weight_epsilon.copy_(self.eps_out @ self.eps_in)
        self.bias_epsilon.copy_(self.eps_out.squeeze())

    def forward(self, x, training, return_ratio=False):
        y_mu = F.linear(x, self.mu_weight, self.mu_bias)

        if training:
            w_noisy = self.sigma_weight * self.weight_epsilon
            b_noisy = self.sigma_bias * self.bias_epsilon
            y_sigma = F.linear(x, w_noisy, b_noisy)
            y = y_mu + y_sigma
        else:
            y_sigma = 0
            y = y_mu

        if return_ratio:
            eps = 1e-8
            ratio = torch.abs(y_sigma) / (torch.abs(y_mu) + eps)
            return y, ratio

        return y


class ClassicHead(nn.Module):
    def __init__(self, input_dim, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, features, return_ratios=False):
        return self.net(features)


class NoisyHead(nn.Module):
    def __init__(self, input_dim, num_actions=4, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.noisy1 = NoisyLinear(input_dim, 64)
        self.noisy2 = NoisyLinear(64, 32)
        self.noisy_output = NoisyLinear(32, num_actions)
        self.relu = nn.ReLU()

    def forward(self, features, return_ratios=False):
        if return_ratios:
            x, ratio1 = self.noisy1(features, self.is_training, True)
            x = self.relu(x)
            x, ratio2 = self.noisy2(x, self.is_training, True)
            x = self.relu(x)
            x, ratio3 = self.noisy_output(x, self.is_training, True)
            return x, (ratio1, ratio2, ratio3)

        x = self.noisy1(features, self.is_training, False)
        x = self.relu(x)
        x = self.noisy2(x, self.is_training, False)
        x = self.relu(x)
        return self.noisy_output(x, self.is_training, False)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DuelingHead(nn.Module):
    def __init__(self, input_dim, num_actions=4):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, features, return_ratios=False):
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingNoisyHead(nn.Module):
    def __init__(self, input_dim, num_actions=4, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.value1 = NoisyLinear(input_dim, 64)
        self.value_out = NoisyLinear(64, 1)
        self.advantage1 = NoisyLinear(input_dim, 64)
        self.advantage_out = NoisyLinear(64, num_actions)
        self.relu = nn.ReLU()

    def forward(self, features, return_ratios=False):
        if return_ratios:
            value, value_ratio1 = self.value1(features, self.is_training, True)
            value = self.relu(value)
            value, value_ratio2 = self.value_out(value, self.is_training, True)

            advantage, advantage_ratio1 = self.advantage1(features, self.is_training, True)
            advantage = self.relu(advantage)
            advantage, advantage_ratio2 = self.advantage_out(advantage, self.is_training, True)

            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            ratios = (value_ratio1, value_ratio2, advantage_ratio1, advantage_ratio2)
            return q_values, ratios

        value = self.relu(self.value1(features, self.is_training, False))
        value = self.value_out(value, self.is_training, False)
        advantage = self.relu(self.advantage1(features, self.is_training, False))
        advantage = self.advantage_out(advantage, self.is_training, False)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class QNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.ratios = False

    @property
    def is_training(self):
        return getattr(self.head, "is_training", self.training)

    @is_training.setter
    def is_training(self, value):
        if hasattr(self.head, "is_training"):
            self.head.is_training = value

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features, return_ratios=self.ratios)

    def reset_noise(self):
        if hasattr(self.head, "reset_noise"):
            self.head.reset_noise()


def build_snake_model(backbone_type="classic", head_type="classic", map_channels=3, num_actions=4):
    if backbone_type == "classic":
        backbone = ClassicConvBackbone(map_channels=map_channels)
    elif backbone_type == "resnext_snake":
        backbone = ResNeXtSnakeBackbone(map_channels=map_channels)
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    if head_type == "classic":
        head = ClassicHead(input_dim=backbone.output_dim, num_actions=num_actions)
    elif head_type == "noisy":
        head = NoisyHead(input_dim=backbone.output_dim, num_actions=num_actions, is_training=True)
    elif head_type == "dueling":
        head = DuelingHead(input_dim=backbone.output_dim, num_actions=num_actions)
    elif head_type == "dueling_noisy":
        head = DuelingNoisyHead(input_dim=backbone.output_dim, num_actions=num_actions, is_training=True)
    else:
        raise ValueError(f"Unsupported head_type: {head_type}")

    return QNetwork(backbone=backbone, head=head)
