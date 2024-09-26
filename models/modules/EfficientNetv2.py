import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _find_optimal_num_groups(num_channels):
    """
    Find the largest divisor of num_channels that is closest to a quarter of num_channels.
    
    Args:
    - num_channels (int): The number of channels in the input tensor.
    
    Returns:
    - int: The optimal number of groups.
    """
    target_num_groups = num_channels // 2
    best_diff = num_channels  # Initialize with the maximum possible difference
    best_divisor = 1
    for divisor in range(1, num_channels + 1):
        if num_channels % divisor == 0:
            diff = abs(divisor - target_num_groups)
            if diff < best_diff:
                best_diff = diff
                best_divisor = divisor
            elif diff == best_diff and divisor > best_divisor:
                best_divisor = divisor
    return best_divisor

def get_backbone_config(variant):
    configs = {

        'b0_v1': {'width_coefficient': 1.0, 'depth_coefficient': 1.0},
        'b1_v1': {'width_coefficient': 1.0, 'depth_coefficient': 1.1},
        'b2_v1': {'width_coefficient': 1.1, 'depth_coefficient': 1.2},
        'b3_v1': {'width_coefficient': 1.2, 'depth_coefficient': 1.4},
        'b4_v1': {'width_coefficient': 1.4, 'depth_coefficient': 1.8},
        'b5_v1': {'width_coefficient': 1.6, 'depth_coefficient': 2.2},
        'b6_v1': {'width_coefficient': 1.8, 'depth_coefficient': 2.6},
        'b7_v1': {'width_coefficient': 2.0, 'depth_coefficient': 3.1},

        'b0_v2': {'width_coefficient': 1.0, 'depth_coefficient': 1.0},
        'b1_v2': {'width_coefficient': 1.0, 'depth_coefficient': 1.1},
        'b2_v2': {'width_coefficient': 1.1, 'depth_coefficient': 1.2},
        'b3_v2': {'width_coefficient': 1.2, 'depth_coefficient': 1.4},
        'b4_v2': {'width_coefficient': 1.4, 'depth_coefficient': 1.8},
        'b5_v2': {'width_coefficient': 1.6, 'depth_coefficient': 2.2},
        'b6_v2': {'width_coefficient': 1.8, 'depth_coefficient': 2.6},
        'b7_v2': {'width_coefficient': 2.0, 'depth_coefficient': 3.1},
        's_v2':  {'width_coefficient': 1.0, 'depth_coefficient': 2.0},
        'm_v2':  {'width_coefficient': 1.1, 'depth_coefficient': 2.1},
        'l_v2':  {'width_coefficient': 1.2, 'depth_coefficient': 2.2},
    }
    return configs[variant]

def get_activation(name='relu'):
    activations = {
        'relu': nn.ReLU(),
        'swish': nn.SiLU(),
        'mish': nn.Mish(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.01),
    }
    return activations[name]

class CustomNorm(nn.Module):
    def __init__(self, num_features, norm_type="batch"):
        """
        Initializes a custom normalization layer.
        
        Args:
        - num_features (int): Number of features in the input.
        - norm_type (str): Type of normalization ('batch', 'group', 'layer', 'instance').
        """
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features)
        elif norm_type == "group":
            optimal_groups = _find_optimal_num_groups(num_features)
            self.norm = nn.GroupNorm(num_groups=optimal_groups, num_channels=num_features)
        elif norm_type == "layer":
            # Assuming 1D LayerNorm for simplicity; adjust as needed for your application
            self.norm = nn.LayerNorm(normalized_shape=[num_features])
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(num_features)
        else:
            raise ValueError(f"Unsupported norm_type {norm_type}")

    def forward(self, x):
        return self.norm(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim, activation_func=nn.ReLU(inplace=True)):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_dim, 1),
            activation_func,
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
        return x

class FusedMBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation='relu', use_se=False, se_ratio=4, dropout_rate=0.0, stochastic_depth_prob=0.0, norm_type="batch"):
        super(FusedMBConv1d, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)

        self.fused_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            CustomNorm(out_channels, norm_type),
            activation_func,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if self.use_residual else nn.Identity()
        self.se = SEBlock(out_channels, max(1, int(out_channels // se_ratio)), activation_func) if use_se else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x if self.use_residual else None
        x = self.fused_conv(x)
        x = self.se(x)
        x = self.dropout(x)
        if self.use_residual:
            x = self.stochastic_depth(x) + identity
        return x

class MBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion=1, activation='relu', use_se=False, se_ratio=4, dropout_rate=0.0, stochastic_depth_prob=0.0, norm_type="batch"):
        super(MBConv1d, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)
        mid_channels = in_channels * expansion

        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            CustomNorm(mid_channels, norm_type),
            activation_func,
        ) if expansion > 1 else nn.Identity()

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False),
            CustomNorm(mid_channels, norm_type),
            activation_func,
        )

        self.se = SEBlock(mid_channels, max(1, int(mid_channels // se_ratio)), activation_func) if use_se else nn.Identity()
        self.project_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            CustomNorm(out_channels, norm_type),
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if self.use_residual else nn.Identity()

    def forward(self, x):
        identity = x if self.use_residual else None
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        x = self.dropout(x)
        if self.use_residual:
            x = self.stochastic_depth(x) + identity
        return x

class EfficientNet1DV2(nn.Module):
    def __init__(self, variant='s', input_channels=12, num_classes=77, activation='swish', se_ratio=4, base_depths=[1, 2, 4, 4, 6, 9, 15, 1], base_channels=[24, 24, 48, 64, 128, 160, 256, 1280], expansion_factors=[1, 6, 6, 6, 6, 6, 6], stochastic_depth_prob=0.0, dropout_rate=0.0, use_se=True, kernel_sizes=[3, 3, 3, 3, 3, 3, 3], strides=[2, 1, 2, 2, 2, 1, 2], norm_type="batch"):
        super(EfficientNet1DV2, self).__init__()
        config = get_backbone_config(variant)
        width_coefficient, depth_coefficient = config['width_coefficient'], config['depth_coefficient']

        # Apply coefficients to channels and depths
        channels = [max(1, int(c * width_coefficient)) for c in base_channels]
        depths = [max(1, math.ceil(d * depth_coefficient)) for d in base_depths]

        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=kernel_sizes[0] // 2, bias=False),
            CustomNorm(channels[0], norm_type),
            get_activation(activation),
        )

        # Constructing the blocks

        self.features = self._make_layers(channels, depths, kernel_sizes, strides, expansion_factors, se_ratio, activation, stochastic_depth_prob, dropout_rate, use_se, norm_type, variant=variant)
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], num_classes),
        )

        self.final_conv = nn.Conv1d(channels[-2], channels[-1], kernel_size=1, stride=1, padding=0)  # channels[-2] is the input to the final conv, channels[-1] is its output
        self.final_norm = CustomNorm( channels[-1], norm_type)

    def _make_layers(self, channels, depths, kernel_sizes, strides, expansion_factors, se_ratio, activation, stochastic_depth_prob, dropout_rate, use_se, norm_type, variant):
        layers = []
        in_channels = channels[0]
        for i, (out_channels, num_blocks) in enumerate(zip(channels[1::], depths[1::])):
            stride = strides[i]
            print(f'block {num_blocks}')
            print(f'in channels {in_channels}')
            print(f'out channels {out_channels}')
            if 'v2' in variant:
                for j in range(num_blocks):
                    if j > 0:  # Only the first block in each sequence uses the defined stride
                        stride = 1
                    block = MBConv1d(in_channels, out_channels, kernel_sizes[i], stride, expansion_factors[i], activation, use_se, se_ratio[i], dropout_rate, stochastic_depth_prob, norm_type) if i > 3 else FusedMBConv1d(in_channels, out_channels, kernel_sizes[i], stride, activation, use_se, se_ratio[i], dropout_rate, stochastic_depth_prob, norm_type)
                    layers.append(block)
                    in_channels = out_channels  # Update in_channels for the next block
            else:
                for j in range(num_blocks):
                    if j > 0:  # Only the first block in each sequence uses the defined stride
                        stride = 1
                    block = MBConv1d(in_channels, out_channels, kernel_sizes[i], stride, expansion_factors[i], activation, use_se, se_ratio[i], dropout_rate, stochastic_depth_prob, norm_type)
                    layers.append(block)
                    in_channels = out_channels  # Update in_channels for the next block                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.features(x)
        x = self.final_conv(x)
        x = self.final_norm(x)
        x = self.classifier(x)
        return x