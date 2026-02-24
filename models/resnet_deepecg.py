import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Configuration for different ResNet architectures
resnet_configs = {
    "resnet18": {"block_channels": [64, 64, 128, 256, 512], "num_blocks": [2, 2, 2, 2]},
    "resnet34": {"block_channels": [64, 64, 128, 256, 512], "num_blocks": [3, 4, 6, 3]},
    "resnet50": {
        "block_channels": [64, 256, 512, 1024, 2048],
        "num_blocks": [3, 4, 6, 3],
    },
    "resnet101": {
        "block_channels": [64, 256, 512, 1024, 2048],
        "num_blocks": [3, 4, 23, 3],
    },
    "resnet152": {
        "block_channels": [64, 256, 512, 1024, 2048],
        "num_blocks": [3, 8, 36, 3],
    },
}


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        model,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation="relu",
        dropout=0.0,
        stochastic_depth=0.0,
        use_bias=True,
    ):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=use_bias,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = self.get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)
        self.stochastic_depth = stochastic_depth

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=use_bias,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            padding = (
                kernel_size // 2 - 1
            )  # Adjust the padding to match the spatial dimensions
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=padding,
                    bias=use_bias,
                ),
                nn.BatchNorm1d(out_channels),
                nn.AvgPool1d(kernel_size=stride),
            )
        else:
            self.shortcut = nn.Identity()
        # Initialize convolutions with He uniform
        self.apply(self.init_weights)

        self.model = model

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, mean=0, std=0.01)
            init.constant_(m.bias, 0)

    def get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "selu":
            return nn.SELU()
        elif activation == "mish":
            return Mish()
        elif activation == "swish":
            return Swish()
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")

    def forward(self, x):
        if self.training and self.stochastic_depth > 0.0:
            if torch.rand(1).item() < self.stochastic_depth:
                return x

        residual = self.shortcut(x)
        # if self.shortcut == nn.Identity():
        # residual = self.shortcut[0](residual)
        if self.model in ["resnet18", "resnet34"]:
            residual = F.avg_pool1d(residual, kernel_size=residual.size(-1))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual

        x = self.activation(x)
        x = self.dropout(x)

        return x


class ResNet1D(nn.Module):
    def __init__(
        self,
        input_channels,
        architecture="resnet18",
        activation="relu",
        dropout=0.0,
        use_batchnorm_padding_before_conv1=False,
        use_padding_pooling_after_conv1=False,
        stochastic_depth=0.0,
        kernel_sizes=None,
        strides=None,
        use_bias=True,
        out_neurons=92,
        out_activation="sigmoid",
        **kwargs,
    ):
        super(ResNet1D, self).__init__()

        config = resnet_configs[architecture]
        config.update(kwargs)

        if use_batchnorm_padding_before_conv1:
            self.bn_padding = nn.BatchNorm1d(input_channels)
            self.zero_padding_before_conv1 = nn.ConstantPad1d((0, 1), 0)

        self.stochastic_depth = stochastic_depth

        self.conv1 = nn.Conv1d(
            input_channels,
            config["block_channels"][0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=kernel_sizes[0] // 2,
            bias=use_bias,
        )
        self.bn1 = nn.BatchNorm1d(config["block_channels"][0])

        self.activation = self.get_activation(activation)

        self.dropout = nn.Dropout(p=dropout)

        if use_padding_pooling_after_conv1:
            self.zero_padding_after_conv1 = nn.ConstantPad1d((0, 1), 0)
            self.pooling_after_conv1 = nn.MaxPool1d(
                strides[1] + 1, stride=strides[1], padding=1
            )

        self.blocks = nn.Sequential(
            *[
                ResNetBlock(
                    architecture,
                    config["block_channels"][i],
                    config["block_channels"][i + 1],
                    kernel_size=kernel_sizes[i + 1],
                    stride=strides[i + 1],
                    activation=activation,
                    dropout=dropout,
                    stochastic_depth=stochastic_depth,
                    use_bias=use_bias,
                )
                for i in range(len(config["block_channels"]) - 1)
            ]
        )

        # Initialize convolutions with He uniform
        self.apply(self.init_weights)

        if out_neurons is not None:
            self.final_layer = nn.Linear(config["block_channels"][-1], out_neurons)
            if out_activation is not None:
                self.final_activation = self.get_activation(out_activation)

        # Initialize EMA
        self.ema = EMA(self)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, mean=0, std=0.01)
            init.constant_(m.bias, 0)

    def get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "selu":
            return nn.SELU()
        elif activation == "mish":
            return Mish()
        elif activation == "swish":
            return Swish()
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")

    def forward(self, x):
        if hasattr(self, "bn_padding") and hasattr(self, "zero_padding_before_conv1"):
            x = self.bn_padding(x)
            x = self.zero_padding_before_conv1(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        if hasattr(self, "zero_padding_after_conv1") and hasattr(
            self, "pooling_after_conv1"
        ):
            x = self.zero_padding_after_conv1(x)
            x = self.pooling_after_conv1(x)

        x = self.blocks(x)

        if hasattr(self, "final_layer"):
            x = x.mean(dim=-1)  # Global average pooling
            x = self.final_layer(x)
            # if hasattr(self, 'final_activation'):
            # x = self.final_activation(x)
        return x

    def apply_ema(self):
        # Update EMA parameters after each training step
        self.ema.update()

    def set_to_eval_mode(self):
        # Set the model to evaluation mode using EMA parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = self.ema.shadow[name]

