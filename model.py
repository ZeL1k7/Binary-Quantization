import torch
import torch.nn as nn


class SignNegative(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sign(x)
        x[x == 0] = -1
        x = x.to(torch.float32)
        return x


class SignPositive(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0)
        x[x > 0] = 1
        x = x.to(torch.float32)
        return x


class BinaryModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        eval_mode: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.eval_mode = eval_mode
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hid_channels,
            kernel_size=9,
            padding=0,
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(
            in_channels=hid_channels,
            out_channels=hid_channels,
            kernel_size=9,
            padding=0,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(
            in_channels=hid_channels,
            out_channels=hid_channels,
            kernel_size=9,
            padding=0,
        )
        self.pool3 = nn.AdaptiveMaxPool1d(output_size=1)
        self.linear1 = nn.Linear(in_features=hid_channels, out_features=out_channels)
        self.activations = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Sigmoid()]

    def forward(self, x: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        if eval_mode:
            self.activations = [
                SignNegative(),
                SignNegative(),
                SignNegative(),
                SignPositive(),
            ]
        batch_size = x.size()[0]

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activations[0](x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activations[1](x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(batch_size, -1)
        x = self.activations[2](x)
        x = self.linear1(x)
        x = self.activations[3](x)

        return x
