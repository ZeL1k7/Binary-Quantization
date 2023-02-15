import torch
import torch.nn as nn


class SignNegative(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.sign(x)
        x[x == 0] = -1
        x = x.to(torch.int8)
        return x


class SignPositive(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.clamp(x, min=0)
        x[x > 0] = 1
        x = x.to(torch.int8)
        return x
