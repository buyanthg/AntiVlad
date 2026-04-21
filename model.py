import torch
from torch import nn
import torch.nn.functional as F

# --- 1. АРХИТЕКТУРА (Твои классы с легкими правками) ---

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x)) + identity
        return F.relu(x)


class AntiVlad(nn.Module):
    def __init__(self, in_channels=13, hidden_channels=64, num_blocks=4):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(hidden_channels) for _ in range(num_blocks)])

        # Policy Head
        self.p_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.p_fc = nn.Linear(2 * 8 * 8, 4096)

        # Value Head
        self.v_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.v_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.input_conv(x))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.p_conv(x))
        p = self.p_fc(p.view(p.size(0), -1))
        # Softmax лучше делать в run_inference после маскировки

        v = F.relu(self.v_conv(x))
        v = F.relu(self.v_fc1(v.view(v.size(0), -1)))
        v = torch.tanh(self.v_fc2(v))

        return p, v