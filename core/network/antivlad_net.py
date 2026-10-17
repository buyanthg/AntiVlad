from torch import nn

from core.constants import *
from core.chess.action_encoder import ACTION_SIZE


class AntiVladNet(nn.Module):
    def __init__(self, out_channels, policy_channels, value_channels, value_hidden, num_residual_blocks):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=TOTAL_PLANES,out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_residual_blocks)]
        )
        self.policy_head = PolicyHead(out_channels, policy_channels, ACTION_SIZE)
        self.value_head = ValueHead(out_channels, value_channels, value_hidden)


    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.residual_blocks(output)
        policy = self.policy_head(output)
        value = self.value_head(output)
        return policy, value



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()


    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)

        output = self.conv2(output)
        output = self.bn2(output)

        output += residual
        output = self.relu(output)

        return output

class PolicyHead(nn.Module):
    def __init__(self, channels, policy_channels, action_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,out_channels=policy_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(policy_channels)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(policy_channels * 8 * 8, action_size)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.flatten(output)
        output = self.fc(output)

        return output


class ValueHead(nn.Module):
    def __init__(self, channels, value_channels, value_hidden):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,out_channels=value_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(value_channels)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(value_channels * 8 * 8, value_hidden)
        self.fc2 = nn.Linear(value_hidden, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.tanh(output)
        return output

def get_standart_model():
    return AntiVladNet(NETWORK_CHANNELS, POLICY_CHANNELS, VALUE_CHANNELS,
                       VALUE_HIDDEN, NUM_RESIDUAL_BLOCKS,)