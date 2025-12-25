# +
import copy
from typing import Tuple
from typing import Callable

import torch
from torch import nn


def symlog(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return torch.sign(x) * torch.log(1 + alpha * torch.abs(x)) / alpha


def symexp(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(alpha * torch.abs(x)) - 1) / alpha


class SymLog(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return symlog(x, self.alpha)


class SymExp(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return symexp(x, self.alpha)


# +
class LambdaLayer(nn.Module):
    lambd: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class FrameNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int):
        super(FrameNet, self).__init__()
        f, h, w, c = input_shape

        self.net = nn.Sequential(
            # Swap NxFxHxWxC -> NxFxCxHxW
            LambdaLayer(lambda t: t.permute(0, 1, 4, 2, 3).float() / 255),
            # NxFxCxHxW -> (N*F)xCxHxW
            LambdaLayer(lambda t: t.view(t.shape[0] * f, c, h, w)),
            nn.LazyBatchNorm2d(momentum=0.001, affine=True, track_running_stats=True),
            # Conv layer
            nn.LazyConv2d(out_channels=16, kernel_size=3, stride=2, bias=False),
            # nn.LazyBatchNorm2d(momentum=0.001, affine=True, track_running_stats=True),
            nn.GELU(),
            # nn.Dropout2d(0.25),
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, bias=False),
            # nn.LazyBatchNorm2d(momentum=0.001, affine=True, track_running_stats=True),
            nn.GELU(),
            # nn.Dropout2d(0.25),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, bias=False),
            # nn.LazyBatchNorm2d(momentum=0.001, affine=True, track_running_stats=True),
            nn.GELU(),
            # nn.Dropout2d(0.25),
            nn.LazyConv2d(out_channels=128, kernel_size=3, stride=2, bias=False),
            # nn.LazyBatchNorm2d(momentum=0.001, affine=True, track_running_stats=True),
            nn.GELU(),
            # nn.Dropout2d(0.25),
            # (N*F)xCxHxW -> NxFxCxHxW
            # NxFxCxHxW -> NxFx(C*H*W)
            LambdaLayer(lambda t: t.view(t.shape[0] // f, f, *t.shape[1:])),
            nn.Flatten(start_dim=2),
            # Bottleneck reduce dim to per frame
            # NxFx(C*H*W) -> NxFx256
            nn.LazyLinear(256, bias=False),
            LambdaLayer(lambda t: t.permute(0, 2, 1)),
            # nn.LazyBatchNorm1d(momentum=0.001, affine=True, track_running_stats=True),
            LambdaLayer(lambda t: t.permute(0, 2, 1)),
            nn.GELU(),
            # nn.Dropout1d(0.25),
            # NxFx256 -> Nx(F*256)
            nn.Flatten(start_dim=1),
            # Output
            nn.LazyLinear(out_features=output_dim, bias=False),
            # nn.LazyBatchNorm1d(momentum=0.001, affine=True, track_running_stats=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class RAMNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int):
        super(RAMNet, self).__init__()
        assert len(input_shape) == 2

        self.net = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.Dropout1d(0.25),
            nn.LazyLinear(256),
            nn.LazyBatchNorm1d(),
            nn.GELU(),
            nn.Dropout1d(0.25),
            nn.LazyLinear(64),
            nn.LazyBatchNorm1d(),
            nn.GELU(),
            nn.LazyBatchNorm1d(),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(out_features=output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x.float())


class ActorCriticNet(nn.Module):
    """MLP Dueling DQN network."""

    def __init__(self, input_shape: Tuple[int, ...], num_actions: int):
        """
        Args:
            input_shape: the shape of the input tensor to the neural network
            num_actions: the number of units for the output liner layer
        """
        super().__init__()

        self.backbone = FrameNet(input_shape=input_shape, output_dim=1024)

        self.actor = nn.Sequential(
            # nn.Dropout1d(0.25),
            nn.LazyLinear(num_actions),
            nn.Softmax(dim=1),
        )

        self.value = nn.Sequential(
            # nn.Dropout1d(0.25),
            nn.LazyLinear(1),
        )

        # Bind lazy layer sizes
        # x = torch.zeros(3, input_shape, dtype=torch.uint8)
        # self.forward(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given state, return state-action value for all possible actions"""
        features = self.backbone(x)
        return self.actor(features), self.value(features)


class DuelingDQNNet(nn.Module):
    """MLP Dueling DQN network."""

    def __init__(self, input_shape: Tuple[int, ...], num_actions: int, hidden_dim: int):
        """
        Args:
            input_shape: the shape of the input tensor to the neural network
            num_actions: the number of units for the output liner layer
        """
        super().__init__()

        self.num_actions = num_actions

        if len(input_shape) == 2:
            self.backbone = RAMNet(input_shape=input_shape, output_dim=128)
        else:
            self.backbone = FrameNet(input_shape=input_shape, output_dim=hidden_dim)

        self.advantage_head = nn.LazyLinear(num_actions)
        self.value_head = nn.LazyLinear(1)

        # Bind lazy layer sizes
        with torch.no_grad():
            self.train(False)
            x = torch.zeros(1, *input_shape, dtype=torch.uint8)
            self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given state, return state-action value for all possible actions"""

        features = self.backbone(x)
        advantages = self.advantage_head(features)  # [batch_size, num_actions]
        values = self.value_head(features)  # [batch_size, 1]

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))  # [batch_size, num_actions]

        return q_values


class MarioFrameNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.online = FrameNet(input_dim, output_dim)
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)


class DuelingDDQNNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int, hidden_dim: int):
        super().__init__()
        self.online = DuelingDQNNet(input_shape, num_actions, hidden_dim)
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, model):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)
