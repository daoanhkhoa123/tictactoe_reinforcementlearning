import random
from typing import Tuple

from numpy.typing import NDArray
import torch
from torch import nn
import math
from src.common import index_to_coords
from src.game.controller import BaseController
from src.rl_learning.dqn_learning.environment import Environment

class ModelDQN(nn.Module):
    def __init__(self, n_observations:int, n_actions:int, hidden:int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_observations, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, n_actions))
        
    def forward(self, x):
        return self.layers(x)
    


class ControllerDQN(BaseController[torch.Tensor, torch.Tensor]):
    def __init__(self, model: nn.Module, env: Environment, device: torch.device,
        *, eps_start: float, eps_end: float, eps_decay: float) -> None:
       
        super().__init__()
        self.policy_net = model
        self.env = env
        self.device = device

        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def pre_processing(self, input_state: NDArray) -> torch.Tensor:
        return torch.as_tensor(input_state, device=self.device, dtype=torch.float32)

    def model_call(self, model_input: torch.Tensor) -> torch.Tensor:
        self.steps_done += 1

        eps_threshold = self.eps_end + (
            self.eps_start - self.eps_end
        ) * math.exp(-self.steps_done / self.eps_decay)

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(model_input).argmax(dim=1, keepdim=True)
        else:
            action = self.env.action_space.sample()
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def post_processing(self, model_output: torch.Tensor) -> Tuple[int, int]:
        return index_to_coords(int(model_output.item()))
