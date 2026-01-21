from torch import nn
from src.common import index_to_coords
from src.game.controller import BaseController
from typing import Tuple
import torch

class DQN(nn.Module):
    def __init__(self, n_observations:int, n_actions:int, hidden:int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_observations, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, n_actions))
        
    def forward(self, x):
        return self.layers(x)
    
class ModelController(BaseController[torch.Tensor, torch.Tensor]):
    def __init__(self, model:nn.Module) -> None:
        super().__init__()
        self._model = model

    def model_call(self, model_input: torch.Tensor) -> torch.Tensor:
        return self._model(model_input)
    
    def post_processing(self, model_output: torch.Tensor) -> Tuple[int, int]:
        return index_to_coords(int(model_output))
