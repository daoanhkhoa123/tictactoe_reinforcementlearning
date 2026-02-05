import json
import random
from collections import deque
from typing import Deque, Dict, Iterator, List, Optional, Tuple

from numpy import dtype
from numpy.typing import NDArray

from src.game.controller import BaseController
from src.game.client import Client
from src.game.interface import BaseInterface
from src.game.table import MarkType, Table
from src.game.client import Client
from dataclasses import dataclass
from src.rl_learning.montelcarloo_treesearch.tree import Action, State, StateHashT, ActionNode, StateNode, MSTC


@dataclass
class MontelHyperParams:
    lr: float = 0.1
    decay_gamma: float = 0.9
    exp_rate: float = 0.3

class MontelCarloController(BaseController[State, Action]):
    def __init__(self) -> None:
        super().__init__()
        self.tree = MSTC()

    def model_call(self, model_input: State) -> Action:
        return self.tree.play(model_input)
    
    def feed_reward(self, reward:float):
        return self.tree.feed_reward(reward)