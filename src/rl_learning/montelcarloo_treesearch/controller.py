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
from src.rl_learning.montelcarloo_treesearch.tree import Action, State, StateHashT, ActionNode, StateNode, MCTS

class MTCSController(BaseController[State, Action]):
    DEFAULT_INNER_STATE = MarkType.BLU

    def __init__(self, mark_type: MarkType, utc_cons:float) -> None:
        super().__init__()
        self.tree = MCTS(utc_cons)
        self.mark_type = mark_type

    def pre_processing(self, input_state: NDArray) -> State:
        if self.mark_type != self.DEFAULT_INNER_STATE:
            input_state *= self.DEFAULT_INNER_STATE.value
        
        return input_state

    def model_call(self, model_input: State) -> Action:
        return self.tree.play(model_input)
    
    def feed_reward(self, reward:float, last_state: State):
        return self.tree.feed_reward(reward, last_state)
    
class MTCSCLient(Client):   ...

def build_mtcslient(client:Client, utc_cons:float)-> MTCSCLient:
    controller =  MTCSController(client.mark_type, utc_cons)
    return MTCSCLient(client.name, client.mark_type, client.interface, controller)