from typing import Optional

from numpy.typing import NDArray

from src.game.controller import BaseController
from src.game.client import Client
from src.game.table import MarkType
from src.game.client import Client
from src.rl_learning.montelcarloo_treesearch.tree import Action, State, MCTS, MCTSParams

class MTCSController(BaseController[State, Action]):
    DEFAULT_INNER_STATE = MarkType.BLU

    def __init__(self, mark_type: MarkType, mcts: MCTS) -> None:
        super().__init__()
        self.tree = mcts
        self.mark_type = mark_type

    def pre_processing(self, input_state: NDArray) -> State:
        if self.mark_type != self.DEFAULT_INNER_STATE:
            input_state *= -1
        
        return input_state

    def model_call(self, model_input: State) -> Action:
        return self.tree.play(model_input)
    
    def feed_reward(self, reward:float, last_state: State):
        return self.tree.feed_reward(reward, last_state)
    
class MTCSCLient(Client):   ...

def build_mtcslient(client:Client, params: Optional[MCTSParams] = None)-> MTCSCLient:
    mtcs = MCTS(params or MCTSParams())
    controller =  MTCSController(client.mark_type, mtcs)
    return MTCSCLient(client.name, client.mark_type, client.interface, controller)