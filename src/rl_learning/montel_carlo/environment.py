from typing import Sequence, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from src.common import emptycoords_from_table, Action
from src.game.client import Client
from src.game.game import Game
from src.game.table import MarkType, Table, TableFactory
from src.rl_learning.montel_carlo.montel_carlo import MontelCarloClient, MontelCarloController

class Environment(Game):
    def __init__(self, client1: Client, client2: MontelCarloClient) -> None:
        super().__init__(client1, client2, TableFactory())
        self._client2 = client2
    
    @property
    def client2(self) -> MontelCarloClient:
        return self._client2

    def owari(self, winner: MarkType):
        if winner == self.client2.mark_type:
            self.client2.controller.feed_reward(1)
        else:
            self.client2.controller.feed_reward(-1)

        super().owari(winner)
    
