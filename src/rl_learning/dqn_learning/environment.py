from typing import Sequence, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from src.common import emptycoords_from_table
from src.game.client import Client
from src.game.game import Game
from src.game.table import MarkType, Table, TableFactory

Action = Tuple[int, int]

class ActionSpace(Table):
    def total_ns(self) -> int:
        return self._tictactoe.shape[0] * self._tictactoe.shape[1]
    
    def action_space(self) -> Sequence[Action]:
        return emptycoords_from_table(self._tictactoe)
    
    def sample(self) -> Action:
        return np.random.choice(self.action_space())

    def invalid_actions(self) -> Sequence[Action]:
        ys, xs = np.where(self._tictactoe != MarkType.EMPTY)
        return list(zip(ys.astype(int), xs.astype(int)))

class ActionTblFactory(TableFactory):
    def __call__(self, *args: Any, **kwargs: Any) -> Table:
        return ActionSpace()

class Environment(Game):
    def __init__(self, client1: Client, client2: Client) -> None:
        super().__init__(client1, client2, ActionTblFactory())
    
    @property
    def info(sef):
        return ...

    @property
    def state(self):
        return self._table.get_table()

    def reset(self) -> Tuple[NDArray, Any]:
        self.reset_table()
        return self.state, self.info