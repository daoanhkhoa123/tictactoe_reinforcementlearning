from typing import Tuple

from src.common import Action
from src.game.client import Client
from src.game.game import Game
from src.game.table import MarkType, TableFactory
from src.rl_learning.montel_carlo.montel_carlo import MontelCarloClient


class Environment(Game):
    def __init__(self, client1: MontelCarloClient, client2: MontelCarloClient | Client) -> None:
        super().__init__(client1, client2, TableFactory())
        self._client1 = client1
        self._client2 = client2

    @property
    def client1(self) -> MontelCarloClient:
        return self._client1

    @property
    def client2(self) -> MontelCarloClient | Client:
        return self._client2

    def owari(self, winner: MarkType):
        c1_reward = 1 if winner == self.client1.mark_type else 0
        c2_reward = 1 if winner == self.client2.mark_type else 0

        self.client1.controller.feed_reward(c1_reward)

        if isinstance(self.client2, MontelCarloClient):
            self.client2.controller.feed_reward(c2_reward)

        super().owari(winner)