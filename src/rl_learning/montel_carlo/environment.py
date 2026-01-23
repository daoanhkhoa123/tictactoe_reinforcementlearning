from typing import Tuple

from src.common import Action
from src.game.client import Client
from src.game.game import Game
from src.game.table import MarkType, TableFactory
from src.rl_learning.montel_carlo.montel_carlo import MontelCarloClient


class Environment(Game):
    ...