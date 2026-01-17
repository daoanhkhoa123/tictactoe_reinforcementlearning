from numpy.typing import NDArray

from src.game.client import Client


class BaseInterface:
    def show(self, client:Client, state:NDArray) -> None:    ...