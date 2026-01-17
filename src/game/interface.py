from numpy.typing import NDArray

from game.client import Client


class BaseInterface:
    def show(self, client:Client, state:NDArray) -> None:    ...