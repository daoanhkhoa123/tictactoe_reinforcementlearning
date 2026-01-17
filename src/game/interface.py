from game.client import Client
from numpy.typing import NDArray

class BaseInterface:
    def show(self, client:Client, state:NDArray) -> None:    ...