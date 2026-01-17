from __future__ import annotations

from numpy.typing import NDArray
from typing import TYPE_CHECKING

# becuse of ciruclar import errors, and we actually just dont need the class
# we just need its typing only
if TYPE_CHECKING:
    from src.game.client import Client


class BaseInterface:
    def show(self, client: "Client", state: NDArray) -> None:
        ...
