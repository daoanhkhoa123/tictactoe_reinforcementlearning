from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

# becuse of ciruclar import errors, and we actually just dont need the class
# we just need its typing only
if TYPE_CHECKING:
    from src.game.client import Client


class BaseInterface:
    def show(self, client: "Client", state: NDArray) -> None:
        ...
