from enum import IntEnum
from typing import Any
import numpy as np
from numpy.typing import NDArray


class MarkType(IntEnum):
    RED = 1
    BLU = -1 
    EMPTY = 0

class Table:
    def __init__(self) -> None:
        self._tictactoe = np.zeros((7, 7), dtype=np.int8)

    def get_table(self) -> NDArray:
        return self._tictactoe.copy()
    
    def is_full(self) -> bool:
        return np.all(self._tictactoe != MarkType.EMPTY).item()
    
    def get_winner(self) -> MarkType:
        b = self._tictactoe

        # Horizontal
        for y in range(7):
            for x in range(4):
                if b[y,x] != 0 and b[y,x] == b[y,x+1] == b[y,x+2] == b[y,x+3]:
                    return MarkType(b[y,x])

        # Vertical
        for x in range(7):
            for y in range(4):
                if b[y,x] != 0 and b[y,x] == b[y+1,x] == b[y+2,x] == b[y+3,x]:
                    return MarkType(b[y,x])

        # Diagonal /
        for y in range(4):
            for x in range(4):
                if b[y,x] != 0 and b[y,x] == b[y+1,x+1] == b[y+2,x+2] == b[y+3,x+3]:
                    return MarkType(b[y,x])

        # Diagonal \
        for y in range(4):
            for x in range(3, 7):
                if b[y,x] != 0 and b[y,x] == b[y+1,x-1] == b[y+2,x-2] == b[y+3,x-3]:
                    return MarkType(b[y,x])

        return MarkType.EMPTY

    def mark(self, y: int, x: int, mark: MarkType) -> bool:
        if mark == MarkType.EMPTY:
            raise ValueError("Cannot mark EMPTY")

        if not (0 <= y < 7 and 0 <= x < 7):
            return False

        if self._tictactoe[y, x] != MarkType.EMPTY:
            return False

        self._tictactoe[y, x] = mark
        return True
    

class TableFactory:
    def __call__(self, *args: Any, **kwargs: Any) -> Table:
        return Table(*args, **kwargs)