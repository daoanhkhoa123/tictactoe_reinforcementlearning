import random
import time
from typing import List, Tuple

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

from src.game.client import Client
from src.game.controller import BaseController
from src.game.interface import BaseInterface
from src.game.table import MarkType

#####################################
#   QUICK MACRO
##########

def coords_to_index(row: int, col: int, ncols: int = 7) -> int:
    return row * ncols + col

def index_to_coords(idx: int, ncols: int = 7) -> Tuple[int, int]:
    return idx // ncols, idx % ncols

def emptycoords_from_table(table: NDArray) -> List[Tuple[int, int]]:
    ys, xs = np.where(table == MarkType.EMPTY)
    return list(zip(ys.astype(int), xs.astype(int)))

def emptyindex_from_table(table: NDArray, ncols: int = 7) -> List[int]:
    return [coords_to_index(y, x, ncols)
        for y, x in emptycoords_from_table(table)]


####################
#   SOME CONTROLLER CLASS
#########

class RandomController(BaseController[List[int], int]):
    def __init__(self, sleep_time: float = 3) -> None:
        super().__init__()
        self._sleep_time = sleep_time

    def pre_processing(self, input_state: NDArray) -> List[int]:
        return emptyindex_from_table(input_state)

    def model_call(self, model_input: List[int]) -> int:
        time.sleep(self._sleep_time)
        return random.choice(model_input)

    def post_processing(self, model_output: int) -> Tuple[int, int]:
        coords = index_to_coords(model_output)
        print("Bot chose:", coords)
        return coords
        
class HumanController(BaseController[NDArray, Tuple[int, int]]):
    def __init__(self, nrows: int = 7, ncols: int = 7, trial: int = 3) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._trial = trial

    def model_call(self, model_input: NDArray) -> Tuple[int, int]: # type: ignore
        trial = 0
        while trial < self._trial:
            trial += 1
            raw = input("Enter move as: row col > ").strip()

            try:
                y_str, x_str = raw.split()
                y, x = int(y_str), int(x_str)
            except ValueError:
                print("Invalid format. Use: row col")
                continue

            if not (0 <= y < self._nrows and 0 <= x < self._ncols):
                print("Coordinates out of bounds.")
                continue
            
            return y, x

############################
#   Interface
#######################

class CMDInterface(BaseInterface):
    """
    Command-line interface for rendering the game state and prompting users.
    """

    @staticmethod
    def to_clienttext(mark_type: MarkType) -> str:
        if mark_type == MarkType.BLU:
            return "Blue"
        if mark_type == MarkType.RED:
            return "Red"
        if mark_type == MarkType.EMPTY:
            return "Empty"
        raise ValueError(f"Unknown MarkType: {mark_type}")

    @staticmethod
    def to_statetext(mark_type: MarkType) -> str:
        if mark_type == MarkType.BLU:
            return "B"
        if mark_type == MarkType.RED:
            return "R"
        if mark_type == MarkType.EMPTY:
            return "."
        raise ValueError(f"Unknown MarkType: {mark_type}")

    def show(self, client: "Client", state: NDArray) -> None:
        """
        Render the board and show whose turn it is.
        """
        nrows, ncols = state.shape

        print()
        print(f"Player: {client.name} ({self.to_clienttext(client.mark_type)})")
        print()

        # column header
        print("   " + " ".join(f"{c:2d}" for c in range(ncols)))

        for y in range(nrows):
            row = " ".join(
                f" {self.to_statetext(MarkType(state[y, x]))}"
                for x in range(ncols)
            )
            print(f"{y:2d} {row}")

        print()

class EmptyInterface(BaseInterface):
    def show(self, client: Client, state: NDArray) -> None:
        print("Bot is thinking...")


########################
#   FOR FASTER BUILD
##########################


@dataclass
class ClientArgs:
    name:str
    controller: BaseController
    interface: BaseInterface

def build_2clients(blu: ClientArgs, red: ClientArgs) -> Tuple[Client, Client]:
    return (
        Client(
            name=blu.name,
            mark_type=MarkType.BLU,
            interface=blu.interface,
            controller=blu.controller,
        ),
        Client(
            name=red.name,
            mark_type=MarkType.RED,
            interface=red.interface,
            controller=red.controller,
        ),
    )

################
#   For reinforcement learning
########################
Action = Tuple[int, int]