from typing import Tuple

from src.game.controller import BaseController
from src.game.interface import BaseInterface
from src.game.table import MarkType, Table


class Client:
    def __init__(self, name:str, mark_type:MarkType, interface:BaseInterface, controller: BaseController, *, max_trial: int = 5) -> None:
        self._name = name
        self._mark_type = mark_type
        self._controller = controller
        self._max_trial = max_trial
        self._interface = interface

    @property
    def name(self) -> str:
        return self._name

    @property
    def mark_type(self) -> MarkType:
        return self._mark_type

    @property
    def interface(self) -> BaseInterface:
        return self._interface

    @property
    def controller(self) -> BaseController:
        return self._controller

    
    def play(self, table: Table) -> None:
        state = table.get_table()

        trial = 0
        while trial < self._max_trial:
            trial += 1
            y, x =self.controller.decide(state)
            good_move = table.mark(y, x, self.mark_type)
            if good_move:
                self.interface.show(self, state)
                return


