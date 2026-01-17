from src.game.table import MarkType, Table
from typing import Tuple
from src.game.controller import BaseController
from src.game.interface import BaseInterface

class Client:
    def __init__(self, name:str, mark_type:MarkType, interface:BaseInterface, controller: BaseController, *, max_trial: int = 5) -> None:
        self._name = name
        self._mark_type = mark_type
        self._controller = controller
        self._max_trial = max_trial
        self._interface = interface

    @property
    def name(self):
        return self._name

    @property
    def mark_type(self):
        return self._mark_type
    
    def play(self, table: Table) -> None:
        state = table.get_table()
        self._interface.show(self, state)

        trial = 0
        while trial < self._max_trial:
            trial += 1
            y, x =self._controller.decide(state)
            good_move = table.mark(y, x, self.mark_type)
            if good_move:
                return


def build_2clients( name1: str, name2: str,
        controller1: BaseController, controller2: BaseController,
        interface1: BaseInterface, interface2: BaseInterface ) -> Tuple[Client, Client]:
    
    client1 = Client(
        name=name1,
        mark_type=MarkType.BLU,
        interface=interface1,
        controller=controller1,
    )

    client2 = Client(
        name=name2,
        mark_type=MarkType.RED,
        interface=interface2,
        controller=controller2,
    )

    return client1, client2