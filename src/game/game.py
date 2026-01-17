from typing import Optional

from src.game.client import Client
from src.game.table import MarkType, Table


class Game:
    def __init__(self, client1: Client, client2:Client) -> None:
        self._client1 = client1
        self._client2 = client2
        self._table:Optional[Table] = None

    @property
    def table(self) -> Table:
        if self._table is None:
            raise KeyError("Please init the table first")
        return self._table

    @property
    def client1(self) -> Client:
        return self._client1

    @property
    def client2(self) -> Client:
        return self._client2
    
    def hajime(self):
        self._table = Table()
        
        while (winner:=self._table.get_winner()) == MarkType.EMPTY:
            self.client1.play(self._table)
            self.client2.play(self._table)

        return winner
    
    def owari(self, winner: MarkType):
        w ="[EMPTY]"
        if winner == self.client1.mark_type:
            w = self.client1.name
        if winner == self.client2.mark_type:
            w  = self.client2.name

        print("Player", w, "won the game!")