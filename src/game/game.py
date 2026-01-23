from src.game.client import Client
from src.game.table import MarkType, TableFactory


class Game:
    def __init__(self, client1: Client, client2:Client, table_factory: TableFactory = TableFactory()) -> None:
        self._client1 = client1
        self._client2 = client2
        
        self._table_factory = table_factory
        self._table = table_factory()

    @property
    def client1(self) -> Client:
        return self._client1

    @property
    def client2(self) -> Client:
        return self._client2

    def reset_table(self) -> None:
        self._table = self._table_factory()

    def hajime(self):
        if self._table is None:
            raise KeyError("Please init the table first")

        while (winner:=self._table.get_winner()) == MarkType.EMPTY:
            if self._table.is_full():
                break
            self.client1.play(self._table)

            if self._table.is_full():
                break
            self.client2.play(self._table)

        return winner
    
    def owari(self, winner: MarkType):
        w ="[EMPTY]"
        if winner == self.client1.mark_type:
            w = self.client1.name
        if winner == self.client2.mark_type:
            w  = self.client2.name
        else:
            print("Draw!")
            return
        
        print("Player", w, "won the game!")