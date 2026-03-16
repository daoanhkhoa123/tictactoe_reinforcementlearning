from src.game.table import Table

class HashableTable(Table):
    def __hash__(self) -> int:
        return hash(str(self.get_table(False)))