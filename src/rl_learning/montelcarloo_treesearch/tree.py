from collections import deque
from typing import Deque

class Node:
    def __init__(self) -> None:
        """ NOTE: can not remove edge """
        self._edges:Deque[Edge] = deque()
        self.value: float = 0

    @property
    def edges(self):
        return self._edges
    

class Edge:
    def __init__(self, start: Node, end:Node) -> None:
        self._start = start
        self._end = end

    @property
    def start(self):
        return self.start
    
    @property
    def end(self):
        return self._end

    
class StateNode(Node): ...
class ActionNode(Node): ...
def getall_possible_actions(state: StateNode) -> list[ActionNode]:  ...

class MSTC:
    def __init__(self) -> None:
        self.state_map: dict[StateNode, Deque[ActionNode]] = {}

    def init_state(self, state: StateNode):
        self.state_map[state] = deque()
        for action in getall_possible_actions(state):
            self.state_map[state].append(action)