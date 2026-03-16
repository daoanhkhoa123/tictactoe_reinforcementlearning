from typing import Optional

class Node: 
    def __init__(self) -> None:
        self.edges: list[Edge] = list()
        self.value:float = 0

    def connect(self, node: "Node"):
        edge = Edge(self, node)
        self.edges.append(edge)

    def connected_nodes(self):
        for edge in self.edges:
            assert edge.start == self
            yield edge.end

class Edge:
    def __init__(self, start:Node, end:Node) -> None:
        self.start = start
        self.end = end

class StateNode(Node):  ...
    
class ActionNode(Node):   ...

def get_all_possible_actions(state: StateNode) -> list[ActionNode]: ...

class Graph:
    def __init__(self) -> None:
        self.graph: dict[Node, dict[Node, Edge]] = {}

    def init_state(self, state:StateNode):
        assert state not in self.graph
        self.graph[state] = {}
        for action in get_all_possible_actions(state):
            self.graph[state][action] = Edge(state, action)

    def neighbors(self, node:Node):
        for neigh in self.graph[node]:
            yield neigh

    def connect(self, start:ActionNode, end:StateNode):
        self.graph[start][end] = Edge(start, end)
        self.backpropgate_add(end, end.value)

    def backpropgate_add(self, node: Node, value: float):
        for neigh in self.neighbors(node):
            neigh.value += value
            self.backpropgate_add(neigh, value)

