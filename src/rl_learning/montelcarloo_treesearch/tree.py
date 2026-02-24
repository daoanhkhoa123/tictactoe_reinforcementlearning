import math
from collections import deque
from enum import IntEnum, auto
from typing import Deque, Generic, Iterator, Optional, TypeVar

from numpy.typing import NDArray
from typing_extensions import Self
from dataclasses import dataclass
from src.common import Action, emptycoords_from_table
import pickle

##################
#   CONNECTIONS
##################

class CONN_TYPE(IntEnum):
    FORWARD = auto()
    BACKWARD = auto()


##################
#   EDGE TYPES
##################

S = TypeVar("S", bound="Node")
D = TypeVar("D", bound="Node")


class Edge(Generic[S, D]):
    def __init__(self, src: S, dst: D) -> None:
        self._src = src
        self._dst = dst

    @property
    def src(self) -> S:
        return self._src

    @property
    def dst(self) -> D:
        return self._dst


##################
#   NODE BASE
##################

ConnectedToT = TypeVar("ConnectedToT", bound="Node")
import uuid
class Node(Generic[ConnectedToT]):
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.value: float = 0.0
        self.visited_time: int = 0
        self._forward: Deque[Edge[Self, ConnectedToT]] = deque()
        self._backward: Deque[Edge[ConnectedToT, Self]] = deque()

    def connect(self, node: ConnectedToT) -> None:
        e: Edge[Self, ConnectedToT] = Edge(self, node)
        self._forward.append(e)
        node._backward.append(e)

    def get_neighbors(self, con_type: CONN_TYPE = CONN_TYPE.FORWARD) -> Iterator[ConnectedToT]:
        if con_type is CONN_TYPE.FORWARD:
            for e in self._forward:
                yield e.dst
        else:
            for e in self._backward:
                yield e.src

    def utc_value(self, parent_count: int, constant: float) -> float:
        if self.visited_time == 0:
            return float("inf")

        return (
            self.value / self.visited_time
            + constant * math.sqrt(math.log(parent_count) / self.visited_time)
        )


##################
#   STATE TYPES
##################

class StateHashT(str):
    ...


State = NDArray


def get_statehash(array: State) -> StateHashT:
    return StateHashT(array.tobytes())


##################
#   TREE NODES
##################

class StateNode(Node["ActionNode"]):
    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state

    def __str__(self) -> str:
        return f"StateNode state= {self.state}, value={self.value}, visits={self.visited_time}"

class ActionNode(Node["StateNode"]):
    def __init__(self, action: Action) -> None:
        super().__init__()
        self.action = action

    def __str__(self) -> str:
        return f" ActionNode {self.action} value={self.value}, visits={self.visited_time}"


##################
#   ACTION SPACE
##################

getall_possible_actions = emptycoords_from_table

##################
#   MEMORY
##################

@dataclass
class LastMemory:
    last_action_node: Optional[ActionNode] = None
    last_state_node: Optional[StateNode] = None

    def reset(self) -> None:
        self.last_action_node = None
        self.last_state_node = None


##################
#   MCTS CORE
##################

class MCTS:
    def __init__(self, utc_cons: float) -> None:
        self._utc_cons = utc_cons
        self._state_map: dict[StateHashT, StateNode] = {}
        self._memory = LastMemory()

    ##################
    #   PROPERTIES
    ##################

    @property
    def state_map(self):
        return self._state_map

    @property
    def utc_cons(self):
        return self._utc_cons

    @property
    def memory(self):
        return self._memory

    @property
    def last_action_node(self):
        return self.memory.last_action_node

    @property
    def last_state_node(self):
        return self.memory.last_state_node


    def in_state_map(self, state: State):
        return get_statehash(state) in self.state_map

    ##################
    #   TRAVERSAL
    ##################

    def bfs_traversal(self, start: State):
        queue = deque()
        queue.append(self.get_statenode(start))

        while queue:
            node = queue.popleft()
            yield node
            for neigh in node.get_neighbors():
                queue.append(neigh)

    ##################
    #   BACKEND
    ##################

    def get_statenode(self, state: State) -> StateNode:
        return self.state_map[get_statehash(state)]

    def set_statenode(self, state: State, state_node: StateNode) -> None:
        self.state_map[get_statehash(state)] = state_node

    def backprogate_add(self, node: Node, value: float) -> None:
        for prev in node.get_neighbors(CONN_TYPE.BACKWARD):
            prev.value += value
            self.backprogate_add(prev, value)

    def choose_best_actionode(self, state: State) -> ActionNode:
        state_node = self.get_statenode(state)
        return max(
            state_node.get_neighbors(),
            key=lambda a: a.utc_value(state_node.visited_time, self.utc_cons),
        )

    def connect_next_state(self, next_state_node: StateNode) -> None:
        if self.memory.last_action_node is None:
            raise ValueError("No previous action")

        action = self.memory.last_action_node
        action.connect(next_state_node)
        action.value += next_state_node.value
        self.backprogate_add(action, next_state_node.value)

    ##############
    #   FOR OPTIMIZATION
    ##############
    def _register_state(self, state: State):
        if not self.in_state_map(state):
            state_node = StateNode(state)
            self.set_statenode(state, state_node)

            for action in getall_possible_actions(state):
                state_node.connect(ActionNode(action))

        else:
            state_node = self.get_statenode(state)

        if self.memory.last_action_node is not None:
            self.connect_next_state(state_node)
        self.memory.last_state_node = state_node
        self.memory.last_state_node.visited_time += 1

    # def _register_state(self, state:State):
    #     k = 5
    #     alpha = 0.5

    #     state_node = self.get_statenode(state)
    #     max_children = int(k * (state_node.visited_time ** alpha))
    #     current = len(list(state_node.get_neighbors()))

    #     if current >= max_children:
    #         return
        
    #     for action in getall_possible_actions(state):
    #         if current >= max_children:
    #             return
            
    #         state_node.connect(ActionNode(action))
    #         current += 1

    def _prune(self):
        prune_threshold = 2
        to_del = [h for h, n in self.state_map.items() if n.visited_time <= prune_threshold]
        for td in to_del:
            del self.state_map[td]

    ##################
    #    USAGE
    ##################

    def play(self, state: State) -> Action:
        self._register_state(state)

        self.memory.last_action_node = self.choose_best_actionode(state)
        self.memory.last_action_node.visited_time += 1
        return self.memory.last_action_node.action

    def feed_reward(self, reward: float, register_state: State) -> None:
        if self.memory.last_state_node is None:
            raise ValueError("Run the model first")

        # we have to also store the finished state for reward
        if register_state is not None:
            self._register_state(register_state)

        self.memory.last_state_node.value += reward
        self.backprogate_add(self.memory.last_state_node, reward)
      

    # ==================
    #   SERIALIZATION
    # ==================

    def _to_dict(self):
        nodes = {}
        edges = []

        for state_node in self.state_map.values():
            nodes[state_node.id] = {
                "type": "state",
                "state": state_node.state.tolist(),
                "value": state_node.value,
                "visits": state_node.visited_time,
            }

            for action_node in state_node.get_neighbors():
                nodes[action_node.id] = {
                    "type": "action",
                    "action": action_node.action,
                    "value": action_node.value,
                    "visits": action_node.visited_time,
                }
                edges.append((state_node.id, action_node.id))

                for next_state in action_node.get_neighbors():
                    edges.append((action_node.id, next_state.id))

        return {
            "utc_cons": self.utc_cons,
            "nodes": nodes,
            "edges": edges,
        }

    def save(self, path: str) -> None:
        """Save this MCTS instance to disk using pickle + manual dict representation."""
        data = self._to_dict()
        with open(path, "wb") as f:
            pickle.dump(data, f)


    @classmethod
    def load(cls, path: str) -> "MCTS":
        import numpy as np 
        """Load an MCTS instance from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise TypeError("Loaded data is not a dictionary")

        utc_cons = data["utc_cons"]
        nodes_dict = data["nodes"]
        edges = data["edges"]

        mcts = cls(utc_cons=utc_cons)
        mcts._state_map.clear()           # just in case

        # 1. Create all nodes (without connections yet)
        id_to_node = {}

        for node_id, info in nodes_dict.items():
            if info["type"] == "state":
                state = np.array(info["state"])   # assuming numpy array
                node = StateNode(state)
                node.id = node_id
                node.value = info["value"]
                node.visited_time = info["visits"]
                mcts._state_map[get_statehash(state)] = node
            else:  # action
                action = info["action"]           # assuming Action is pickle-able
                node = ActionNode(action)
                node.id = node_id
                node.value = info["value"]
                node.visited_time = info["visits"]

            id_to_node[node_id] = node

        # 2. Rebuild connections using edges
        for src_id, dst_id in edges:
            src = id_to_node[src_id]
            dst = id_to_node[dst_id]
            src.connect(dst)   # this also adds backward edge

        # 3. Memory is usually transient → reset or leave empty
        mcts.memory.reset()

        return mcts