import math
import pickle
import random
import uuid
from collections import deque
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Deque, Generic, Iterator, Optional, TypeVar, Dict, Set

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from src.common import Action, emptycoords_from_table


##################
#   CONNECTIONS
##################

class CONN_TYPE(IntEnum):
    FORWARD = auto()
    BACKWARD = auto()


##################
#   EDGE
##################

S = TypeVar("S", bound="Node")
D = TypeVar("D", bound="Node")


class Edge(Generic[S, D]):
    __slots__ = ("_src", "_dst")

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


class Node(Generic[ConnectedToT]):
    __slots__ = ("id", "value", "visited_time", "_forward", "_backward")

    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.value: float = 0.0
        self.visited_time: int = 0
        self._forward: Deque[Edge[Self, ConnectedToT]] = deque()
        self._backward: Deque[Edge[ConnectedToT, Self]] = deque()

    def connect(self, node: ConnectedToT) -> None:
        e = Edge(self, node)
        self._forward.append(e)
        node._backward.append(e)

    def get_neighbors(
        self, con_type: CONN_TYPE = CONN_TYPE.FORWARD
    ) -> Iterator[ConnectedToT]:
        if con_type is CONN_TYPE.FORWARD:
            for e in self._forward:
                yield e.dst
        else:
            for e in self._backward:
                yield e.src

    def utc_value(self, parent_visits: int, constant: float) -> float:
        if self.visited_time == 0:
            return float("inf")
        return (
            self.value / self.visited_time
            + constant * math.sqrt(math.log(parent_visits + 1) / self.visited_time)
        )


##################
#   STATE
##################

State = NDArray


def get_statehash(state: State) -> bytes:
    return state.data.tobytes()


##################
#   TREE NODES
##################

class StateNode(Node["ActionNode"]):
    __slots__ = ("state", "hash", "expanded")

    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state
        self.hash = get_statehash(state)
        self.expanded = False


class ActionNode(Node["StateNode"]):
    __slots__ = ("action",)

    def __init__(self, action: Action) -> None:
        super().__init__()
        self.action = action


##################
#   MEMORY
##################

@dataclass
class LastMemory:
    last_action_node: Optional[ActionNode] = None
    last_state_node: Optional[StateNode] = None
    first_root: Optional[State] = None

    def reset(self) -> None:
        self.last_action_node = None
        self.last_state_node = None


##################
#   PARAMS
##################

@dataclass(frozen=True)
class MCTSParams:
    safe_choice: bool = True
    utc_const: float = 1.4
    prune_threshold: int = 10


##################
#   MCTS
##################

class MCTS:
    def __init__(self, params: MCTSParams = MCTSParams()) -> None:
        self._params = params
        self._state_map: Dict[bytes, StateNode] = {}
        self._memory = LastMemory()

    @property
    def memory(self):
        return self._memory
 

    # ---------- helpers ----------

    def _get_or_create_state(self, state: State) -> StateNode:
        h = get_statehash(state)
        node = self._state_map.get(h)
        if node is None:
            node = StateNode(state)
            self._state_map[h] = node
        return node

    def _expand_state(self, state_node: StateNode) -> None:
        if state_node.expanded:
            return
        for action in emptycoords_from_table(state_node.state):
            state_node.connect(ActionNode(action))
        state_node.expanded = True

    # ---------- core ----------

    def backpropagate(self, start: Node, value: float) -> None:
        stack = [start]
        visited: Set[str] = set()

        while stack:
            node = stack.pop()
            if node.id in visited:
                continue
            visited.add(node.id)

            node.value += value
            for parent in node.get_neighbors(CONN_TYPE.BACKWARD):
                stack.append(parent)

    def choose_best_actionnode(self, state_node: StateNode) -> ActionNode:
        return max(
            state_node.get_neighbors(),
            key=lambda a: a.utc_value(
                state_node.visited_time, self._params.utc_const
            ),
        )

    # ---------- public API ----------

    def play(self, state: State) -> Action:
        if self._memory.first_root is None:
            self._memory.first_root = state

        state_node = self._get_or_create_state(state)
        self._expand_state(state_node)

        state_node.visited_time += 1

        action_node = (
            random.choice(list(state_node.get_neighbors()))
            if self._params.safe_choice and state_node._forward
            else self.choose_best_actionnode(state_node)
        )

        action_node.visited_time += 1

        self._memory.last_state_node = state_node
        self._memory.last_action_node = action_node

        return action_node.action

    def feed_reward(self, reward: float, next_state: Optional[State]) -> None:
        if self._memory.last_action_node is None:
            raise ValueError("play() must be called before feed_reward()")

        # ----- register transition -----
        if next_state is not None:
            next_node = self._get_or_create_state(next_state)

            if next_node not in self._memory.last_action_node.get_neighbors():
                self._memory.last_action_node.connect(next_node)

            next_node.visited_time += 1
            self._expand_state(next_node)

            self._memory.last_state_node = next_node

        # ----- backprop reward -----
        self.backpropagate(self._memory.last_state_node, reward)  # type: ignore

    def prune(self) -> int:
        to_delete = [
            h for h, n in self._state_map.items()
            if n.visited_time < self._params.prune_threshold
        ]
        for h in to_delete:
            del self._state_map[h]
        return len(to_delete)

    def bfs_traversal(self, start_state: State):
        start = self._state_map.get(get_statehash(start_state))
        if start is None:
            return

        queue = deque([start])
        visited: Set[str] = set()

        while queue:
            node = queue.popleft()
            if node.id in visited:
                continue
            visited.add(node.id)

            yield node
            for neigh in node.get_neighbors():
                queue.append(neigh)  # type: ignore

    # ---------- serialization ----------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MCTS":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, MCTS):
            raise TypeError("Invalid MCTS file")
        obj._memory.reset()
        return obj