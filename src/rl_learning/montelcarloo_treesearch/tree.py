from collections import deque
from typing import Deque, Iterator, Generic, TypeVar, Optional
from enum import IntEnum, auto
from typing_extensions import Self
from src.common import Action, emptycoords_from_table
from numpy.typing import NDArray

class CONN_TYPE(IntEnum):
    FORWARD = auto()
    BACKWARD = auto()


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


T = TypeVar("T", bound="Node")


class Node(Generic[T]):
    def __init__(self) -> None:
        self.value: float = 0.0
        self._forward: Deque[Edge[Self, T]] = deque()
        self._backward: Deque[Edge[T, Self]] = deque()

    def connect(self, node: T) -> None:
        e: Edge[Self, T] = Edge(self, node)
        self._forward.append(e)
        node._backward.append(e)

    def get_neighbors(self, con_type: CONN_TYPE = CONN_TYPE.FORWARD) -> Iterator[T]:
        if con_type is CONN_TYPE.FORWARD:
            for e in self._forward:
                yield e.dst
        else:
            for e in self._backward:
                yield e.src

class StateHashT(str):  ...
class State(NDArray):
    def get_hash(self) -> StateHashT:
        return StateHashT(self.flatten())
    

class StateNode(Node["ActionNode"]):
    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state


class ActionNode(Node[StateNode]):
    def __init__(self, action: Action) -> None:
        super().__init__()
        self.action = action


getall_possible_actions = emptycoords_from_table

class MSTC:
    def __init__(self) -> None:
        self._state_map: dict[StateHashT, StateNode] = {}
        self._last_action_node: Optional[ActionNode] = None
        self._last_state_node: Optional[StateNode] = None

    @property
    def state_map(self):
        return self._state_map
    
    @property
    def last_action_node(self):
        return self._last_action_node

    @property
    def last_state_node(self):
        return self._last_state_node

    #############
    #   BACKEDN
    ############

    def backprogate_add(self, node: Node, value: float) -> None:
        for prev in node.get_neighbors(CONN_TYPE.BACKWARD):
            prev.value += value
            self.backprogate_add(prev, value)



    ################## 
    #    USAGE
    ##################

    def execute_action(self, action_node: ActionNode) -> Action:
        self._last_action_node = action_node
        return action_node.action

    def choose_best_actionode(self, state: State) -> ActionNode:
        state_node = self.state_map[state.get_hash()]
        best = max(state_node.get_neighbors(), key=lambda a: a.value)
        return best

    def update_state(self, state_node: StateNode) -> None:
        if self.last_action_node is None:
            raise ValueError
        action = self.last_action_node
        action.connect(state_node)
        action.value += state_node.value
        self.backprogate_add(action, state_node.value)

    def init_state(self, state: State) -> StateNode:
        state_node = StateNode(state)
        self.state_map[state.get_hash()] = state_node
        for action in getall_possible_actions(state):
            state_node.connect(ActionNode(action))

        return state_node


    #######################
    #
    #####################      
    def play(self, state:State) -> Action:
        # get state node, or create it
        if state not in self.state_map:
            # create new state node
            state_node = self.init_state(state)
        else:
            state_node = self.state_map[state.get_hash()]

        # sovling consequences of the last action
        if self.last_action_node is not None:
            # that last action leads to this state, so connect them
            self.update_state(state_node)

        # update last action and state then return them
        self._last_action_node = self.choose_best_actionode(state)
        self._last_state_node = state_node
        return self._last_action_node.action

    def feed_reward(self, reward: float) -> None:
        if self.last_state_node is None:
            raise ValueError(" Run the model first ")
        
        self.last_state_node.value += reward
        self.backprogate_add(self.last_state_node, reward)