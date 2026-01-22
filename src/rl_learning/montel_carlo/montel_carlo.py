import json
import random
from collections import deque
from typing import Deque, Dict, Iterator, List, Optional, Tuple

from numpy.typing import NDArray

from src.common import Action, emptycoords_from_table
from src.game.controller import BaseController
from src.game.client import Client
from src.game.interface import BaseInterface
from src.game.table import MarkType
from src.game.client import Client

class _ReplayMemory:
    def __init__(self, capacity: Optional[int] = None) -> None:
        self.memory: Deque[str] = deque(maxlen=capacity)

    def __iter__(self) -> Iterator[str]:
        return iter(self.memory)

    def __reversed__(self) -> Iterator[str]:
        return reversed(self.memory)

    def push(self, obj: str) -> None:
        self.memory.append(obj)

    def sample(self, batch_size: int) -> List[str]:
        return random.sample(self.memory, batch_size)

    def reset(self) -> None:
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)


class MontelCarloController(BaseController[Tuple[NDArray, List[Action]], Action]):
    def __init__(self, *,
                 lr: float, decay_gamma: float, exp_rate: float = 0.3, file_name: str = "mnc_01") -> None:
        
        super().__init__()
        self.mark_type = None
        self.exp = exp_rate
        self.lr = lr
        self.decay_gamma = decay_gamma
        self.state_value: Dict[str, float] = {}
        self.memory = _ReplayMemory()
        self.file_name = file_name

    @staticmethod
    def get_hash(state: NDArray) -> str:
        return str(state.flatten())

    def pre_processing(self, input_state: NDArray) -> Tuple[NDArray, List[Action]]:
        return input_state, emptycoords_from_table(input_state)

    def model_call(self, model_input: Tuple[NDArray, List[Action]]) -> Action:
        state, positions = model_input

        if not positions:
            raise ValueError("Empty possible move")

        if random.random() <= self.exp:
            return random.choice(positions)

        value_max: Optional[float] = None
        action: Action

        for pos in positions:
            next_state = state.copy()
            next_state[pos[0], pos[1]] = self.mark_type
            key = self.get_hash(next_state)
            value = self.state_value.get(key, 0.0)

            if value_max is None or value >= value_max:
                value_max = value
                action = pos

        return action # type: ignore

    #############
    # Addicitional
    #############

    def feed_reward(self, reward: float) -> None:
        for state_key in reversed(self.memory):
            current = self.state_value.get(state_key, 0.0)
            updated = current + self.lr * (self.decay_gamma * reward - current)
            self.state_value[state_key] = updated
            reward = updated

    def reset_memory(self) -> None:
        self.memory.reset()

    def save_policy(self) -> None:
        with open(f"{self.file_name}.json", "w", encoding="utf-8") as f:
            json.dump(self.state_value, f)

    def load_policy(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, float] = json.load(f)
        self.state_value = data


class MontelCarloClient(Client):
    def __init__(self, name: str, mark_type: MarkType, interface: BaseInterface, controller: MontelCarloController, *, max_trial: int = 5) -> None:
        super().__init__(name, mark_type, interface, controller, max_trial=max_trial)
        self._controller = controller

    @property
    def controller(self) -> MontelCarloController:
        return self._controller

    @classmethod
    def build_from_client(cls, client:Client):
        if not isinstance(client.controller, MontelCarloController):
            raise ValueError("Build that thing plese")
        client.controller.mark_type = client.mark_type
        return cls(client.name, client.mark_type, client.interface, client.controller) # type: ignore
    
    
