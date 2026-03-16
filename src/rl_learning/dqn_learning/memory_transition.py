from collections import deque
import random
from dataclasses import dataclass

@dataclass
class Transition:
    state: object
    acttion: object
    next_state: object
    reward: object

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
