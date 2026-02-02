# Monte Carlo Control (State-Value–Based)

## Core Idea
- This implementation follows Monte Carlo Control with an epsilon-greedy policy and a tabular state-value function.
- The agent learns only after complete episodes by propagating the final reward backward through all visited states.
- No environment model or transition dynamics are required.
- Learning is purely experience-based and relies on self-generated trajectories.

---

## State Representation
- Each game state is represented by the full board configuration (NDArray).
- To enable tabular storage, the state is converted into a hashable key:
``` python
def get_hash(state: NDArray) -> str:
    return str(state.flatten())
```

- This key indexes the value table:

``` python
state_value: Dict[str, float] = {}

def feed_reward(self, reward: float) -> None:
    ...
    updated = current + self.lr * (self.decay_gamma * reward - current)
    self.state_value[state_key] = updated

```

---

## Action Space
- Legal actions correspond to empty board coordinates obtained from:
  `emptycoords_from_table(state)`
- An action is a tuple (row, column) where the agent places its `mark_type`.

---

## Policy (epsilon-Greedy)
Action selection is implemented in `model_call`:

1. With probability `epsilon = exp_rate`, select a random legal action (exploration).
2. Otherwise, select the action that maximizes the estimated value of the next state. The flow is like this:
``` python
if random.random() <= self.exp:
    return random.choice(positions)

def value_of_sprime(pos: Action) -> float:
    """ Try the action and get next state value """
    next_state = state.copy()
    next_state[pos[0], pos[1]] = self.mark_type
    
    key = self.get_hash(next_state)
    return self.state_value.get(key, 0.0)

return max(positions, key=value_of_sprime)
```

Value Evaluation
$V(s') = \texttt{self.statevalue.get}(\texttt{hash}(s'), 0.0)$

Policy Definition
$\pi(s) = \begin{cases}
\text{random action}, & \text{with probability } \varepsilon \\\
\arg\max_a V(s'), & \text{otherwise}
\end{cases}$

This is an on-policy Monte Carlo control strategy.

---

## Episode Memory
- During an episode, every visited state hash is appended to memory:
  `self.controller.memory.push(state_hash)`
- Memory preserves temporal order using a deque.
- Learning uses complete episodes only; no sampling or replay is applied.

---

## Learning Rule (Backward Return Propagation)

After the episode terminates, `feed_reward` is called with the terminal reward.

Updates proceed backward through the episode:

$V(s) \leftarrow V(s) + \alpha \big[ \gamma \cdot \text{reward} - V(s) \big]$  
$\text{reward} \leftarrow V(s)$

Where:
- $\alpha$ is the learning rate (`self.lr`)
- $\gamma$ is the discount factor (`self.decay_gamma`)
- $\text{reward}$ is the propagated return

This corresponds to an **incremental Monte Carlo return update** without explicit return storage.

```python
def feed_reward(self, reward: float) -> None:
    for state_key in reversed(self.memory):
        current = self.state_value.get(state_key, 0.0)
        updated = current + self.lr * (self.decay_gamma * reward - current)
        self.state_value[state_key] = updated
        reward = updated
```

## Debug Note
- `train: bool`
Currently, this flag is used only for debugging/logging purpose

---

## Policy Persistence
- Learned values can be saved to disk using `save_policy` (JSON format).
- Policies can be restored using `load_policy` for evaluation or continued training.

---

## Initialization

Initialize client with any controller then use `build_montelclient(client1, params)` to insert the montel carlo controller

```python
from src.game.interface import BaseInterface
from src.game.controller import BaseController
from src.rl_learning.montel_carlo.montel_carlo import MontelHyperParams, build_montelclient

client1, client2 = build_2clients(
    blu = ClientArgs(
        name="montel-carlo",
        controller=BaseController(), # any
        interface=BaseInterface(),
    ),
    red = ...
)

params = MontelHyperParams(
    lr=0.05,
    decay_gamma=0.95,
    exp_rate=0.2,
)

client1 =  build_montelclient(client1, params)
```

## Training

Training is performed through self-play within a fixed environment instance:

- An Environment is created with two clients (client1 and client2).
- client1 is the learning agent using Monte Carlo control.
- client2 acts as an opponent and could shares the same value function during training.

Value sharing can be enforced each episode:
```python
client2.controller.state_value = client1.controller.state_value
```

This makes both agents evaluate states using a single, shared state-value table.

Training runs for a fixed number of episodes:
```python
  for _ in range(1024 * 5):
```

For each episode:

- Environment reset  

```python
client1.reset_memory()
client2.reset_memory()
env.reset_table()
```
For Monte Carlo updates, the memory cache must be reset at the end of every episode. This ensures that each run starts with a clean trajectory history, preventing old state–action traces from interfering with new episodes.

- Gameplay  
```python
winner = env.hajime()  
```
The environment runs until a terminal state is reached and returns the winner’s mark.

- Episode termination  
```python
env.owari(winner)  
```
Finalizes the episode and triggers any end-of-game logic.

- Reward semantics:
```python
client1.controller.feed_reward(c1_reward)
```

Win: reward = 1; 
Loss: reward = -1;
Draw: reward = 0

Only terminal rewards are used. No intermediate rewards are provided during gameplay.

Then, you can load and save policy
```python
client1.controller.load_policy("loadpath_here")
client2.controller.save_policy("savepath_here")
```
---

## Algorithm Characteristics
- Model-free
- On-policy
- Episodic learning
- Tabular state-value approximation
- No bootstrapping
- epsilon-greedy exploration

This approach is suitable for small, finite environments where full episodes are inexpensive to generate and the state space is manageable.
