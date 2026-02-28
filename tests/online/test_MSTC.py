from src.rl_learning.montelcarloo_treesearch.tree import MCTS, StateNode, ActionNode, MCTSParams
import numpy as np

# ==============================
# CLEAN, CONSISTENT TEST CODE
# ==============================

# ---- Environment logic (VERY IMPORTANT) ----

def apply_action(state: np.ndarray, action) -> np.ndarray:
    """Apply an action (i, j) to the board and return a new state."""
    i, j = action
    new_state = state.copy()
    new_state[i, j] = 1
    return new_state


# ---- Helper pretty-printers ----

def print_state(state, title="State"):
    print(f"\n{title}:")
    print(state)

def print_action(action):
    print(f"Chosen action: {action}")

def print_reward(reward):
    print(f"\nReward fed: {reward}")

def print_node(node):
    if isinstance(node, StateNode):
        print(
            f"StateNode | visits={node.visited_time:2d} | value={node.value:5.2f}\n"
            f"{node.state}"
        )
    else:
        print(
            f"  ActionNode {node.action} | visits={node.visited_time:2d} | value={node.value:5.2f}"
        )


# ---- Run MCTS ----

mcts = MCTS(MCTSParams(prune_threshold=1))

# ==============================
# EPISODE 1
# ==============================

print("========== EPISODE 1 ==========")

state = np.array([[0, 0],
                  [0, 0]])

print_state(state, "Initial state")
action = mcts.play(state)
print_action(action)

state = apply_action(state, action)
print_state(state, "Next state")

action = mcts.play(state)
print_action(action)

state = apply_action(state, action)
print_state(state, "Next state")

print_reward(1.0)
mcts.feed_reward(1.0, state)

# ==============================
# EPISODE 2 (same tree, reset memory)
# ==============================

print("\n========== EPISODE 2 ==========")
mcts.memory.reset()

state = np.array([[0, 0],
                  [0, 0]])

print_state(state, "Initial state")
action = mcts.play(state)
print_action(action)

state = apply_action(state, action)
print_state(state, "Next state")

action = mcts.play(state)
print_action(action)

state = apply_action(state, action)
print_state(state, "Next state")

print_reward(4.5)
mcts.feed_reward(4.5, state)


print("\n========== EPISODE 3 ==========")
mcts.memory.reset()

state = np.array([[0, 0],
                  [0, 0]])

print_state(state, "Initial state")
action = mcts.play(state)
print_action(action)

state = apply_action(state, action)
print_state(state, "Next state")

action = mcts.play(state)
print_action(action)

state = apply_action(state, action)
print_state(state, "Next state")

print_reward(4.5)
mcts.feed_reward(4.5, state)

# ==============================
# TREE INSPECTION
# ==============================

print("Pruned nodes:", mcts.prune())

print("\n========== TREE (BFS from initial state) ==========")

for node in mcts.bfs_traversal(mcts.memory.first_root):
    print_node(node)
    print("-" * 40)

