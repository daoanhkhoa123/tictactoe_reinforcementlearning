# General

Tic tac toe game default at 7x7, winning at 4 and no available configuration, mainly for reinforcement learning typa thing

# How to run the game

From `/src/runtimes/*.ipynb`, you can find:

```python
from src.common import RandomController, HumanController, CMDInterface, EmptyInterface, build_2clients, ClientArgs
from src.game.game import Game

client1, client2 = build_2clients(
    blu=ClientArgs(
        name="human",
        controller=HumanController(),
        interface=CMDInterface(),
    ),
    red=ClientArgs(
        name="bot",
        controller=RandomController(),
        interface=CMDInterface(),
    ),
)

game = Game(client1, client2)
winner = game.hajime()
game.owari(winner)
```

# Big folders

- /src/: for all the code
- /runtimes/: ipynbs or anything for runtimes code, mainly for usage on cloud like kaggle or collab
- /weights/: all model weights (optionals)

# Game

Available at /src/game

## /src/game/table.py

`Table` defines the core game state and rule enforcement for the Tic-Tac-Toe variant.  
It owns the board, validates moves, and determines the game outcome.

The table is fixed at **7×7** with a **win condition of 4 in a row**.

## /src/game/client.py

`Client` abstraction used by the Tic-Tac-Toe game engine.  
A `Client` represents a single player and acts as the coordination layer between:

- decision logic (`BaseController`)
- presentation logic (`BaseInterface`)
- game state and rules (`Table`)

The class is intentionally lightweight and stateless with respect to game rules, making it suitable for reinforcement learning, scripted agents, and human-controlled players.

The `Client` does **not** implement any game rules.  
Its sole responsibility is to execute a player turn by:

- exposing the current board state,
- requesting an action from a controller,
- applying the action through the table.

## /src/game/controller.py

`BaseController` defines a strict, three-stage decision pipeline for all controllers.  
It separates state transformation, model inference, and action decoding while enforcing a fixed execution order.

Decision pipeline:

- input_state (NDArray)
- pre_processing
- model_call
- post_processing
- coordinates of marking on table (row, col)

Subclasses may override:

- `pre_processing`: transform board state
- `model_call`: core decision logic (required)
- `post_processing`: decode model output

Move legality is handled by `Table`, not the controller.

## /src/game/interface.py
`BaseInterface` defines the presentation contract between the game engine and a player.  
It is responsible solely for **displaying the game state**, not for decision-making or rule enforcement.

Its purpose:

- Decouple visualization/output from game logic
- Support multiple frontends (CLI, GUI, silent/headless)
- Remain compatible with automated and RL-based agents

## /src/game/game.py

`Game` defines the high-level game loop and orchestration logic.  
It coordinates two clients, manages the game lifecycle, and determines the final outcome.

# common.py

## Overview

This module provides **runtime utilities and concrete implementations** used for quick experimentation, demos, and interactive play.  
It includes helper functions, example controllers, command-line interfaces, and client builders.

The code is intentionally pragmatic and optimized for iteration speed rather than strict abstraction purity.

## Quick Macros

Utility helpers for index/coordinate conversion and board inspection.

- `coords_to_index`: convert `(row, col)` to flat index
- `index_to_coords`: convert flat index back to `(row, col)`
- `emptycoords_from_table`: list empty board coordinates
- `emptyindex_from_table`: list empty board indices

These helpers are commonly used by simple policies and RL controllers.

## Controllers

### `RandomController`

A minimal stochastic controller.

- Selects randomly from available empty cells
- Uses flat index space for simplicity
- Optional sleep to simulate thinking time

Pipeline usage:
- `pre_processing`: extract valid action space
- `model_call`: random sampling
- `post_processing`: index → coordinate mapping

### `HumanController`

Interactive controller for command-line play.

- Accepts `(row, col)` input from stdin
- Performs basic format and boundary validation
- Retries input up to a fixed number of attempts

No preprocessing or postprocessing override is required.

## Interfaces

### `CMDInterface`

Command-line interface for rendering the board.

- Renders row/column indices
- Uses compact textual marks (`R`, `B`, `.`)
- Displays active player information

Designed for human-readable debugging and demos.

### `EmptyInterface`

Silent / minimal interface.

- Produces no board output
- Suitable for bot vs bot or RL rollouts

## Client Builder Utilities

### `ClientArgs`

Lightweight container for client configuration.

Encapsulates:
- name
- controller
- interface

### `build_2clients`

Convenience factory for constructing two clients.

- Assigns `BLU` and `RED` marks deterministically
- Reduces boilerplate during experiments and demos

## Intended Usage

This module is meant for:
- rapid prototyping
- interactive testing
- reinforcement learning experiments
- example implementations

It is **not** part of the core game engine and may evolve freely.

# Reinforcement Learning

## Overview
- Implementations are located in `src\rl_learning`, with each folder corresponding to a specific algorithm.

- Check `README.md` under each alogrithm for more detail
