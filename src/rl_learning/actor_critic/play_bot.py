"""Simple entry point to play against the trained actor-critic bot with move history."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.common import CMDInterface, HumanController
from src.game.client import Client
from src.game.table import MarkType, Table
from src.rl_learning.actor_critic.actor_critic import (
    build_client_from_weights,
)
from src.rl_learning.actor_critic.deep_actor_critic import (
    build_deep_client,
)

HistoryEntry = Tuple[int, int, MarkType]


def _print_history(history: List[HistoryEntry]) -> None:
    if not history:
        return

    lines = ["Move history:"]
    for step, (y, x, mark) in enumerate(history, start=1):
        actor = "You" if mark == MarkType.BLU else "Actor-Critic"
        lines.append(f"{step:02d}. {actor} ({mark.name}) -> ({y}, {x})")
    print("\n" + "\n".join(lines))
    print()


def _announce_result(winner: MarkType, human: Client, bot: Client) -> None:
    if winner == human.mark_type:
        print("You won! Nice job.")
    elif winner == bot.mark_type:
        print("Actor-critic bot wins. Keep practicing!")
    else:
        print("Draw. Both players defended well.")


def _build_human_client(name: str) -> Client:
    return Client(
        name=name,
        mark_type=MarkType.BLU,
        interface=CMDInterface(),
        controller=HumanController(),
    )


def _run_match(human: Client, bot: Client) -> None:
    table = Table()
    players = {
        MarkType.BLU: human,
        MarkType.RED: bot,
    }
    sequence = [MarkType.BLU, MarkType.RED]
    history: List[HistoryEntry] = []
    turn = 0

    while True:
        current_mark = sequence[turn % 2]
        player = players[current_mark]
        move = player.play(table)
        history.append((move[0], move[1], current_mark))
        _print_history(history)

        winner = table.get_winner()
        if winner != MarkType.EMPTY or table.is_full():
            _announce_result(winner, human, bot)
            break

        turn += 1


def main() -> None:
    parser = ArgumentParser(
        description="Play against the trained actor-critic bot with history logging."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/actor_critic_latest.npz",
        help="Checkpoint file for the actor-critic policy.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Let the bot sample moves instead of greedy picks.",
    )
    parser.add_argument(
        "--human-name",
        type=str,
        default="You",
        help="Name that represents the human player in the log.",
    )

    args = parser.parse_args()

    human = _build_human_client(args.human_name)
    weights_path = Path(args.weights)
    if weights_path.suffix == ".pt":
        bot = build_deep_client(
            weights_path,
            interface=CMDInterface(),
            mark=MarkType.RED,
            deterministic=not args.stochastic,
            name="deep-actor-critic",
        )
    else:
        bot = build_client_from_weights(
            args.weights,
            interface=CMDInterface(),
            mark=MarkType.RED,
            deterministic=not args.stochastic,
            name="actor-critic",
        )

    _run_match(human, bot)


if __name__ == "__main__":
    main()
