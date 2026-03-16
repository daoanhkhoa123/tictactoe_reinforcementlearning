"""CLI to run long actor-critic self-play training."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.rl_learning.actor_critic.actor_critic import ActorCriticPolicy, ActorCriticTrainer


def main() -> None:
    parser = ArgumentParser(description="Train actor-critic self-play with checkpoints and logging.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Total number of self-play episodes to generate (default: 5000).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Checkpoint interval (default: 100 episodes).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="How often to log progress (default: 50 episodes).",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=2000,
        help="Baseline Elo estimate for reference logging (default: 2000).",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory where checkpoints will be stored (default: weights/).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Log a reminder that you can run this on a GPU-backed Python build.",
    )

    args = parser.parse_args()

    if args.use_gpu:
        print("GPU flag enabled: run this script inside a GPU-backed Python environment for faster training.")

    policy = ActorCriticPolicy()
    trainer = ActorCriticTrainer(
        policy=policy,
        episodes=args.episodes,
        save_every=args.save_every,
        progress_every=args.progress_every,
        baseline_rating=args.baseline,
        weights_dir=args.weights_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
