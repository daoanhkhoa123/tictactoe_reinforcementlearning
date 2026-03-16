"""Deep actor-critic training CLI."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.rl_learning.actor_critic.deep_actor_critic import DeepActorCriticTrainer


def main() -> None:
    parser = ArgumentParser(description="Train the CNN-based actor-critic via self-play.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of self-play episodes to generate (default: 5000).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Checkpoint frequency (default: 100).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Progress log frequency (default: 50).",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=2000,
        help="Reference Elo baseline logged with progress (default: 2000).",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory to write checkpoints.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0005,
        help="Entropy regularization coefficient (default: 0.0005).",
    )
    parser.add_argument(
        "--block-bonus",
        type=float,
        default=0.35,
        help="Reward bonus for reducing opponent threats (default: 0.35).",
    )
    parser.add_argument(
        "--block-threat-bonus",
        type=float,
        default=0.3,
        help="Immediate reward for blocking an opponent win (default: 0.3).",
    )
    parser.add_argument(
        "--block-threat-penalty",
        type=float,
        default=-1.0,
        help="Penalty when an imminent opponent win is not blocked (default: -1.0).",
    )
    parser.add_argument(
        "--win-move-bonus",
        type=float,
        default=3.0,
        help="Extra reward when a move closes the game (default: 3.0).",
    )
    parser.add_argument(
        "--create-threat-bonus",
        type=float,
        default=0.2,
        help="Bonus per new threat created (default: 0.2).",
    )
    parser.add_argument(
        "--two-way-bonus",
        type=float,
        default=0.5,
        help="Extra bonus when multiple threats appear in one move (default: 0.5).",
    )
    parser.add_argument(
        "--win-reward",
        type=float,
        default=5.0,
        help="Episode reward for wins (default: 5.0).",
    )
    parser.add_argument(
        "--lose-reward",
        type=float,
        default=-5.0,
        help="Episode penalty for losses (default: -5.0).",
    )
    parser.add_argument(
        "--draw-reward",
        type=float,
        default=1.0,
        help="Episode reward for draws (default: 1.0).",
    )
    parser.add_argument(
        "--offense-bonus",
        type=float,
        default=0.1,
        help="Reward bonus for creating your own threats (default: 0.1).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint (.pt) to resume from.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help="Episode number already completed before this run (for logging).",
    )
    parser.add_argument(
        "--start-blu",
        type=int,
        default=0,
        help="Initial BLU win count for progress logs.",
    )
    parser.add_argument(
        "--start-red",
        type=int,
        default=0,
        help="Initial RED win count for progress logs.",
    )
    parser.add_argument(
        "--start-draw",
        type=int,
        default=0,
        help="Initial draw count for progress logs.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Log a reminder to run inside a GPU-capable environment.",
    )

    args = parser.parse_args()
    if args.use_gpu:
        print("GPU hint: run this inside a CUDA-capable Python environment for faster convolutional training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trainer = DeepActorCriticTrainer(
        episodes=args.episodes,
        save_every=args.save_every,
        progress_every=args.progress_every,
        baseline_rating=args.baseline,
        device=device,
        weights_dir=args.weights_dir,
        entropy_coef=args.entropy_coef,
        block_bonus=args.block_bonus,
        offense_bonus=args.offense_bonus,
        win_move_bonus=args.win_move_bonus,
        block_threat_bonus=args.block_threat_bonus,
        block_fail_penalty=args.block_threat_penalty,
        create_threat_bonus=args.create_threat_bonus,
        two_way_threat_bonus=args.two_way_bonus,
        win_reward=args.win_reward,
        lose_reward=args.lose_reward,
        draw_reward=args.draw_reward,
    )

    if args.resume:
        episode_done = trainer.resume_from_checkpoint(args.resume)
        print(f"Resuming from checkpoint {args.resume.name} (episode {episode_done}).")

    if args.start_episode is not None:
        trainer._last_episode = args.start_episode
        trainer._start_episode = args.start_episode + 1

    trainer.stats["blu"] = args.start_blu
    trainer.stats["red"] = args.start_red
    trainer.stats["draw"] = args.start_draw

    trainer.train()


if __name__ == "__main__":
    main()
