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
    parser.add_argument("--episodes", type=int, default=5000, help="Number of self-play episodes.")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint frequency.")
    parser.add_argument("--progress-every", type=int, default=50, help="Progress log frequency.")
    parser.add_argument("--baseline", type=int, default=2000, help="Reference Elo baseline for logs.")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for TD updates.")
    parser.add_argument("--greedy-after", type=int, default=200000, help="Episode after which training acts greedily.")
    parser.add_argument("--force-winning-until", type=int, default=700000, help="Teacher-force immediate winning moves until this episode.")
    parser.add_argument("--force-blocking-from", type=int, default=700001, help="Teacher-force immediate blocking moves starting from this episode.")
    parser.add_argument("--entropy-start-coef", type=float, default=0.1, help="Initial entropy coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Final entropy coefficient after decay.")
    parser.add_argument("--entropy-decay-episodes", type=int, default=500000, help="Episodes used for linear entropy decay.")

    parser.add_argument("--opponent-prob", type=float, default=0.2, help="Probability of playing against a saved self-play opponent.")
    parser.add_argument("--opponent-interval", type=int, default=1000, help="Episodes between saving an opponent snapshot.")
    parser.add_argument("--opponent-snapshot-win-rate", type=float, default=0.65, help="Recent win-rate threshold that triggers a new opponent snapshot.")
    parser.add_argument("--opponent-snapshot-window", type=int, default=200, help="Window size used to measure recent training win rate.")
    parser.add_argument("--opponent-snapshot-cooldown", type=int, default=250, help="Minimum episode gap between adaptive opponent snapshots.")
    parser.add_argument("--opponent-pool-size", type=int, default=5, help="Number of saved opponents kept in memory.")
    parser.add_argument("--random-opponent-prob", type=float, default=0.2, help="Chance the opponent uses the heuristic win-block bot instead of policy play.")
    parser.add_argument("--heuristic-random-move-prob", type=float, default=0.1, help="Chance the heuristic bot ignores tactics and plays a random move.")
    parser.add_argument("--step-penalty", type=float, default=0.1, help="Penalty applied after every training move to encourage faster wins.")
    parser.add_argument("--missed-block-penalty", type=float, default=30.0, help="Extra penalty when an immediate opponent threat is left unblocked.")
    parser.add_argument("--missed-win-penalty", type=float, default=50.0, help="Extra penalty when an immediate winning move is missed.")

    parser.add_argument("--opening-min-moves", type=int, default=2, help="Minimum number of random opening moves.")
    parser.add_argument("--opening-max-moves", type=int, default=4, help="Maximum number of random opening moves.")

    parser.add_argument("--weights-dir", type=Path, default=Path("weights"), help="Directory to write checkpoints.")
    parser.add_argument("--block-bonus", type=float, default=2.0, help="Reward bonus for reducing opponent threats.")
    parser.add_argument("--block-threat-bonus", type=float, default=2.5, help="Reward for blocking an immediate opponent win.")
    parser.add_argument("--block-threat-penalty", type=float, default=-3.0, help="Penalty when an immediate opponent win is left open.")
    parser.add_argument("--win-move-bonus", type=float, default=20.0, help="Bonus when a move immediately wins.")
    parser.add_argument("--create-threat-bonus", type=float, default=0.2, help="Bonus per new threat created.")
    parser.add_argument("--two-way-bonus", type=float, default=0.5, help="Bonus when two or more threats appear at once.")
    parser.add_argument("--survival-bonus", type=float, default=0.0, help="Bonus for surviving one more turn.")
    parser.add_argument("--risk-penalty", type=float, default=-0.25, help="Penalty when the move increases opponent threats.")
    parser.add_argument("--win-reward", type=float, default=1.0, help="Terminal reward for wins.")
    parser.add_argument("--lose-reward", type=float, default=-10.0, help="Terminal reward for losses.")
    parser.add_argument("--draw-reward", type=float, default=1.0, help="Terminal reward for draws.")
    parser.add_argument("--offense-bonus", type=float, default=0.1, help="Bonus for increasing your own threats.")

    parser.add_argument("--resume", type=Path, help="Checkpoint (.pt) to resume from.")
    parser.add_argument("--start-episode", type=int, default=None, help="Episode number already completed before this run.")
    parser.add_argument("--start-blu", type=int, default=0, help="Initial BLU win count for progress logs.")
    parser.add_argument("--start-red", type=int, default=0, help="Initial RED win count for progress logs.")
    parser.add_argument("--start-draw", type=int, default=0, help="Initial draw count for progress logs.")
    parser.add_argument("--use-gpu", action="store_true", help="Log a reminder to run inside a GPU-capable environment.")

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
        entropy_start_coef=args.entropy_start_coef,
        entropy_decay_episodes=args.entropy_decay_episodes,
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
        gamma=args.gamma,
        greedy_after=args.greedy_after,
        force_winning_until=args.force_winning_until,
        force_blocking_from=args.force_blocking_from,
        opponent_prob=args.opponent_prob,
        opponent_interval=args.opponent_interval,
        opponent_pool_size=args.opponent_pool_size,
        random_opponent_prob=args.random_opponent_prob,
        heuristic_random_move_prob=args.heuristic_random_move_prob,
        step_penalty=args.step_penalty,
        missed_block_penalty=args.missed_block_penalty,
        missed_win_penalty=args.missed_win_penalty,
        survival_bonus=args.survival_bonus,
        risk_penalty=args.risk_penalty,
        opening_min_moves=args.opening_min_moves,
        opening_max_moves=args.opening_max_moves,
        opponent_snapshot_win_rate=args.opponent_snapshot_win_rate,
        opponent_snapshot_window=args.opponent_snapshot_window,
        opponent_snapshot_cooldown=args.opponent_snapshot_cooldown,
    )

    if args.resume:
        episode_done = trainer.resume_from_checkpoint(args.resume)
        if trainer.episodes <= episode_done:
            trainer.episodes = episode_done + args.episodes
            print(
                f"Resume target interpreted as +{args.episodes} episodes; new target is {trainer.episodes}."
            )
        print(f"Resuming from checkpoint {args.resume.name} (episode {episode_done}).")

    if args.start_episode is not None:
        trainer._last_episode = args.start_episode
        trainer._start_episode = args.start_episode + 1
        if trainer.episodes <= args.start_episode:
            trainer.episodes = args.start_episode + args.episodes

    trainer.stats["blu"] = args.start_blu
    trainer.stats["red"] = args.start_red
    trainer.stats["draw"] = args.start_draw

    trainer.train()


if __name__ == "__main__":
    main()
