"""Actor-critic self-play utilities and controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.common import Action, emptycoords_from_table
from src.game.client import Client
from src.game.controller import BaseController
from src.game.interface import BaseInterface
from src.game.table import MarkType, Table



@dataclass(frozen=True)
class _ActionChoiceInfo:
    action: Action
    features: NDArray[np.float32]
    legal_indices: Sequence[int]
    local_action_idx: int
    logits: NDArray[np.float32]
    value: float
    probs: NDArray[np.float32]
    mark: MarkType


@dataclass(frozen=True)
class TransitionRecord:
    choice: _ActionChoiceInfo
    next_state: NDArray[np.int8]
    done: bool


class ActorCriticPolicy:
    def __init__(
        self,
        *,
        n_rows: int = 7,
        n_cols: int = 7,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_rows * n_cols
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        scale = 0.01
        self.policy_weights: NDArray[np.float32] = (
            self.rng.normal(scale=scale, size=(self.n_actions, self.n_actions))
            .astype(np.float32)
        )
        self.policy_bias: NDArray[np.float32] = np.zeros(self.n_actions, dtype=np.float32)

        self.value_weights: NDArray[np.float32] = (
            self.rng.normal(scale=scale, size=(self.n_actions,)).astype(np.float32)
        )
        self.value_bias: float = 0.0

    def _encode(self, state: NDArray, mark: MarkType) -> NDArray[np.float32]:
        approx = (state * mark).astype(np.float32)
        return approx.flatten()

    def value(self, features: NDArray[np.float32]) -> float:
        return float(np.dot(self.value_weights, features) + self.value_bias)

    @staticmethod
    def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
        shift = logits - np.max(logits)
        exp = np.exp(shift)
        return exp / np.sum(exp)

    def select_action(
        self,
        state: NDArray,
        mark: MarkType,
        legal_actions: Sequence[Action],
        deterministic: bool = False,
    ) -> _ActionChoiceInfo:
        if not legal_actions:
            raise ValueError("No legal actions available")

        features = self._encode(state, mark)
        logits = self.policy_weights @ features + self.policy_bias
        value = self.value(features)

        legal_indices = tuple(y * self.n_cols + x for y, x in legal_actions)
        legal_logits = logits[list(legal_indices)].astype(np.float32)
        probs = self._softmax(legal_logits)

        if deterministic:
            chosen_local = int(np.argmax(legal_logits))
        else:
            chosen_local = int(self.rng.choice(len(legal_indices), p=probs))

        return _ActionChoiceInfo(
            action=legal_actions[chosen_local],
            features=features,
            legal_indices=legal_indices,
            local_action_idx=chosen_local,
            logits=legal_logits,
            value=value,
            probs=probs,
            mark=mark,
        )

    def learn(self, record: TransitionRecord, reward: float) -> None:
        choice = record.choice
        next_features = (
            self._encode(record.next_state, choice.mark)
            if not record.done
            else np.zeros_like(choice.features)
        )

        next_value = 0.0 if record.done else self.value(next_features)
        advantage = reward + self.gamma * next_value - choice.value

        grads = -choice.probs
        grads[choice.local_action_idx] += 1.0
        grads *= advantage

        for local_idx, global_idx in enumerate(choice.legal_indices):
            scalar = grads[local_idx]
            self.policy_weights[global_idx] += (
                self.actor_lr * scalar * choice.features
            )
            self.policy_bias[global_idx] += self.actor_lr * scalar

        critic_delta = self.critic_lr * advantage
        self.value_weights += critic_delta * choice.features
        self.value_bias += critic_delta

    def save(self, path: Path | str) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            policy_weights=self.policy_weights,
            policy_bias=self.policy_bias,
            value_weights=self.value_weights,
            value_bias=self.value_bias,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            gamma=self.gamma,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ActorCriticPolicy":
        path = Path(path)
        with np.load(path) as data:
            obj = cls(
                n_rows=int(data["n_rows"]),
                n_cols=int(data["n_cols"]),
                actor_lr=float(data["actor_lr"]),
                critic_lr=float(data["critic_lr"]),
                gamma=float(data["gamma"]),
            )
            obj.policy_weights = data["policy_weights"].astype(np.float32)
            obj.policy_bias = data["policy_bias"].astype(np.float32)
            obj.value_weights = data["value_weights"].astype(np.float32)
            obj.value_bias = float(data["value_bias"])
        return obj


class ActorCriticController(BaseController[NDArray, Action]):
    def __init__(
        self,
        policy: ActorCriticPolicy,
        mark: MarkType,
        *,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self._policy = policy
        self._mark = mark
        self._deterministic = deterministic

    def model_call(self, model_input: NDArray) -> Action:
        legal_actions = emptycoords_from_table(model_input)
        choice = self._policy.select_action(
            model_input, self._mark, legal_actions, deterministic=self._deterministic
        )
        return choice.action


class ActorCriticTrainer:
    def __init__(
        self,
        policy: ActorCriticPolicy,
        *,
        episodes: int = 5000,
        save_every: int = 100,
        progress_every: int = 50,
        baseline_rating: int = 1500,
        weights_dir: Path | str = Path("weights"),
    ) -> None:
        self.policy = policy
        self.episodes = episodes
        self.save_every = save_every
        self.progress_every = progress_every
        self.baseline_rating = baseline_rating
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.stats: Dict[str, int] = {"blu": 0, "red": 0, "draw": 0}

    def _run_episode(self) -> Tuple[MarkType, Dict[MarkType, List[TransitionRecord]]]:
        table = Table()
        history: Dict[MarkType, List[TransitionRecord]] = {
            MarkType.BLU: [],
            MarkType.RED: [],
        }

        current = MarkType.BLU
        while True:
            state = table.get_table(False).copy()
            legal_actions = emptycoords_from_table(state)
            if not legal_actions:
                break

            choice = self.policy.select_action(state, current, legal_actions)
            placed = table.mark(choice.action[0], choice.action[1], current)
            if not placed:
                raise RuntimeError("Policy produced illegal move")

            next_state = table.get_table(False).copy()
            done = table.is_full() or table.get_winner() != MarkType.EMPTY
            record = TransitionRecord(choice=choice, next_state=next_state, done=done)
            history[current].append(record)

            if done:
                break

            current = (
                MarkType.RED if current == MarkType.BLU else MarkType.BLU
            )

        return table.get_winner(), history

    def _reward_map(self, winner: MarkType) -> Dict[MarkType, float]:
        if winner == MarkType.BLU:
            return {MarkType.BLU: 1.0, MarkType.RED: -1.0}
        if winner == MarkType.RED:
            return {MarkType.BLU: -1.0, MarkType.RED: 1.0}
        return {MarkType.BLU: 0.0, MarkType.RED: 0.0}

    def _apply_rewards(
        self,
        history: Dict[MarkType, List[TransitionRecord]],
        winner: MarkType,
    ) -> None:
        rewards = self._reward_map(winner)
        for mark, reward in rewards.items():
            for record in history[mark]:
                self.policy.learn(record, reward)

    def _save_checkpoint(self, episode: int) -> None:
        path = self.weights_dir / f"actor_critic_ep{episode:04d}.npz"
        self.policy.save(path)
        latest = self.weights_dir / "actor_critic_latest.npz"
        self.policy.save(latest)

    def _log_progress(self, episode: int) -> None:
        blu = self.stats["blu"]
        red = self.stats["red"]
        draw = self.stats["draw"]
        print(
            f"[Episode {episode}/{self.episodes}] "
            f"BLU wins: {blu}, RED wins: {red}, draws: {draw}. "
            f"Baseline target ~{self.baseline_rating} elo vs Stockfish."
        )

    def train(self) -> None:
        for episode in range(1, self.episodes + 1):
            winner, history = self._run_episode()
            if winner == MarkType.BLU:
                self.stats["blu"] += 1
            elif winner == MarkType.RED:
                self.stats["red"] += 1
            else:
                self.stats["draw"] += 1

            self._apply_rewards(history, winner)

            if episode % self.progress_every == 0:
                self._log_progress(episode)

            if episode % self.save_every == 0:
                self._save_checkpoint(episode)

        if self.episodes % self.save_every != 0:
            self._save_checkpoint(self.episodes)

        print("Training completed.")

    def best_client(
        self,
        *,
        interface: BaseInterface,
        mark: MarkType = MarkType.RED,
        deterministic: bool = True,
        name: str = "actor-critic",
    ) -> Client:
        controller = ActorCriticController(
            policy=self.policy, mark=mark, deterministic=deterministic
        )
        return Client(name=name, mark_type=mark, interface=interface, controller=controller)


def build_client_from_weights(
    path: Path | str,
    *,
    interface: BaseInterface,
    mark: MarkType = MarkType.RED,
    deterministic: bool = True,
    name: str = "actor-critic",
) -> Client:
    policy = ActorCriticPolicy.load(path)
    controller = ActorCriticController(
        policy=policy, mark=mark, deterministic=deterministic
    )
    return Client(name=name, mark_type=mark, interface=interface, controller=controller)


def run_training() -> None:
    policy = ActorCriticPolicy()
    trainer = ActorCriticTrainer(policy)
    trainer.train()


if __name__ == "__main__":
    run_training()
