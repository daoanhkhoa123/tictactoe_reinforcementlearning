"""Deep convolutional actor-critic for the 7x7 board."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.common import Action, emptycoords_from_table
from src.game.client import Client
from src.game.controller import BaseController
from src.game.interface import BaseInterface
from src.game.table import MarkType, Table


def _action_to_coords(action_idx: int) -> Tuple[int, int]:
    return divmod(action_idx, 7)


def count_threats(board: np.ndarray, mark: MarkType) -> int:
    target = int(mark.value)
    threats = 0

    def window_score(cells: np.ndarray) -> bool:
        return int(np.count_nonzero(cells == target)) == 3 and int(np.count_nonzero(cells == 0)) == 1

    for y in range(7):
        for x in range(4):
            window = board[y, x : x + 4]
            if window_score(window):
                threats += 1

    for x in range(7):
        for y in range(4):
            window = board[y : y + 4, x]
            if window_score(window):
                threats += 1

    for y in range(4):
        for x in range(4):
            window = np.array([board[y + i, x + i] for i in range(4)])
            if window_score(window):
                threats += 1

    for y in range(3, 7):
        for x in range(4):
            window = np.array([board[y - i, x + i] for i in range(4)])
            if window_score(window):
                threats += 1

    return threats


def _winner_from_board(board: np.ndarray) -> MarkType:
    def check_line(values: list[int]) -> bool:
        return values[0] != 0 and all(val == values[0] for val in values)

    for y in range(7):
        for x in range(4):
            if check_line([board[y, x + i] for i in range(4)]):
                return MarkType(board[y, x])

    for x in range(7):
        for y in range(4):
            if check_line([board[y + i, x] for i in range(4)]):
                return MarkType(board[y, x])

    for y in range(4):
        for x in range(4):
            if check_line([board[y + i, x + i] for i in range(4)]):
                return MarkType(board[y, x])

    for y in range(4):
        for x in range(3, 7):
            if check_line([board[y + i, x - i] for i in range(4)]):
                return MarkType(board[y, x])

    return MarkType.EMPTY


def _can_win_next(board: np.ndarray, mark: MarkType) -> bool:
    for y, x in emptycoords_from_table(board):
        board[y, x] = int(mark)
        winner = _winner_from_board(board)
        board[y, x] = MarkType.EMPTY
        if winner == mark:
            return True
    return False


class DeepActorCriticNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.shared_dense = nn.Linear(128 * 7 * 7, 512)
        self.shared_norm = nn.LayerNorm(512)
        self.shared_dropout = nn.Dropout(0.25)
        self.actor = nn.Linear(512, 49)
        self.critic = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        shared = self.shared_dense(x)
        shared = self.shared_norm(shared)
        shared = F.relu(shared)
        shared = self.shared_dropout(shared)
        logits = self.actor(shared)
        value = self.critic(shared).squeeze(-1)
        return logits, value


def _state_tensor(board: torch.Tensor, mark: MarkType) -> torch.Tensor:
    mark_value = float(mark)
    player_plane = (board == mark_value).to(torch.float32)
    opponent_plane = (board == -mark_value).to(torch.float32)
    mark_plane = torch.full_like(board, mark_value)
    stacked = torch.stack([player_plane, opponent_plane, mark_plane])
    return stacked.unsqueeze(0)


def _mask_for_actions(actions: List[Action], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(49, dtype=torch.bool, device=device)
    for action in actions:
        mask[action[0] * 7 + action[1]] = True
    return mask


@dataclass
class DeepTransition:
    log_prob: torch.Tensor
    value: torch.Tensor
    mark: MarkType
    entropy: torch.Tensor
    bonus: float
    defense_bonus: float
    offense_bonus: float
    win_bonus: float
    block_reward: float
    block_penalty: float
    threat_bonus: float
    two_way_bonus: float


class DeepActorCriticTrainer:
    def __init__(
        self,
        *,
        device: torch.device | None = None,
        episodes: int = 5000,
        save_every: int = 100,
        progress_every: int = 50,
        baseline_rating: int = 2000,
        weights_dir: Path | str = Path("weights"),
        lr: float = 1e-3,
        entropy_coef: float = 0.0005,
        block_bonus: float = 0.35,
        offense_bonus: float = 0.1,
        win_move_bonus: float = 3.0,
        block_threat_bonus: float = 0.3,
        block_fail_penalty: float = -1.0,
        create_threat_bonus: float = 0.2,
        two_way_threat_bonus: float = 0.5,
        win_reward: float = 5.0,
        lose_reward: float = -5.0,
        draw_reward: float = 1.0,
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.episodes = episodes
        self.save_every = save_every
        self.progress_every = progress_every
        self.baseline_rating = baseline_rating
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.model = DeepActorCriticNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.stats = {"blu": 0, "red": 0, "draw": 0}
        self._last_episode = 0
        self._start_episode = 1
        self.entropy_coef = entropy_coef
        self.block_bonus = block_bonus
        self.offense_bonus = offense_bonus
        self.win_move_bonus = win_move_bonus
        self.block_threat_bonus = block_threat_bonus
        self.block_fail_penalty = block_fail_penalty
        self.create_threat_bonus = create_threat_bonus
        self.two_way_threat_bonus = two_way_threat_bonus
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self._bonus_total = 0.0
        self._entropy_total = 0.0
        self._chunk_counter = 0
        self._defense_bonus_total = 0.0
        self._offense_bonus_total = 0.0
        self._threat_bonus_total = 0.0
        self._two_way_bonus_total = 0.0
        self._block_reward_total = 0.0
        self._block_penalty_total = 0.0

    def _legal_mask(self, legal_actions: List[Action]) -> torch.Tensor:
        return _mask_for_actions(legal_actions, self.device)

    def _run_episode(self) -> Tuple[MarkType, List[DeepTransition]]:
        table = Table()
        transitions: List[DeepTransition] = []
        current_mark = MarkType.BLU

        while True:
            board = table.get_table(False).copy()
            legal_actions = emptycoords_from_table(board)
            if not legal_actions:
                break

            opponent = MarkType.RED if current_mark == MarkType.BLU else MarkType.BLU
            prev_opponent_threats = count_threats(board, opponent)
            prev_self_threats = count_threats(board, current_mark)
            opponent_was_about_to_win = _can_win_next(board, opponent)

            state_tensor = _state_tensor(torch.tensor(board, dtype=torch.float32, device=self.device), current_mark)
            logits, value = self.model(state_tensor)
            mask = self._legal_mask(legal_actions)
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            probs = F.softmax(masked_logits, dim=-1)
            dist = Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()

            y, x = _action_to_coords(int(action_idx))
            if not table.mark(y, x, current_mark):
                raise RuntimeError("Illegal move produced by trainer")

            next_board = table.get_table(False).copy()
            new_opponent_threats = count_threats(next_board, opponent)
            new_self_threats = count_threats(next_board, current_mark)
            opponent_can_win_after = _can_win_next(next_board, opponent)

            defense_bonus = self.block_bonus * max(0, prev_opponent_threats - new_opponent_threats)
            offense_bonus = self.offense_bonus * max(0, new_self_threats - prev_self_threats)
            bonus = defense_bonus + offense_bonus

            winner_now = table.get_winner()
            win_bonus = self.win_move_bonus if winner_now == current_mark else 0.0

            threat_delta = max(0, new_self_threats - prev_self_threats)
            threat_bonus = self.create_threat_bonus * threat_delta
            two_way_bonus = self.two_way_threat_bonus if threat_delta >= 2 else 0.0

            block_reward = self.block_threat_bonus if opponent_was_about_to_win and not opponent_can_win_after else 0.0
            block_penalty = self.block_fail_penalty if opponent_was_about_to_win and opponent_can_win_after else 0.0

            transitions.append(
                DeepTransition(
                    log_prob=log_prob,
                    value=value,
                    mark=current_mark,
                    entropy=entropy,
                    bonus=bonus,
                    defense_bonus=defense_bonus,
                    offense_bonus=offense_bonus,
                    win_bonus=win_bonus,
                    block_reward=block_reward,
                    block_penalty=block_penalty,
                    threat_bonus=threat_bonus,
                    two_way_bonus=two_way_bonus,
                )
            )

            if winner_now != MarkType.EMPTY or table.is_full():
                break

            current_mark = opponent

        return table.get_winner(), transitions


    def _reward_map(self, winner: MarkType) -> dict[MarkType, float]:
        if winner == MarkType.BLU:
            return {MarkType.BLU: self.win_reward, MarkType.RED: self.lose_reward}
        if winner == MarkType.RED:
            return {MarkType.BLU: self.lose_reward, MarkType.RED: self.win_reward}
        return {MarkType.BLU: self.draw_reward, MarkType.RED: self.draw_reward}

    def _step(self, transitions: List[DeepTransition], reward_map: dict[MarkType, float]) -> None:
        actor_loss = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)

        for record in transitions:
            target = torch.tensor(reward_map[record.mark], device=self.device)
            advantage = target - record.value
            extras = (
                record.bonus
                + record.win_bonus
                + record.block_reward
                + record.block_penalty
                + record.threat_bonus
                + record.two_way_bonus
            )
            shaped_advantage = advantage + extras
            actor_loss = actor_loss + (-record.log_prob * shaped_advantage)
            actor_loss = actor_loss - self.entropy_coef * record.entropy
            critic_loss = critic_loss + advantage.pow(2)
            self._bonus_total += record.bonus
            self._defense_bonus_total += record.defense_bonus
            self._offense_bonus_total += record.offense_bonus
            self._threat_bonus_total += record.threat_bonus
            self._two_way_bonus_total += record.two_way_bonus
            self._block_reward_total += record.block_reward
            self._block_penalty_total += record.block_penalty
            self._entropy_total += float(record.entropy.detach())

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()


    def _save_checkpoint(self, episode: int) -> Path:
        path = self.weights_dir / f"deep_actor_critic_ep{episode:04d}.pt"
        torch.save(self.model.state_dict(), path)
        torch.save(self.model.state_dict(), self.weights_dir / "deep_actor_critic_latest.pt")
        return path

    def _log_progress(self, episode: int) -> None:
        chunk = max(1, self._chunk_counter)
        bonus_avg = self._bonus_total / chunk
        entropy_avg = self._entropy_total / chunk
        defense_avg = self._defense_bonus_total / chunk
        offense_avg = self._offense_bonus_total / chunk
        threat_avg = self._threat_bonus_total / chunk
        two_way_avg = self._two_way_bonus_total / chunk
        block_avg = self._block_reward_total / chunk
        block_penalty_avg = self._block_penalty_total / chunk
        print(
            f"[Episode {episode}/{self.episodes}] "
            f"BLU wins: {self.stats['blu']}, RED wins: {self.stats['red']}, draws: {self.stats['draw']}. "
            f"Target ~{self.baseline_rating} Elo | bonus {bonus_avg:.4f} (def {defense_avg:.4f}, off {offense_avg:.4f}), "
            f"threat {threat_avg:.4f} (2way {two_way_avg:.4f}), block {block_avg:.4f}, penalty {block_penalty_avg:.4f}, "
            f"entropy {entropy_avg:.4f}"
        )
        self._bonus_total = 0.0
        self._entropy_total = 0.0
        self._chunk_counter = 0
        self._defense_bonus_total = 0.0
        self._offense_bonus_total = 0.0
        self._threat_bonus_total = 0.0
        self._two_way_bonus_total = 0.0
        self._block_reward_total = 0.0
        self._block_penalty_total = 0.0

    def train(self) -> None:
        for episode in range(self._start_episode, self.episodes + 1):
            winner, transitions = self._run_episode()
            if winner == MarkType.BLU:
                self.stats["blu"] += 1
            elif winner == MarkType.RED:
                self.stats["red"] += 1
            else:
                self.stats["draw"] += 1

            rewards = self._reward_map(winner)
            self._step(transitions, rewards)
            self._chunk_counter += 1
            self._last_episode = episode

            if episode % self.progress_every == 0:
                self._log_progress(episode)
            if episode % self.save_every == 0:
                self._save_checkpoint(episode)

        if self.episodes % self.save_every != 0:
            self._save_checkpoint(self.episodes)

        print("Deep training completed.")

    def resume_from_checkpoint(self, path: Path) -> int:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        match = re.search(r"ep(\d+)", path.name)
        episode = int(match.group(1)) if match else 0
        self._last_episode = episode
        self._start_episode = max(self._start_episode, episode + 1)
        return episode

    def save_checkpoint_now(self, episode: int | None = None) -> Path:
        if episode is None:
            episode = max(1, self._last_episode)
        return self._save_checkpoint(episode)


class DeepActorCriticController(BaseController[torch.Tensor, Action]):
    def __init__(
        self,
        model: DeepActorCriticNet,
        mark: MarkType,
        *,
        device: torch.device | None = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.mark = mark
        self.device = device or torch.device("cpu")
        self.deterministic = deterministic

    def model_call(self, model_input: torch.Tensor) -> Action:
        board_tensor = torch.tensor(model_input.copy(), dtype=torch.float32, device=self.device)
        encoded = _state_tensor(board_tensor, self.mark)
        with torch.no_grad():
            logits, _ = self.model(encoded)

        legal_actions = emptycoords_from_table(model_input)
        if not legal_actions:
            raise ValueError("No legal moves left")

        mask = _mask_for_actions(legal_actions, self.device)
        masked_logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(masked_logits, dim=-1)

        if self.deterministic:
            action_idx = int(torch.argmax(probs))
        else:
            dist = Categorical(probs)
            action_idx = int(dist.sample())

        return _action_to_coords(action_idx)


def build_deep_client(
    checkpoint: Path | str,
    *,
    interface: BaseInterface,
    mark: MarkType,
    deterministic: bool = True,
    name: str = "deep-actor-critic",
) -> Client:
    model = DeepActorCriticNet()
    state_dict = torch.load(Path(checkpoint))
    model.load_state_dict(state_dict)
    controller = DeepActorCriticController(
        model=model,
        mark=mark,
        deterministic=deterministic,
    )
    return Client(name=name, mark_type=mark, interface=interface, controller=controller)
