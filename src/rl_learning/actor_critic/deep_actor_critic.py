'''Deep convolutional actor-critic for the 7x7 board.'''

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple, Union

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
            if window_score(board[y, x : x + 4]):
                threats += 1

    for x in range(7):
        for y in range(4):
            if window_score(board[y : y + 4, x]):
                threats += 1

    for y in range(4):
        for x in range(4):
            window = np.array([board[y + i, x + i] for i in range(4)])
            if window_score(window):
                threats += 1

    for y in range(4):
        for x in range(3, 7):
            window = np.array([board[y + i, x - i] for i in range(4)])
            if window_score(window):
                threats += 1

    return threats


def _winner_from_board(board: np.ndarray) -> MarkType:
    def check_line(values: List[int]) -> bool:
        return values[0] != 0 and all(val == values[0] for val in values)

    for y in range(7):
        for x in range(4):
            if check_line([int(board[y, x + i]) for i in range(4)]):
                return MarkType(int(board[y, x]))

    for x in range(7):
        for y in range(4):
            if check_line([int(board[y + i, x]) for i in range(4)]):
                return MarkType(int(board[y, x]))

    for y in range(4):
        for x in range(4):
            if check_line([int(board[y + i, x + i]) for i in range(4)]):
                return MarkType(int(board[y, x]))

    for y in range(4):
        for x in range(3, 7):
            if check_line([int(board[y + i, x - i]) for i in range(4)]):
                return MarkType(int(board[y, x]))

    return MarkType.EMPTY


def _can_win_next(board: np.ndarray, mark: MarkType) -> bool:
    for y, x in emptycoords_from_table(board):
        board[y, x] = int(mark)
        winner = _winner_from_board(board)
        board[y, x] = MarkType.EMPTY
        if winner == mark:
            return True
    return False


def _winning_actions(board: np.ndarray, mark: MarkType, legal_actions: List[Action]) -> List[Action]:
    winning: List[Action] = []
    for y, x in legal_actions:
        board[y, x] = int(mark)
        winner = _winner_from_board(board)
        board[y, x] = MarkType.EMPTY
        if winner == mark:
            winning.append((y, x))
    return winning


def _load_torch_file(path: Path, *, map_location: Union[str, torch.device, None] = None) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_model_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return payload['model_state_dict']
    return payload


def _extract_optimizer_state_dict(payload: Any) -> Optional[dict[str, Any]]:
    if isinstance(payload, dict) and 'optimizer_state_dict' in payload:
        return payload['optimizer_state_dict']
    return None


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + identity)


class DeepActorCriticNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_block = ResidualBlock(128)
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
        x = self.res_block(x)
        x = self.flatten(x)
        shared = self.shared_dense(x)
        shared = self.shared_norm(shared)
        shared = F.relu(shared)
        shared = self.shared_dropout(shared)
        logits = self.actor(shared)
        value = self.critic(shared).squeeze(-1)
        return logits, value


def _state_tensor(board: torch.Tensor, mark: MarkType) -> torch.Tensor:
    # Encode from the current player's perspective so both BLU and RED
    # always see their own stones in the player plane.
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
    defense_bonus: float
    offense_bonus: float
    win_bonus: float
    block_reward: float
    block_penalty: float
    threat_bonus: float
    two_way_bonus: float
    reward: float
    terminal: bool


class DeepActorCriticTrainer:
    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        episodes: int = 5000,
        save_every: int = 100,
        progress_every: int = 50,
        baseline_rating: int = 2000,
        weights_dir: Union[Path, str] = Path('weights'),
        lr: float = 1e-3,
        entropy_coef: float = 0.01,
        entropy_start_coef: float = 0.1,
        entropy_decay_episodes: int = 500000,
        block_bonus: float = 2.0,
        offense_bonus: float = 0.0,
        win_move_bonus: float = 20.0,
        block_threat_bonus: float = 2.5,
        block_fail_penalty: float = -3.0,
        create_threat_bonus: float = 0.0,
        two_way_threat_bonus: float = 0.0,
        win_reward: float = 1.0,
        lose_reward: float = -10.0,
        draw_reward: float = 1.0,
        gamma: float = 0.99,
        greedy_after: int = 200000,
        force_winning_until: int = 700000,
        force_blocking_from: int = 700001,
        opponent_prob: float = 0.2,
        opponent_interval: int = 1000,
        opponent_pool_size: int = 5,
        random_opponent_prob: float = 0.2,
        heuristic_random_move_prob: float = 0.1,
        step_penalty: float = 0.1,
        missed_block_penalty: float = 30.0,
        missed_win_penalty: float = 50.0,
        survival_bonus: float = 0.0,
        risk_penalty: float = -0.25,
        opening_min_moves: int = 2,
        opening_max_moves: int = 4,
        opponent_snapshot_win_rate: float = 0.65,
        opponent_snapshot_window: int = 200,
        opponent_snapshot_cooldown: int = 250,
    ) -> None:
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.episodes = episodes
        self.save_every = save_every
        self.progress_every = progress_every
        self.baseline_rating = baseline_rating
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.opponent_dir = self.weights_dir / 'opponents'
        self.opponent_dir.mkdir(parents=True, exist_ok=True)

        self.model = DeepActorCriticNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.stats = {'blu': 0, 'red': 0, 'draw': 0}
        self._last_episode = 0
        self._start_episode = 1

        self.entropy_end_coef = entropy_coef
        self.entropy_start_coef = max(entropy_start_coef, entropy_coef)
        self.entropy_decay_episodes = max(1, entropy_decay_episodes)
        self._last_entropy_coef = self.entropy_start_coef

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
        self.gamma = gamma
        self.greedy_after = greedy_after
        self.force_winning_until = max(0, force_winning_until)
        self.force_blocking_from = max(0, force_blocking_from)
        self.opponent_prob = max(0.0, min(1.0, opponent_prob))
        self.opponent_interval = max(0, opponent_interval)
        self.opponent_pool_size = max(0, opponent_pool_size)
        self.random_opponent_prob = max(0.0, min(1.0, random_opponent_prob))
        self.heuristic_random_move_prob = max(0.0, min(1.0, heuristic_random_move_prob))
        self.step_penalty = max(0.0, step_penalty)
        self.missed_block_penalty = max(0.0, missed_block_penalty)
        self.missed_win_penalty = max(0.0, missed_win_penalty)
        self.survival_bonus = survival_bonus
        self.risk_penalty = risk_penalty
        self.opening_min_moves = max(0, opening_min_moves)
        self.opening_max_moves = max(self.opening_min_moves, opening_max_moves)
        self.opponent_snapshot_win_rate = max(0.0, min(1.0, opponent_snapshot_win_rate))
        self.opponent_snapshot_cooldown = max(1, opponent_snapshot_cooldown)

        self._opponent_pool: List[DeepActorCriticNet] = []
        self._rng = np.random.default_rng()
        self._chunk_counter = 0
        self._last_snapshot_episode = 0
        self._recent_training_results: Deque[float] = deque(
            maxlen=max(1, opponent_snapshot_window)
        )

        self._reward_total = 0.0
        self._entropy_total = 0.0
        self._defense_bonus_total = 0.0
        self._offense_bonus_total = 0.0
        self._threat_bonus_total = 0.0
        self._two_way_bonus_total = 0.0
        self._block_reward_total = 0.0
        self._block_penalty_total = 0.0
        self._win_bonus_total = 0.0
        self._random_opponent_moves = 0
        self._opponent_move_total = 0

        if self.opponent_pool_size > 0:
            self._snapshot_opponent()

    def _clone_model(self) -> DeepActorCriticNet:
        clone = DeepActorCriticNet()
        state_dict = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        clone.load_state_dict(state_dict)
        clone.to(self.device)
        clone.eval()
        return clone

    def _legal_mask(self, legal_actions: List[Action]) -> torch.Tensor:
        mask = torch.zeros(49, dtype=torch.bool, device=self.device)
        for action in legal_actions:
            mask[action[0] * 7 + action[1]] = True
        return mask

    def _entropy_coef_for_episode(self, episode: int) -> float:
        progress = min(1.0, max(0.0, float(episode - 1) / float(self.entropy_decay_episodes)))
        return self.entropy_start_coef + (self.entropy_end_coef - self.entropy_start_coef) * progress

    def _opening_move_choices(self) -> List[int]:
        choices = [count for count in range(self.opening_min_moves, self.opening_max_moves + 1) if count % 2 == 0]
        return choices or [0]

    def _apply_random_opening(self) -> Tuple[Table, MarkType]:
        choices = self._opening_move_choices()
        if choices == [0]:
            return Table(), MarkType.BLU

        for _ in range(12):
            table = Table()
            current_mark = MarkType.BLU
            opening_moves = int(self._rng.choice(choices))

            for _ in range(opening_moves):
                legal_actions = emptycoords_from_table(table.get_table(False))
                if not legal_actions:
                    break
                idx = int(self._rng.integers(0, len(legal_actions)))
                y, x = legal_actions[idx]
                table.mark(y, x, current_mark)
                if table.get_winner() != MarkType.EMPTY or table.is_full():
                    break
                current_mark = MarkType.RED if current_mark == MarkType.BLU else MarkType.BLU

            if table.get_winner() == MarkType.EMPTY and not table.is_full():
                return table, current_mark

        return Table(), MarkType.BLU

    def _select_opponent_net(self) -> Optional[DeepActorCriticNet]:
        if not self._opponent_pool or self.opponent_prob <= 0.0:
            return None
        if self._rng.random() < self.opponent_prob:
            idx = int(self._rng.integers(0, len(self._opponent_pool)))
            return self._opponent_pool[idx]
        return None

    def _select_heuristic_action(
        self,
        board: np.ndarray,
        legal_actions: List[Action],
        mark: MarkType,
    ) -> Action:
        if self.heuristic_random_move_prob > 0.0 and self._rng.random() < self.heuristic_random_move_prob:
            idx = int(self._rng.integers(0, len(legal_actions)))
            return legal_actions[idx]

        winning_now = _winning_actions(board, mark, legal_actions)
        if winning_now:
            idx = int(self._rng.integers(0, len(winning_now)))
            return winning_now[idx]

        opponent = MarkType.RED if mark == MarkType.BLU else MarkType.BLU
        blocking_moves = _winning_actions(board, opponent, legal_actions)
        if blocking_moves:
            idx = int(self._rng.integers(0, len(blocking_moves)))
            return blocking_moves[idx]

        idx = int(self._rng.integers(0, len(legal_actions)))
        return legal_actions[idx]

    def _load_saved_opponents(self) -> None:
        self._opponent_pool = []
        if self.opponent_pool_size <= 0:
            return
        paths = sorted(self.opponent_dir.glob('opponent_ep*.pt'))[-self.opponent_pool_size :]
        for path in paths:
            try:
                clone = DeepActorCriticNet()
                clone.load_state_dict(
                    _extract_model_state_dict(_load_torch_file(path, map_location='cpu'))
                )
            except (RuntimeError, OSError, EOFError) as exc:
                print(f"Skipping corrupted opponent snapshot {path.name}: {exc}")
                continue
            clone.to(self.device)
            clone.eval()
            self._opponent_pool.append(clone)
        if not self._opponent_pool:
            self._snapshot_opponent()

    def _snapshot_opponent(self, episode: Optional[int] = None) -> None:
        if self.opponent_pool_size <= 0:
            return
        clone = self._clone_model()
        self._opponent_pool.append(clone)
        if len(self._opponent_pool) > self.opponent_pool_size:
            self._opponent_pool.pop(0)
        if episode is not None:
            target_path = self.opponent_dir / f'opponent_ep{episode:07d}.pt'
            temp_path = target_path.with_suffix('.tmp')
            torch.save(clone.state_dict(), temp_path)
            temp_path.replace(target_path)
            self._last_snapshot_episode = episode

    def _record_training_result(self, training_mark: MarkType, winner: MarkType) -> None:
        if winner == training_mark:
            self._recent_training_results.append(1.0)
        elif winner == MarkType.EMPTY:
            self._recent_training_results.append(0.5)
        else:
            self._recent_training_results.append(0.0)

    def _recent_training_win_rate(self) -> Optional[float]:
        if not self._recent_training_results:
            return None
        return float(sum(self._recent_training_results) / len(self._recent_training_results))

    def _should_snapshot_opponent(self, episode: int) -> bool:
        fixed_snapshot_due = self.opponent_interval > 0 and episode % self.opponent_interval == 0
        if fixed_snapshot_due:
            return True
        if self.opponent_pool_size <= 0:
            return False
        if len(self._recent_training_results) < self._recent_training_results.maxlen:
            return False
        if episode - self._last_snapshot_episode < self.opponent_snapshot_cooldown:
            return False
        recent_win_rate = self._recent_training_win_rate()
        return recent_win_rate is not None and recent_win_rate >= self.opponent_snapshot_win_rate

    def _run_episode(
        self,
        episode: int,
        training_mark: MarkType,
        opponent_net: Optional[DeepActorCriticNet],
    ) -> Tuple[MarkType, List[DeepTransition]]:
        table, current_mark = self._apply_random_opening()
        transitions: List[DeepTransition] = []
        deterministic = episode >= self.greedy_after
        step_count = 0

        while True:
            board = table.get_table(False).copy()
            legal_actions = emptycoords_from_table(board)
            if not legal_actions:
                break

            opponent = MarkType.RED if current_mark == MarkType.BLU else MarkType.BLU
            prev_opponent_threats = count_threats(board, opponent)
            prev_self_threats = count_threats(board, current_mark)
            opponent_was_about_to_win = _can_win_next(board, opponent)
            bot_could_win_this_turn = _can_win_next(board, current_mark)

            state_tensor = _state_tensor(
                torch.tensor(board, dtype=torch.float32, device=self.device),
                current_mark,
            )
            is_training_player = current_mark == training_mark
            net = self.model if is_training_player else (opponent_net or self.model)
            mask = self._legal_mask(legal_actions)
            legal_indices = [action[0] * 7 + action[1] for action in legal_actions]

            if is_training_player:
                logits, value = self.model(state_tensor)
            else:
                with torch.no_grad():
                    logits, _ = net(state_tensor)

            logits = logits.flatten()
            masked_logits = logits.masked_fill(~mask, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)

            if is_training_player:
                value = value.squeeze(0)
                winning_moves = _winning_actions(board, current_mark, legal_actions)
                blocking_moves = _winning_actions(board, opponent, legal_actions)
                should_force_winning_move = bool(winning_moves) and episode <= self.force_winning_until
                should_force_blocking_move = (
                    bool(blocking_moves) and episode >= self.force_blocking_from
                )

                if should_force_winning_move:
                    winning_idx = int(self._rng.integers(0, len(winning_moves)))
                    action_tuple = winning_moves[winning_idx]
                    action_idx = torch.tensor(
                        action_tuple[0] * 7 + action_tuple[1],
                        device=self.device,
                    )
                    action_prob = probs[action_idx].clamp(min=1e-9)
                    log_prob = torch.log(action_prob)
                    entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum()
                elif should_force_blocking_move:
                    blocking_idx = int(self._rng.integers(0, len(blocking_moves)))
                    action_tuple = blocking_moves[blocking_idx]
                    action_idx = torch.tensor(
                        action_tuple[0] * 7 + action_tuple[1],
                        device=self.device,
                    )
                    action_prob = probs[action_idx].clamp(min=1e-9)
                    log_prob = torch.log(action_prob)
                    entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum()
                elif deterministic:
                    action_idx = torch.argmax(probs)
                    action_prob = probs[action_idx].clamp(min=1e-9)
                    log_prob = torch.log(action_prob)
                    entropy = torch.tensor(0.0, device=self.device)
                else:
                    dist = Categorical(probs)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                    entropy = dist.entropy()
            else:
                use_random_opponent = (
                    self.random_opponent_prob > 0.0
                    and self._rng.random() < self.random_opponent_prob
                )
                if use_random_opponent:
                    heuristic_action = self._select_heuristic_action(
                        board.copy(),
                        legal_actions,
                        current_mark,
                    )
                    action_idx = torch.tensor(
                        heuristic_action[0] * 7 + heuristic_action[1],
                        device=self.device,
                    )
                    self._random_opponent_moves += 1
                else:
                    action_idx = torch.argmax(probs)
                self._opponent_move_total += 1

            action = int(action_idx.item())
            y, x = _action_to_coords(action)
            if not table.mark(y, x, current_mark):
                raise RuntimeError('Illegal move produced by trainer')

            step_count += 1
            next_board = table.get_table(False).copy()
            new_opponent_threats = count_threats(next_board, opponent)
            new_self_threats = count_threats(next_board, current_mark)
            opponent_can_win_after = _can_win_next(next_board, opponent)

            defense_bonus = self.block_bonus * max(0, prev_opponent_threats - new_opponent_threats)
            offense_bonus = self.offense_bonus * max(0, new_self_threats - prev_self_threats)
            winner_now = table.get_winner()
            win_bonus = self.win_move_bonus if winner_now == current_mark else 0.0
            threat_delta = max(0, new_self_threats - prev_self_threats)
            threat_bonus = self.create_threat_bonus * threat_delta
            two_way_bonus = self.two_way_threat_bonus if threat_delta >= 2 else 0.0
            block_reward = (
                self.block_threat_bonus
                if opponent_was_about_to_win and not opponent_can_win_after
                else 0.0
            )
            block_penalty = (
                self.block_fail_penalty
                if opponent_was_about_to_win and opponent_can_win_after
                else 0.0
            )
            bot_won = winner_now == current_mark
            bot_blocked_threat = opponent_was_about_to_win and not opponent_can_win_after
            step_reward = -self.step_penalty
            if is_training_player and opponent_was_about_to_win and not bot_blocked_threat:
                step_reward -= self.missed_block_penalty
            if is_training_player and bot_could_win_this_turn and not bot_won:
                step_reward -= self.missed_win_penalty
            reward_signal = (
                step_reward
                + win_bonus
                + block_reward
                + block_penalty
                + defense_bonus
                + offense_bonus
                + threat_bonus
                + two_way_bonus
                + self.survival_bonus
            )
            if is_training_player and new_opponent_threats > prev_opponent_threats:
                reward_signal += self.risk_penalty

            terminal = winner_now != MarkType.EMPTY or table.is_full()
            if is_training_player:
                transitions.append(
                    DeepTransition(
                        log_prob=log_prob,
                        value=value,
                        mark=current_mark,
                        entropy=entropy,
                        defense_bonus=defense_bonus,
                        offense_bonus=offense_bonus,
                        win_bonus=win_bonus,
                        block_reward=block_reward,
                        block_penalty=block_penalty,
                        threat_bonus=threat_bonus,
                        two_way_bonus=two_way_bonus,
                        reward=reward_signal,
                        terminal=terminal,
                    )
                )

            if terminal:
                break

            current_mark = opponent

        return table.get_winner(), transitions
    def _reward_map(self, winner: MarkType) -> dict[MarkType, float]:
        if winner == MarkType.BLU:
            return {MarkType.BLU: self.win_reward, MarkType.RED: self.lose_reward}
        if winner == MarkType.RED:
            return {MarkType.BLU: self.lose_reward, MarkType.RED: self.win_reward}
        return {MarkType.BLU: self.draw_reward, MarkType.RED: self.draw_reward}

    def _step(self, transitions: List[DeepTransition], episode: int) -> None:
        if not transitions:
            return

        entropy_coef = self._entropy_coef_for_episode(episode)
        self._last_entropy_coef = entropy_coef

        values = torch.stack([record.value for record in transitions]).view(-1)
        rewards = torch.tensor([record.reward for record in transitions], device=self.device)
        terminals = torch.tensor(
            [record.terminal for record in transitions], dtype=torch.float32, device=self.device
        )
        next_values = torch.cat([values[1:].detach(), torch.zeros_like(values[:1])])
        next_values = next_values * (1.0 - terminals)
        td_targets = rewards + self.gamma * next_values
        advantages = td_targets - values.detach()
        normalized_adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_loss = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)

        for idx, record in enumerate(transitions):
            actor_loss = actor_loss + (-record.log_prob * normalized_adv[idx])
            actor_loss = actor_loss - entropy_coef * record.entropy
            critic_loss = critic_loss + F.mse_loss(values[idx], td_targets[idx], reduction='sum')
            self._reward_total += record.reward
            self._defense_bonus_total += record.defense_bonus
            self._offense_bonus_total += record.offense_bonus
            self._threat_bonus_total += record.threat_bonus
            self._two_way_bonus_total += record.two_way_bonus
            self._block_reward_total += record.block_reward
            self._block_penalty_total += record.block_penalty
            self._win_bonus_total += record.win_bonus
            self._entropy_total += float(record.entropy.detach())

        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        self._chunk_counter += 1

    def _save_checkpoint(self, episode: int) -> Path:
        path = self.weights_dir / f'deep_actor_critic_ep{episode:04d}.pt'
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
        }
        torch.save(checkpoint, path)
        torch.save(checkpoint, self.weights_dir / 'deep_actor_critic_latest.pt')
        return path

    def _log_progress(self, episode: int) -> None:
        chunk = max(1, self._chunk_counter)
        reward_avg = self._reward_total / chunk
        block_avg = self._block_reward_total / chunk
        block_penalty_avg = self._block_penalty_total / chunk
        defense_avg = self._defense_bonus_total / chunk
        offense_avg = self._offense_bonus_total / chunk
        threat_avg = self._threat_bonus_total / chunk
        two_way_avg = self._two_way_bonus_total / chunk
        win_bonus_avg = self._win_bonus_total / chunk
        entropy_avg = self._entropy_total / chunk
        denominator = max(1, self._opponent_move_total)
        random_percent = (self._random_opponent_moves / denominator) * 100
        print(
            f"[Episode {episode}/{self.episodes}] "
            f"BLU wins: {self.stats['blu']}, RED wins: {self.stats['red']}, draws: {self.stats['draw']}. "
            f"Target ~{self.baseline_rating} Elo - reward {reward_avg:.4f} (win {win_bonus_avg:.4f}, "
            f"block {block_avg:.4f}, penalty {block_penalty_avg:.4f}) - "
            f"def {defense_avg:.4f}, off {offense_avg:.4f}, "
            f"threat {threat_avg:.4f} (2way {two_way_avg:.4f}), "
            f"entropy {entropy_avg:.4f}, ent-coef {self._last_entropy_coef:.4f}, "
            f"opponent pool {len(self._opponent_pool)}, heuristic-opponent {random_percent:.2f}%"
        )
        self._chunk_counter = 0
        self._reward_total = 0.0
        self._entropy_total = 0.0
        self._defense_bonus_total = 0.0
        self._offense_bonus_total = 0.0
        self._threat_bonus_total = 0.0
        self._two_way_bonus_total = 0.0
        self._block_reward_total = 0.0
        self._block_penalty_total = 0.0
        self._win_bonus_total = 0.0
        self._random_opponent_moves = 0
        self._opponent_move_total = 0

    def train(self) -> None:
        for episode in range(self._start_episode, self.episodes + 1):
            training_mark = MarkType(self._rng.choice([int(MarkType.BLU), int(MarkType.RED)]))
            opponent_net = self._select_opponent_net()
            winner, transitions = self._run_episode(episode, training_mark, opponent_net)
            self._record_training_result(training_mark, winner)

            if winner == MarkType.BLU:
                self.stats['blu'] += 1
            elif winner == MarkType.RED:
                self.stats['red'] += 1
            else:
                self.stats['draw'] += 1

            reward_map = self._reward_map(winner)
            if transitions:
                transitions[-1].reward += reward_map[transitions[-1].mark]
                self._step(transitions, episode)

            if self._should_snapshot_opponent(episode):
                self._snapshot_opponent(episode)

            self._last_episode = episode

            if episode % self.progress_every == 0:
                self._log_progress(episode)
            if episode % self.save_every == 0:
                self._save_checkpoint(episode)

        if self.episodes % self.save_every != 0:
            self._save_checkpoint(self.episodes)

        print('Deep training completed.')

    def resume_from_checkpoint(self, path: Path) -> int:
        payload = _load_torch_file(path, map_location=self.device)
        self.model.load_state_dict(_extract_model_state_dict(payload))
        optimizer_state = _extract_optimizer_state_dict(payload)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self._load_saved_opponents()
        match = re.search(r'ep(\d+)', path.name)
        if isinstance(payload, dict) and 'episode' in payload:
            episode = int(payload['episode'])
        else:
            episode = int(match.group(1)) if match else 0
        self._last_episode = episode
        self._start_episode = max(self._start_episode, episode + 1)
        return episode

    def save_checkpoint_now(self, episode: Optional[int] = None) -> Path:
        if episode is None:
            episode = max(1, self._last_episode)
        return self._save_checkpoint(episode)


class DeepActorCriticController(BaseController[torch.Tensor, Action]):
    def __init__(
        self,
        model: DeepActorCriticNet,
        mark: MarkType,
        *,
        device: Optional[torch.device] = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.mark = mark
        self.device = device or torch.device('cpu')
        self.deterministic = deterministic

    def model_call(self, model_input: torch.Tensor) -> Action:
        board_tensor = torch.tensor(model_input.copy(), dtype=torch.float32, device=self.device)
        encoded = _state_tensor(board_tensor, self.mark)
        with torch.no_grad():
            logits, _ = self.model(encoded)

        legal_actions = emptycoords_from_table(model_input)
        if not legal_actions:
            raise ValueError('No legal moves left')

        mask = _mask_for_actions(legal_actions, self.device)
        masked_logits = logits.flatten().masked_fill(~mask, -1e9)
        probs = F.softmax(masked_logits, dim=-1)

        if self.deterministic:
            action_idx = int(torch.argmax(probs).item())
        else:
            dist = Categorical(probs)
            action_idx = int(dist.sample().item())

        return _action_to_coords(action_idx)


def build_deep_client(
    checkpoint: Union[Path, str],
    *,
    interface: BaseInterface,
    mark: MarkType,
    deterministic: bool = True,
    name: str = 'deep-actor-critic',
) -> Client:
    model = DeepActorCriticNet()
    payload = _load_torch_file(Path(checkpoint), map_location='cpu')
    model.load_state_dict(_extract_model_state_dict(payload))
    model.eval()
    controller = DeepActorCriticController(
        model=model,
        mark=mark,
        deterministic=deterministic,
    )
    return Client(name=name, mark_type=mark, interface=interface, controller=controller)













