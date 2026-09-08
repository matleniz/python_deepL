"""Mini-moteur Connect Four — même format d'obs / API proche de PettingZoo classic."""
from __future__ import annotations

from typing import Any, Iterator

import numpy as np

ROWS, COLS = 6, 7
PLAYERS = ("player_0", "player_1")


class Connect4Env:
    """
    API style PettingZoo AEC :
      reset → for agent in agent_iter(): last() → step(action|None)
    Obs : {"observation": (6,7,2) int8, "action_mask": (7,) int8}
      plane 0 = joueur courant, plane 1 = adversaire
    """

    metadata = {"name": "connect4_mini"}

    def __init__(self):
        self.agents: list[str] = []
        self.possible_agents = list(PLAYERS)
        self.agent_selection: str | None = None
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.rewards: dict[str, float] = {}
        self.infos: dict[str, dict] = {}
        self._board = np.zeros((ROWS, COLS), dtype=np.int8)  # 0 vide, 1 P0, 2 P1
        self._cumulative_rewards: dict[str, float] = {}

    # --- factory helpers -------------------------------------------------

    @classmethod
    def from_observation(cls, observation: np.ndarray, whose_turn: str) -> Connect4Env:
        """Recrée un env à partir d'un board PettingZoo (6,7,2) + joueur qui doit jouer."""
        env = cls()
        env.reset()
        me = 1 if whose_turn == "player_0" else 2
        opp = 3 - me
        board = np.asarray(observation, dtype=np.int8)
        env._board = np.zeros((ROWS, COLS), dtype=np.int8)
        env._board[board[:, :, 0] == 1] = me
        env._board[board[:, :, 1] == 1] = opp
        env.agent_selection = whose_turn
        return env

    def copy(self) -> Connect4Env:
        env = Connect4Env()
        env.agents = self.agents[:]
        env.agent_selection = self.agent_selection
        env.terminations = dict(self.terminations)
        env.truncations = dict(self.truncations)
        env.rewards = dict(self.rewards)
        env.infos = {k: dict(v) for k, v in self.infos.items()}
        env._board = self._board.copy()
        env._cumulative_rewards = dict(self._cumulative_rewards)
        return env

    # --- PettingZoo-like API ---------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = list(PLAYERS)
        self.agent_selection = "player_0"
        self.terminations = {a: False for a in PLAYERS}
        self.truncations = {a: False for a in PLAYERS}
        self.rewards = {a: 0.0 for a in PLAYERS}
        self.infos = {a: {} for a in PLAYERS}
        self._cumulative_rewards = {a: 0.0 for a in PLAYERS}
        self._board = np.zeros((ROWS, COLS), dtype=np.int8)
        return self.observe(self.agent_selection), {}

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        me = 1 if agent == "player_0" else 2
        opp = 3 - me
        planes = np.stack(
            [(self._board == me).astype(np.int8), (self._board == opp).astype(np.int8)],
            axis=-1,
        )
        return {"observation": planes, "action_mask": self._action_mask(agent)}

    def last(self) -> tuple[Any, float, bool, bool, dict]:
        agent = self.agent_selection
        assert agent is not None
        return (
            self.observe(agent),
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action: int | None):
        agent = self.agent_selection
        assert agent is not None

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}

        pid = 1 if agent == "player_0" else 2
        opp = "player_1" if agent == "player_0" else "player_0"

        if action is None or not self._legal(action):
            self.rewards[agent] = -1.0
            self.rewards[opp] = 0.0
            self._end()
        else:
            self._drop(action, pid)
            w = self._winner()
            if w == pid:
                self.rewards[agent] = 1.0
                self.rewards[opp] = -1.0
                self._end()
            elif w == 3 - pid:
                self.rewards[agent] = -1.0
                self.rewards[opp] = 1.0
                self._end()
            elif self._full():
                self.rewards[agent] = 0.0
                self.rewards[opp] = 0.0
                self._end()
            else:
                self.agent_selection = opp

        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]

    def agent_iter(self, max_iter: int = 2**63) -> Iterator[str]:
        n = 0
        while self.agents and n < max_iter:
            yield self.agent_selection  # type: ignore[misc]
            n += 1

    def close(self):
        pass

    def action_space(self, agent: str | None = None):
        return type("Discrete", (), {"n": COLS, "sample": lambda mask=None: int(np.random.randint(0, COLS))})()

    def observation_space(self, agent: str | None = None):
        return None

    # --- internals -------------------------------------------------------

    def _action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(COLS, dtype=np.int8)
        if agent != self.agent_selection or self.terminations.get(agent, False):
            return mask
        for c in range(COLS):
            if self._board[0, c] == 0:
                mask[c] = 1
        return mask

    def _legal(self, action: int) -> bool:
        return isinstance(action, (int, np.integer)) and 0 <= int(action) < COLS and self._board[0, int(action)] == 0

    def _drop(self, col: int, pid: int):
        col = int(col)
        for r in range(ROWS - 1, -1, -1):
            if self._board[r, col] == 0:
                self._board[r, col] = pid
                return
        raise RuntimeError("illegal drop")

    def _full(self) -> bool:
        return bool(np.all(self._board[0, :] != 0))

    def _winner(self) -> int:
        b = self._board
        for p in (1, 2):
            for r in range(ROWS):
                for c in range(COLS - 3):
                    if all(b[r, c + i] == p for i in range(4)):
                        return p
            for c in range(COLS):
                for r in range(ROWS - 3):
                    if all(b[r + i, c] == p for i in range(4)):
                        return p
            for r in range(ROWS - 3):
                for c in range(COLS - 3):
                    if all(b[r + i, c + i] == p for i in range(4)):
                        return p
            for r in range(3, ROWS):
                for c in range(COLS - 3):
                    if all(b[r - i, c + i] == p for i in range(4)):
                        return p
        return 0

    def _end(self):
        self.terminations = {a: True for a in PLAYERS}

    def _was_dead_step(self, action: int | None):
        if action is not None:
            raise ValueError("dead step expects action=None")
        agent = self.agent_selection
        assert agent is not None
        if agent in self.agents:
            self.agents.remove(agent)
        self._cumulative_rewards[agent] = 0.0
        if self.agents:
            self.agent_selection = self.agents[0]
        else:
            self.agent_selection = None


def env() -> Connect4Env:
    """Alias style `connect_four_v3.env()`."""
    return Connect4Env()
