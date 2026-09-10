import random

import numpy as np

from projet import bitboard as bb
from projet.agents.monte_carlo import MCTS
from projet.minimooteur import Connect4Env


def _position(rows):
    """6 chaînes de 7 caractères, haut vers bas : '.' vide, 'x' au trait, 'o' adversaire."""
    obs = np.zeros((6, 7, 2), dtype=np.int8)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "x":
                obs[r, c, 0] = 1
            elif ch == "o":
                obs[r, c, 1] = 1
    mask = np.array(
        [1 if obs[0, c, 0] == 0 and obs[0, c, 1] == 0 else 0 for c in range(7)],
        dtype=np.int8,
    )
    return obs, mask


def test_from_observation_roundtrip():
    rng = random.Random(0)
    for _ in range(50):
        e = Connect4Env()
        e.reset()
        for _ in range(rng.randrange(0, 25)):
            if e.terminations["player_0"]:
                break
            free = [c for c in range(7) if e._board[0, c] == 0]
            if not free:
                break
            e.step(rng.choice(free))
        cur = e.agent_selection
        if cur is None:
            continue
        obs = e.observe(cur)
        state = bb.from_observation(obs["observation"])

        expected = np.zeros((6, 7), dtype=np.int8)
        expected[obs["observation"][:, :, 0] == 1] = 1
        expected[obs["observation"][:, :, 1] == 1] = 2
        assert np.array_equal(bb.to_board(state), expected)
        assert bb.legal_columns(state[2]) == [c for c in range(7) if e._board[0, c] == 0]
        assert state[3] == int(np.count_nonzero(e._board))


def test_won_matches_minimooteur():
    """`bitboard.won` doit être d'accord avec `_winner` sur des parties entières."""
    rng = random.Random(1)
    for _ in range(60):
        e = Connect4Env()
        e.reset()
        while not e.terminations["player_0"]:
            free = [c for c in range(7) if e._board[0, c] == 0]
            if not free:
                break
            cur = e.agent_selection
            state = bb.from_observation(e.observe(cur)["observation"])
            winner = e._winner()
            pid = 1 if cur == "player_0" else 2
            assert bb.won(state[0]) == (winner == pid)
            assert bb.won(state[1]) == (winner == 3 - pid)
            e.step(rng.choice(free))


def test_play_detects_win_and_switches_side():
    obs, _ = _position(
        [
            ".......",
            ".......",
            ".......",
            ".......",
            "..ooo..",
            "..xxx..",
        ]
    )
    state = bb.from_observation(obs)
    _, terminal = bb.play(state, 5)
    assert terminal == -1.0  # vu par l'adversaire, qui vient de perdre

    next_state, terminal = bb.play(state, 0)
    assert terminal is None
    assert next_state[1] == bb.from_observation(obs)[0] | (1 << bb.COL_BASE[0])


def test_rollout_returns_valid_outcome():
    e = Connect4Env()
    e.reset()
    state = bb.from_observation(e.observe("player_0")["observation"])
    assert {bb.rollout(state) for _ in range(200)} <= {-1.0, 0.0, 1.0}


def test_mcts_plays_immediate_win():
    obs, mask = _position(
        [
            ".......",
            ".......",
            ".......",
            ".......",
            "..ooo..",
            "..xxx..",
        ]
    )
    for _ in range(5):
        assert MCTS(obs, "player_0", mask, n_sim=200) in (1, 5)


def test_mcts_blocks_immediate_threat():
    obs, mask = _position(
        [
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            ".ooo.x.",
        ]
    )
    for _ in range(5):
        assert MCTS(obs, "player_0", mask, n_sim=400) in (0, 4)
