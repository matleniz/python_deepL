import numpy as np

from projet.minimooteur import Connect4Env, env
from projet.play import play
from pettingzoo.classic import connect_four_v3


def test_obs_format_matches_pettingzoo():
    e = env()
    e.reset(seed=0)
    obs, *_ = e.last()
    assert obs["observation"].shape == (6, 7, 2)
    assert obs["action_mask"].shape == (7,)
    assert obs["observation"].dtype == np.int8


def test_play_loop_and_from_observation():
    e = env()
    e.reset(seed=0)
    for name in e.agent_iter():
        obs, rew, term, trunc, _ = e.last()
        if term or trunc:
            e.step(None)
            continue
        mask = obs["action_mask"]
        legal = [i for i, ok in enumerate(mask) if ok]
        e.step(legal[0])
        if np.count_nonzero(e._board) >= 4:
            break

    obs, *_ = e.last()
    cur = e.agent_selection
    e2 = Connect4Env.from_observation(obs["observation"], cur)
    o2, *_ = e2.last()
    assert np.array_equal(o2["observation"], obs["observation"])
    assert e2.agent_selection == cur


def test_winner_horizontal():
    e = env()
    e.reset()
    # P0: cols 0,1,2,3 — P1 plays elsewhere between
    seq = [0, 6, 1, 6, 2, 6, 3]
    for name in e.agent_iter():
        obs, rew, term, trunc, _ = e.last()
        if term or trunc:
            e.step(None)
            continue
        if not seq:
            break
        e.step(seq.pop(0))
    # after 3 for P0, game should have ended on last step
    assert e.terminations["player_0"] is True
