from projet.play import evaluate, play
from projet.agents.monte_carlo import Agent, MCplat_rollout, MCplat_select
from projet.agents.random_agent import Agent as RandomAgent
from projet.minimooteur import env as mini_env


def test_mcplat_rollout_smoke():
    e = mini_env()
    e.reset(seed=0)
    obs, *_ = e.last()
    r = MCplat_rollout(obs["observation"], "player_0", 3)
    assert r in (-1.0, 0.0, 1.0)


def test_mcplat_select():
    e = mini_env()
    e.reset(seed=0)
    obs, *_ = e.last()
    a = MCplat_select(obs["observation"], "player_0", obs["action_mask"], n=5)
    assert a in range(7)


def test_mcplat_agent_one_game():
    r = play(lambda: Agent(n_sim=5, time_budget=None), seed=0, my_seat=0)
    assert r in (-1.0, 0.0, 1.0)


def test_random_still_ok():
    mean, wr = evaluate(RandomAgent, n=10)
    assert -1.0 <= mean <= 1.0
