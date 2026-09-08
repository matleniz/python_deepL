"""Éval locale Connect Four (PettingZoo) — adversaire random."""
from __future__ import annotations

import random
from typing import Callable

from pettingzoo.classic import connect_four_v3


def play(agent_factory: Callable[[], object], seed: int, my_seat: int) -> float:
    env = connect_four_v3.env()
    env.reset(seed=seed)
    names, rng, me = list(env.agents), random.Random(seed), agent_factory()
    me.reset(names[my_seat], seed)
    reward = 0.0
    for name in env.agent_iter():
        obs, rew, term, trunc, _ = env.last()
        mine = name == names[my_seat]
        if mine:
            reward = rew
        if term or trunc:
            env.step(None)
            continue
        mask, board = obs["action_mask"], obs["observation"]
        legal = [i for i, ok in enumerate(mask) if ok]
        if mine:
            env.step(me.choose_action(board, rew, False, False, {}, mask))
        else:
            env.step(rng.choice(legal))
    env.close()
    return reward


def evaluate(agent_factory: Callable[[], object], n: int = 400) -> tuple[float, float]:
    rs = [play(agent_factory, s, s % 2) for s in range(n)]
    mean = sum(rs) / len(rs)
    winrate = sum(r > 0 for r in rs) / len(rs)
    return mean, winrate


def duel(
    factory_a: Callable[[], object],
    factory_b: Callable[[], object],
    seed: int,
    a_seat: int = 0,
) -> float:
    """Une partie A vs B. Return le reward de A."""
    env = connect_four_v3.env()
    env.reset(seed=seed)
    names = list(env.agents)
    a = factory_a()
    b = factory_b()
    a.reset(names[a_seat], seed)
    b.reset(names[1 - a_seat], seed)
    reward_a = 0.0
    for name in env.agent_iter():
        obs, rew, term, trunc, _ = env.last()
        seat = names.index(name)
        if seat == a_seat:
            reward_a = rew
        if term or trunc:
            env.step(None)
            continue
        mask, board = obs["action_mask"], obs["observation"]
        agent = a if seat == a_seat else b
        env.step(agent.choose_action(board, rew, False, False, {}, mask))
    env.close()
    return reward_a


def compare(
    factory_a: Callable[[], object],
    factory_b: Callable[[], object],
    n: int = 100,
) -> dict[str, float]:
    """n parties (sièges alternés). Stats du point de vue de A."""
    rs = [duel(factory_a, factory_b, s, a_seat=s % 2) for s in range(n)]
    return {
        "n": float(n),
        "mean_a": sum(rs) / len(rs),
        "winrate_a": sum(r > 0 for r in rs) / len(rs),
        "drawrate": sum(r == 0 for r in rs) / len(rs),
        "winrate_b": sum(r < 0 for r in rs) / len(rs),
    }


def inspect_obs(seed: int = 0) -> None:
    env = connect_four_v3.env()
    env.reset(seed=seed)
    agent = env.agent_selection
    obs, *_ = env.last()
    board, mask = obs["observation"], obs["action_mask"]
    print("board shape:", getattr(board, "shape", type(board)), "dtype:", getattr(board, "dtype", None))
    print("board:\n", board)
    print("mask:", mask, "shape:", getattr(mask, "shape", None))
    print("agents:", env.agents, "current:", agent)
    env.close()


if __name__ == "__main__":
    from projet.agents.random_agent import Agent

    inspect_obs()
    mean, wr = evaluate(Agent, n=50)
    print(f"random  n=50  mean {mean:+.3f}  wins {wr:.3f}")
