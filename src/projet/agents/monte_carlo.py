"""Monte Carlo plat — rollouts via `projet.minimooteur`."""
from __future__ import annotations

import random

from projet.minimooteur import Connect4Env


def MCplat_rollout(observation, my_name: str, action: int) -> float:
    """Joue `action` depuis l'obs, puis random jusqu'à la fin. Return reward de `my_name`."""
    env = Connect4Env.from_observation(observation, my_name)
    reward = 0.0
    first = True
    for name in env.agent_iter():
        obs, rew, term, trunc, _ = env.last()
        if name == my_name:
            reward = rew
        if term or trunc:
            env.step(None)
            continue
        if first:
            env.step(action)
            first = False
        else:
            legal = [i for i, ok in enumerate(obs["action_mask"]) if ok]
            env.step(random.choice(legal))
    env.close()
    return reward


def MCplat_value(observation, my_name: str, action: int, n: int = 40) -> float:
    return sum(MCplat_rollout(observation, my_name, action) for _ in range(n)) / n


def MCplat_select(observation, my_name: str, action_mask, n: int = 40) -> int:
    legal = [i for i, ok in enumerate(action_mask) if ok]
    best, best_v = legal[0], float("-inf")
    for a in legal:
        v = MCplat_value(observation, my_name, a, n=n)
        if v > best_v:
            best, best_v = a, v
    return best


class Agent:
    def __init__(self, n_sim: int = 40):
        self.env_player_name = ""
        self.episode_index = 0
        self.n_sim = n_sim

    def setup(self, observation_space, action_space):
        return True

    def reset(self, env_player_name, episode_index):
        self.env_player_name = env_player_name
        self.episode_index = episode_index
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return int(
            MCplat_select(
                observation, self.env_player_name, action_mask, n=self.n_sim
            )
        )
