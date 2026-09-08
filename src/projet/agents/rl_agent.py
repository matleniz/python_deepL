"""Agent RL — à implémenter (+ checkpoints/rl_policy.pt)."""
from __future__ import annotations

import torch


class Agent:
    def __init__(self):
        self.env_player_name = ""
        self.episode_index = 0
        self.device = torch.device("cpu")
        self.policy = None

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
        raise NotImplementedError
