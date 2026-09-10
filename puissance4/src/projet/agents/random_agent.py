"""Baseline random légal — pour smoke / format obs."""
from __future__ import annotations

import random


class Agent:
    def __init__(self):
        self.env_player_name = ""
        self.episode_index = 0

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
        legal = [i for i, ok in enumerate(action_mask) if ok]
        return random.choice(legal)
