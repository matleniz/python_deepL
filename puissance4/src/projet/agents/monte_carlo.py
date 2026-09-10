"""Monte Carlo plat et MCTS — rollouts via `projet.bitboard`.

Les observations arrivent au format PettingZoo (6,7,2) et sont converties une
seule fois par coup en état bitboard ; tout le reste de la recherche travaille
sur des entiers.

Les deux familles d'agents acceptent soit un budget en simulations (`n_sim`),
soit un budget en secondes par coup (`time_budget`), auquel cas la recherche
tourne jusqu'à l'échéance et le nombre de simulations effectuées est exposé via
`last_simulations`.
"""
from __future__ import annotations

import gc
import math
import random
from time import perf_counter

from projet import bitboard
from projet.bitboard import COL_TOP, COLS

# Les colonnes sont développées du centre vers les bords : à budget faible, les
# premières branches ouvertes sont ainsi les plus prometteuses. `pop()` prenant
# la fin de liste, l'ordre est stocké à l'envers.
_EXPANSION_ORDER = (6, 0, 5, 1, 4, 2, 3)

# Nombre de rollouts par bras et par passe en Monte Carlo plat sous budget
# temps : assez petit pour ne pas dépasser l'échéance, assez grand pour que le
# coût de `perf_counter` reste négligeable.
_MCPLAT_BATCH = 4


def _play_then_rollout(state, action: int) -> float:
    """Joue `action` puis déroule au hasard, vu par le joueur au trait de `state`."""
    next_state, terminal = bitboard.play(state, action)
    if terminal is not None:
        return -terminal
    return -bitboard.rollout(next_state)


def MCplat_rollout(observation, my_name: str, action: int) -> float:
    """Joue `action` depuis l'obs, puis au hasard jusqu'à la fin.

    Return le reward du joueur au trait dans `observation`, c'est-à-dire
    `my_name` (conservé pour la compatibilité de signature ; l'observation est
    déjà exprimée de son point de vue).
    """
    state = bitboard.from_observation(observation)
    action = int(action)
    if not 0 <= action < COLS or state[2][action] >= COL_TOP[action]:
        return -1.0  # coup illégal : défaite immédiate, comme le mini-moteur
    return _play_then_rollout(state, action)


def MCplat_value(observation, my_name: str, action: int, n: int = 40) -> float:
    state = bitboard.from_observation(observation)
    return sum(_play_then_rollout(state, action) for _ in range(n)) / n


def MCplat_select(observation, my_name: str, action_mask, n: int = 40) -> int:
    """Répartit `n` rollouts sur chaque coup légal et garde le meilleur."""
    state = bitboard.from_observation(observation)
    legal = [i for i, ok in enumerate(action_mask) if ok]
    best, best_v = legal[0], float("-inf")
    for a in legal:
        v = sum(_play_then_rollout(state, a) for _ in range(n)) / n
        if v > best_v:
            best, best_v = a, v
    return best


def _mcplat_search(state, legal, n, time_budget) -> tuple[int, int]:
    """Monte Carlo plat. Return (meilleur coup, rollouts effectués).

    Sous budget temps, les bras sont explorés par passes complètes : tous
    reçoivent exactement le même nombre de rollouts, sans biais d'ordre.
    """
    totals = [0.0] * len(legal)
    done = 0

    if time_budget is None:
        for i, a in enumerate(legal):
            totals[i] = sum(_play_then_rollout(state, a) for _ in range(n))
        done = n * len(legal)
        per_arm = n
    else:
        deadline = perf_counter() + time_budget
        per_arm = 0
        while perf_counter() < deadline:
            for i, a in enumerate(legal):
                totals[i] += sum(
                    _play_then_rollout(state, a) for _ in range(_MCPLAT_BATCH)
                )
            per_arm += _MCPLAT_BATCH
        done = per_arm * len(legal)

    if per_arm == 0:  # budget épuisé avant la première passe
        return legal[0], 0
    best = max(range(len(legal)), key=lambda i: totals[i])
    return legal[best], done


class _Node:
    """Nœud de l'arbre. `value_sum` est vu par le joueur au trait dans `state`.

    Aucun lien vers le parent : l'arbre reste acyclique, donc libérable par
    simple comptage de références. Un arbre cyclique ne peut être récupéré que
    par le GC générationnel, dont les passes de génération 2 provoquaient des
    pauses de plusieurs dizaines de millisecondes en plein temps de réflexion.
    La remontée utilise le chemin de descente, conservé par `_simulate`.
    """

    __slots__ = (
        "state",
        "terminal_value",
        "action",
        "children",
        "untried",
        "visits",
        "value_sum",
    )

    def __init__(self, state, terminal_value, action=None):
        self.state = state
        self.terminal_value = terminal_value
        self.action = action
        self.children: list[_Node] = []
        self.visits = 0
        self.value_sum = 0.0
        heights = state[2]
        self.untried = (
            []
            if terminal_value is not None
            else [c for c in _EXPANSION_ORDER if heights[c] < COL_TOP[c]]
        )

    def value_for_parent(self) -> float:
        return -self.value_sum / self.visits


def _uct_select(node: _Node, exploration_weight: float) -> _Node:
    log_parent = math.log(node.visits)
    best, best_score = None, float("-inf")
    for child in node.children:
        score = child.value_for_parent() + exploration_weight * math.sqrt(
            2 * log_parent / child.visits
        )
        if score > best_score:
            best, best_score = child, score
    assert best is not None
    return best


def _expand(node: _Node) -> _Node:
    action = node.untried.pop()
    state, terminal_value = bitboard.play(node.state, action)
    child = _Node(state, terminal_value, action=action)
    node.children.append(child)
    return child


def _simulate(root: _Node, exploration_weight: float, path: list) -> None:
    path.clear()
    path.append(root)
    node = root
    while node.terminal_value is None and not node.untried:
        node = _uct_select(node, exploration_weight)
        path.append(node)

    if node.terminal_value is None:
        node = _expand(node)
        path.append(node)

    value = (
        node.terminal_value
        if node.terminal_value is not None
        else bitboard.rollout(node.state)
    )

    # Remontée en négamax : la valeur change de camp à chaque niveau.
    for ancestor in reversed(path):
        ancestor.visits += 1
        ancestor.value_sum += value
        value = -value


def _mcts_search(state, n_sim, exploration_weight, time_budget) -> tuple[_Node, int]:
    root = _Node(state, None)
    if not root.untried:
        return root, 0

    done = 0
    path: list[_Node] = []  # réutilisé d'une simulation à l'autre
    if time_budget is None:
        while done < n_sim:
            _simulate(root, exploration_weight, path)
            done += 1
    else:
        deadline = perf_counter() + time_budget
        while perf_counter() < deadline:
            _simulate(root, exploration_weight, path)
            done += 1
    return root, done


def _best_action(root: _Node) -> int | None:
    """Coup le plus visité, départagé par sa valeur vue de la racine."""
    if not root.children:
        return None
    return int(max(root.children, key=lambda c: (c.visits, c.value_for_parent())).action)


def _mcts_decide(observation, action_mask, n_sim, exploration_weight, time_budget):
    """Choisit un coup ; sous `time_budget`, toute la décision est chronométrée.

    Le GC est suspendu du début à la fin de la fenêtre : une collecte gen2 au
    milieu (ou juste après la boucle, pendant `_best_action`) coûterait des
    dizaines de millisecondes. L'arbre est acyclique, donc libéré par comptage
    de références avant la reprise du GC — rien ne fuit.
    """
    if time_budget is None:
        state = bitboard.from_observation(observation)
        root, done = _mcts_search(state, n_sim, exploration_weight, None)
        action = _best_action(root)
        if action is None:
            action = int(random.choice([i for i, ok in enumerate(action_mask) if ok]))
        return action, done

    collecting = gc.isenabled()
    if collecting:
        gc.disable()
    root = None
    try:
        deadline = perf_counter() + time_budget
        state = bitboard.from_observation(observation)
        remaining = deadline - perf_counter()
        root, done = _mcts_search(
            state, n_sim, exploration_weight, max(0.0, remaining)
        )
        action = _best_action(root)
        if action is None:
            action = int(random.choice([i for i, ok in enumerate(action_mask) if ok]))
        return action, done
    finally:
        root = None  # libère l'arbre avant que le GC ne reprenne
        if collecting:
            gc.enable()


def MCTS(
    observation,
    my_name: str,
    action_mask,
    n_sim: int = 40,
    exploration_weight: float = 1.0,
    time_budget: float | None = None,
) -> int:
    action, _ = _mcts_decide(
        observation, action_mask, n_sim, exploration_weight, time_budget
    )
    return action


class Agent:
    """MCTS. Par défaut 100 ms/coup (large sous la limite challenge de 250 ms).

    Passer `time_budget=None` pour basculer sur un budget en simulations `n_sim`.
    """

    def __init__(
        self,
        n_sim: int = 40,
        time_budget: float | None = 0.1,
        exploration_weight: float = 1.0,
    ):
        self.env_player_name = ""
        self.episode_index = 0
        self.n_sim = n_sim
        self.time_budget = time_budget
        self.exploration_weight = exploration_weight
        self.last_simulations = 0

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
        action, self.last_simulations = _mcts_decide(
            observation,
            action_mask,
            self.n_sim,
            self.exploration_weight,
            self.time_budget,
        )
        return action


class MCplatAgent:
    """Monte Carlo plat. Par défaut 100 ms/coup.

    Passer `time_budget=None` pour basculer sur `n_sim` rollouts par coup légal.
    """

    def __init__(self, n_sim: int = 40, time_budget: float | None = 0.1):
        self.env_player_name = ""
        self.episode_index = 0
        self.n_sim = n_sim
        self.time_budget = time_budget
        self.last_simulations = 0

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
        if not legal:
            return None
        state = bitboard.from_observation(observation)
        action, self.last_simulations = _mcplat_search(
            state, legal, self.n_sim, self.time_budget
        )
        return int(action)
