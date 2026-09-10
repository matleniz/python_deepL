"""Représentation bitboard de Connect Four, pour les rollouts.

Complément de `projet.minimooteur`, qui reste le moteur de référence (API
PettingZoo, lisible, testé). Ce module ne sert qu'aux boucles chaudes : Monte
Carlo plat et MCTS, où le coût par simulation domine tout le reste.

Encodage (classique, dit « Fhourstones ») : une colonne occupe H1 = 7 bits, dont
un bit sentinelle toujours nul en haut. Le bit `H1 * col + row` vaut 1 si le
joueur possède la case, avec row = 0 en bas du plateau. La sentinelle empêche un
alignement de déborder d'une colonne sur la suivante, ce qui permet de détecter
une victoire en quatre décalages au lieu de scanner le plateau.

Un état est le quadruplet `(bb_turn, bb_other, heights, moves)` :
  bb_turn   bits du joueur au trait
  bb_other  bits de son adversaire
  heights   pour chaque colonne, l'indice du prochain bit libre
  moves     nombre de jetons posés (pour détecter le nul sans rescan)
"""
from __future__ import annotations

import random

import numpy as np

ROWS, COLS = 6, 7
H1 = ROWS + 1
SIZE = ROWS * COLS

# Première position hors plateau de chaque colonne : la colonne est pleine
# lorsque sa hauteur atteint cette valeur.
COL_TOP = tuple(H1 * c + ROWS for c in range(COLS))
COL_BASE = tuple(H1 * c for c in range(COLS))

State = tuple[int, int, tuple[int, ...], int]


def won(bb: int) -> bool:
    """Le masque contient-il un alignement de quatre ?"""
    m = bb & (bb >> 1)  # vertical
    if m & (m >> 2):
        return True
    m = bb & (bb >> H1)  # horizontal
    if m & (m >> (2 * H1)):
        return True
    m = bb & (bb >> (H1 - 1))  # diagonale montante
    if m & (m >> (2 * (H1 - 1))):
        return True
    m = bb & (bb >> (H1 + 1))  # diagonale descendante
    return bool(m & (m >> (2 * (H1 + 1))))


def from_observation(observation) -> State:
    """Convertit une obs PettingZoo (6,7,2) en état bitboard.

    Le plan 0 est le joueur au trait, le plan 1 son adversaire, et la ligne 0
    du tableau est le *haut* du plateau — d'où le parcours à l'envers.
    """
    board = np.asarray(observation)
    mine = board[:, :, 0]
    theirs = board[:, :, 1]

    bb_turn = 0
    bb_other = 0
    heights = []
    moves = 0
    for c in range(COLS):
        base = COL_BASE[c]
        row = 0
        for r in range(ROWS - 1, -1, -1):
            if mine[r, c]:
                bb_turn |= 1 << (base + row)
            elif theirs[r, c]:
                bb_other |= 1 << (base + row)
            else:
                break  # gravité : plus rien au-dessus
            row += 1
        heights.append(base + row)
        moves += row
    return bb_turn, bb_other, tuple(heights), moves


def legal_columns(heights) -> list[int]:
    return [c for c in range(COLS) if heights[c] < COL_TOP[c]]


def play(state: State, col: int) -> tuple[State, float | None]:
    """Joue `col` pour le joueur au trait.

    Renvoie le nouvel état (l'adversaire est maintenant au trait) et sa valeur
    terminale *du point de vue du nouveau joueur au trait* : -1.0 s'il vient de
    se faire battre, 0.0 en cas de nul, None si la partie continue.
    """
    bb_turn, bb_other, heights, moves = state
    pos = heights[col]
    bb_turn |= 1 << pos

    next_heights = list(heights)
    next_heights[col] = pos + 1
    moves += 1
    next_state = (bb_other, bb_turn, tuple(next_heights), moves)

    if won(bb_turn):
        return next_state, -1.0
    if moves == SIZE:
        return next_state, 0.0
    return next_state, None


def rollout(state: State) -> float:
    """Partie aléatoire jusqu'au bout, vue par le joueur au trait dans `state`.

    Renvoie +1.0 s'il gagne, -1.0 s'il perd, 0.0 en cas de nul.
    """
    bb_turn, bb_other, heights, moves = state
    h = list(heights)
    playable = legal_columns(h)
    randrange = random.randrange
    sign = 1.0  # +1 tant que c'est le joueur d'origine qui vient de jouer

    while playable:
        i = randrange(len(playable))
        col = playable[i]
        pos = h[col]
        bb_turn |= 1 << pos
        pos += 1
        h[col] = pos
        if pos == COL_TOP[col]:
            playable[i] = playable[-1]
            playable.pop()

        moves += 1
        if won(bb_turn):
            return sign
        if moves == SIZE:
            return 0.0

        bb_turn, bb_other = bb_other, bb_turn
        sign = -sign

    return 0.0


def to_board(state: State) -> np.ndarray:
    """État bitboard -> tableau (6,7) : 0 vide, 1 joueur au trait, 2 adversaire.

    Utilitaire de debug et de test, hors chemin chaud.
    """
    bb_turn, bb_other, _, _ = state
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    for c in range(COLS):
        for row in range(ROWS):
            bit = 1 << (COL_BASE[c] + row)
            r = ROWS - 1 - row
            if bb_turn & bit:
                board[r, c] = 1
            elif bb_other & bit:
                board[r, c] = 2
    return board
