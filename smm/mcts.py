"""smm.mcts — Simultaneous-Move MCTS with decoupled UCB.

Algorithm
─────────
At each node, every player independently maximises their own UCB score over
their own action set.  The joint action is the Cartesian product of the
per-player argmaxes.  This decoupled structure:

  • Runs in O(|A₀| + |A₁| + … + |Aₙ|) space per node vs O(|A₀|×…×|Aₙ|)
    for joint-action UCB.
  • Provably converges to Nash equilibrium in two-player zero-sum games
    (Lanctot et al., 2013).
  • Outperforms sequential MCTS (which converges to an exploitable pure
    strategy) on simultaneous-move games; see §11.5 of the Orbit Wars blog.

Key extensions
──────────────
  • Progressive widening: each node starts with 1 action allowed per player
    and widens as visit count grows (ceil(pw_c × (n+1)^pw_alpha)).
  • Cross-turn subtree reuse: the root promoted to the child matching the
    observed post-transition state, carrying accumulated visit statistics.
  • Opponent-model hook: callers can provide an opponent_policy callable that
    replaces standard UCB for non-ego players (e.g. archetype-biased sampling).

Usage
─────
    from smm import SMBot
    from my_game import MyWorldModel

    wm  = MyWorldModel()
    bot = SMBot(wm, num_players=2, weights={"w_material": 0.9})

    action = bot.choose_action(state, player_id=0, deadline=time.monotonic()+0.17)
    concrete = wm.to_concrete(state, player_id=0, abstract_action=action)
"""
from __future__ import annotations

import copy
import math
import random
import time
from typing import Any, Callable

from .protocols import WorldModel

# ── Constants ──────────────────────────────────────────────────────────────────

_UCB_C       = math.sqrt(2)
_DEFAULT_MAX_DEPTH = 3
_DEFAULT_PW_C      = 4.0
_DEFAULT_PW_ALPHA  = 0.5


# ── Tree node ──────────────────────────────────────────────────────────────────

class _Node:
    """A simultaneous-move search node with per-player decoupled UCB tables."""

    __slots__ = ("state", "terminal", "num_players", "candidates", "stats",
                 "children", "n")

    def __init__(
        self,
        state: Any,
        wm: WorldModel,
        num_players: int,
        action_kwargs: dict,
    ) -> None:
        self.state       = state
        self.num_players = num_players
        self.terminal    = wm.is_terminal(state)
        self.n           = 0
        self.children: dict = {}
        if self.terminal:
            self.candidates = None
            self.stats      = None
        else:
            self.candidates = [
                wm.action_candidates(state, pid, **action_kwargs)
                for pid in range(num_players)
            ]
            self.stats = [dict() for _ in range(num_players)]


# ── UCB selection ──────────────────────────────────────────────────────────────

def _select(
    node: _Node,
    pid: int,
    pw_c: float,
    pw_alpha: float,
) -> Any:
    """Decoupled UCB with progressive widening for one player."""
    cands   = node.candidates[pid]
    k       = len(cands)
    allowed = max(1, min(k, math.ceil(pw_c * (node.n + 1) ** pw_alpha)))
    stats   = node.stats[pid]
    log_n   = math.log(node.n + 1)

    best       = cands[0]
    best_score = -math.inf
    for a in cands[:allowed]:
        st = stats.get(a)
        if st is None or st[1] == 0:
            return a
        ucb = st[0] / st[1] + _UCB_C * math.sqrt(log_n / st[1])
        if ucb > best_score:
            best_score = ucb
            best       = a
    return best


def _update(node: _Node, actions: tuple, values: list) -> None:
    """Backpropagate a value vector into per-player UCB tables."""
    node.n += 1
    for pid in range(node.num_players):
        a   = actions[pid]
        st  = node.stats[pid].get(a)
        if st is None:
            node.stats[pid][a] = [values[pid], 1]
        else:
            st[0] += values[pid]
            st[1] += 1


# ── Simulation ─────────────────────────────────────────────────────────────────

def _simulate(
    node: _Node,
    wm: WorldModel,
    weights: dict,
    depth: int,
    max_depth: int,
    pw_c: float,
    pw_alpha: float,
    action_kwargs: dict,
    ego_pid: int,
    opponent_policy: Callable | None,
    rng: random.Random,
) -> list:
    """One SM-MCTS rollout; returns the leaf value vector."""
    if node.terminal:
        return wm.terminal_values(node.state, node.num_players)
    if depth >= max_depth:
        return wm.value_vector(node.state, node.num_players, weights)

    # Per-player action selection
    actions = tuple(
        (opponent_policy(node, pid, rng)
         if (opponent_policy is not None and pid != ego_pid)
         else _select(node, pid, pw_c, pw_alpha))
        for pid in range(node.num_players)
    )

    child = node.children.get(actions)
    if child is None:
        joint = [
            wm.to_concrete(node.state, pid, actions[pid])
            for pid in range(node.num_players)
        ]
        next_state = wm.apply_joint_action(copy.deepcopy(node.state), joint)
        child = _Node(next_state, wm, node.num_players, action_kwargs)
        node.children[actions] = child
        if child.terminal:
            values = wm.terminal_values(child.state, node.num_players)
        else:
            values = wm.value_vector(child.state, node.num_players, weights)
    else:
        values = _simulate(child, wm, weights, depth + 1, max_depth,
                           pw_c, pw_alpha, action_kwargs, ego_pid,
                           opponent_policy, rng)

    _update(node, actions, values)
    return values


# ── Subtree reuse ──────────────────────────────────────────────────────────────

class _TreeCache:
    """Per-player root cache for cross-turn subtree promotion."""

    def __init__(self) -> None:
        self._roots: dict[int, _Node] = {}

    def get_root(
        self,
        wm: WorldModel,
        state: Any,
        player_id: int,
        num_players: int,
        action_kwargs: dict,
    ) -> _Node:
        cached = self._roots.get(player_id)
        sig    = wm.state_signature(state)
        if cached is not None and sig is not None and not cached.terminal:
            for child in cached.children.values():
                if (not child.terminal
                        and wm.state_signature(child.state) == sig):
                    return child
        return _Node(state, wm, num_players, action_kwargs)

    def store(self, player_id: int, root: _Node) -> None:
        self._roots[player_id] = root

    def clear(self) -> None:
        self._roots.clear()


# ── Public API ─────────────────────────────────────────────────────────────────

class SMBot:
    """Simultaneous-Move MCTS agent parameterised by a WorldModel.

    Parameters
    ──────────
    world_model    : WorldModel  — game engine adapter (see protocols.py)
    num_players    : int         — number of players (2 or 4)
    weights        : dict        — value-function coefficients; passed through
                                   to world_model.value_vector()
    max_depth      : int         — rollout depth (default 3)
    pw_c           : float       — progressive-widening coefficient
    pw_alpha       : float       — progressive-widening exponent
    action_kwargs  : dict        — forwarded to world_model.action_candidates()
    opponent_policy: callable    — optional override for opponent action
                                   selection; signature:
                                   (node, pid, rng) -> abstract_action
                                   Use for archetype / prior biasing.
    reuse_tree     : bool        — enable cross-turn subtree reuse (default True)
    """

    def __init__(
        self,
        world_model: WorldModel,
        num_players: int = 2,
        weights: dict | None = None,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        pw_c: float = _DEFAULT_PW_C,
        pw_alpha: float = _DEFAULT_PW_ALPHA,
        action_kwargs: dict | None = None,
        opponent_policy: Callable | None = None,
        reuse_tree: bool = True,
    ) -> None:
        self.wm              = world_model
        self.num_players     = num_players
        self.weights         = weights or {}
        self.max_depth       = max_depth
        self.pw_c            = pw_c
        self.pw_alpha        = pw_alpha
        self.action_kwargs   = action_kwargs or {}
        self.opponent_policy = opponent_policy
        self._cache          = _TreeCache() if reuse_tree else None

    def choose_action(
        self,
        state: Any,
        player_id: int,
        deadline: float | None = None,
        budget_s: float = 1.0,
        rng: random.Random | None = None,
    ) -> Any:
        """Run SM-MCTS and return the best abstract action for player_id.

        Parameters
        ──────────
        state      : current game state (not mutated)
        player_id  : 0-indexed player we are deciding for
        deadline   : absolute time.monotonic() deadline (preferred)
        budget_s   : wall-clock budget in seconds (used when deadline is None)
        rng        : seeded random.Random for reproducibility

        Returns
        ───────
        The abstract action with the highest visit count.  Pass it to
        world_model.to_concrete(state, player_id, action) to get the concrete
        action for apply_joint_action().
        """
        if rng is None:
            rng = random.Random()
        if deadline is None:
            deadline = time.monotonic() + budget_s

        if self._cache is not None:
            root = self._cache.get_root(
                self.wm, state, player_id, self.num_players, self.action_kwargs)
        else:
            root = _Node(state, self.wm, self.num_players, self.action_kwargs)

        if root.terminal or root.candidates is None:
            cands = self.wm.action_candidates(state, player_id, **self.action_kwargs)
            return cands[0] if cands else None

        if len(root.candidates[player_id]) == 1:
            if self._cache is not None:
                self._cache.store(player_id, root)
            return root.candidates[player_id][0]

        while time.monotonic() < deadline:
            _simulate(
                root, self.wm, self.weights,
                0, self.max_depth, self.pw_c, self.pw_alpha,
                self.action_kwargs, player_id, self.opponent_policy, rng,
            )

        if self._cache is not None:
            self._cache.store(player_id, root)

        stats = root.stats[player_id]
        if not stats:
            return root.candidates[player_id][0]

        return max(stats, key=lambda a: (stats[a][1], stats[a][0] / stats[a][1]))

    def reset_tree(self) -> None:
        """Clear the subtree cache (call between independent game episodes)."""
        if self._cache is not None:
            self._cache.clear()
