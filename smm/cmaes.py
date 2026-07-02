"""smm.cmaes — CMA-ES self-play weight optimizer.

Finds value-function weight vectors that maximise win rate through iterative
self-play, using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
as the outer optimisation loop.

Algorithm
─────────
1.  Encode: weight dict → normalised numpy vector ∈ [0, 1]ⁿ
2.  CMA-ES generates a population of candidate vectors each generation
3.  Each candidate is evaluated by playing eval_games games against a
    rotating pool of snapshot agents (+ baseline random / greedy opponents)
4.  Fitness = pooled win rate (higher is better)
5.  CMA-ES updates its search distribution
6.  Repeat until n_gens generations or budget exhausted
7.  Decode: best vector → weight dict

Pool training
─────────────
Maintaining a diverse pool of past snapshots (as in AlphaZero / OpenSpiel's
self-play training) prevents weight collapse onto strategies that exploit
only the current opponent distribution.  Snapshots are added whenever the
candidate improves on the current best against the pool.

Usage
─────
    from smm import CMAESTrainer, WeightSpec
    from my_game import MyWorldModel

    spec = [
        WeightSpec("w_material",   0.0, 1.0, 0.5),
        WeightSpec("w_production", 0.0, 1.0, 0.5),
    ]

    trainer = CMAESTrainer(
        world_model     = MyWorldModel(),
        num_players     = 2,
        weight_spec     = spec,
        initial_state_fn= lambda: MyGameState(),
        eval_games      = 6,
        pool_size       = 5,
        popsize         = 12,
        sigma_init      = 0.3,
    )

    best_weights = trainer.run(n_gens=100)
    print(best_weights)
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cma
import numpy as np

from .mcts import SMAgent
from .protocols import WorldModel


# ── Weight spec ────────────────────────────────────────────────────────────────

@dataclass
class WeightSpec:
    """Defines one weight dimension in the CMA-ES search space.

    Attributes
    ──────────
    name    : weight dict key (passed to world_model.value_vector)
    lo, hi  : clamp bounds; CMA-ES stays within [lo, hi] via boundary handling
    default : starting value for the initial mean vector
    fixed   : if True, excluded from search (kept at `default`)
    """
    name:    str
    lo:      float
    hi:      float
    default: float = 0.0
    fixed:   bool  = False


# ── Encoding / decoding ────────────────────────────────────────────────────────

def _encode(weights: dict, spec: list[WeightSpec]) -> np.ndarray:
    """Weight dict → normalised [0, 1]ⁿ vector (free dimensions only)."""
    out = []
    for s in spec:
        if s.fixed:
            continue
        v = float(weights.get(s.name, s.default))
        out.append((v - s.lo) / (s.hi - s.lo) if s.hi > s.lo else 0.0)
    return np.array(out, dtype=float)


def _decode(vec: np.ndarray, spec: list[WeightSpec]) -> dict:
    """Normalised vector → weight dict (clamped to [lo, hi])."""
    weights = {}
    i = 0
    for s in spec:
        if s.fixed:
            weights[s.name] = s.default
        else:
            raw = float(vec[i])
            weights[s.name] = float(np.clip(s.lo + raw * (s.hi - s.lo),
                                            s.lo, s.hi))
            i += 1
    return weights


def _free_dims(spec: list[WeightSpec]) -> int:
    return sum(1 for s in spec if not s.fixed)


# ── Self-play helpers ──────────────────────────────────────────────────────────

def _play_game(
    wm: WorldModel,
    bot_a: SMAgent,
    bot_b: SMAgent,
    init_state: Any,
    num_players: int,
    budget_s: float,
    rng: random.Random,
) -> list[float]:
    """Play one game; returns the terminal value vector."""
    state = copy.deepcopy(init_state)
    bots  = [bot_a, bot_b] if num_players == 2 else [bot_a] + [bot_b] * (num_players - 1)

    while not wm.is_terminal(state):
        joint = []
        for pid in range(num_players):
            bot  = bots[pid]
            rng2 = random.Random(rng.randint(0, 2**31))
            abstract = bot.choose_action(
                state, pid,
                deadline=time.monotonic() + budget_s,
                rng=rng2,
            )
            joint.append(wm.to_concrete(state, pid, abstract))
        state = wm.apply_joint_action(state, joint)

    return wm.terminal_values(state, num_players)


def _eval_weights(
    wm: WorldModel,
    weights: dict,
    pool_weights: list[dict],
    initial_state_fn: Callable[[], Any],
    num_players: int,
    eval_games: int,
    budget_s: float,
    bot_kwargs: dict,
    rng: random.Random,
) -> float:
    """Evaluate a weight candidate against the current pool; return win rate."""
    candidate_kwargs = dict(bot_kwargs, weights=weights)
    pool = pool_weights if pool_weights else [{}]

    wins = 0
    total = 0

    for g in range(eval_games):
        pool_w = pool[g % len(pool)]
        state  = initial_state_fn()

        # Alternate sides to cancel positional advantage
        if g % 2 == 0:
            bot_a = SMAgent(wm, num_players, **candidate_kwargs)
            bot_b = SMAgent(wm, num_players, **dict(bot_kwargs, weights=pool_w))
            vals  = _play_game(wm, bot_a, bot_b, state, num_players, budget_s, rng)
            if vals[0] > 0.6:
                wins += 1
        else:
            bot_a = SMAgent(wm, num_players, **dict(bot_kwargs, weights=pool_w))
            bot_b = SMAgent(wm, num_players, **candidate_kwargs)
            vals  = _play_game(wm, bot_a, bot_b, state, num_players, budget_s, rng)
            if vals[1] > 0.6:
                wins += 1
        total += 1

    return wins / total if total > 0 else 0.0


# ── Trainer ────────────────────────────────────────────────────────────────────

class CMAESTrainer:
    """CMA-ES self-play weight optimizer.

    Parameters
    ──────────
    world_model      : WorldModel  — game engine adapter
    num_players      : int         — 2 or 4
    weight_spec      : list[WeightSpec] — search space definition
    initial_state_fn : () -> state — factory for fresh game states
    eval_games       : int         — games per candidate evaluation (default 6)
    pool_size        : int         — max snapshot pool size (default 5)
    popsize          : int         — CMA-ES population per generation (default 12)
    sigma_init       : float       — initial step size in normalised space (0.3)
    budget_s         : float       — per-turn time budget for self-play games
    bot_kwargs       : dict        — extra SMAgent kwargs (max_depth, pw_c, etc.)
    seed             : int | None  — random seed for reproducibility
    verbose          : bool        — print progress to stdout (default True)
    """

    def __init__(
        self,
        world_model: WorldModel,
        num_players: int,
        weight_spec: list[WeightSpec],
        initial_state_fn: Callable[[], Any],
        eval_games: int = 6,
        pool_size: int = 5,
        popsize: int = 12,
        sigma_init: float = 0.3,
        budget_s: float = 0.05,
        bot_kwargs: dict | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.wm              = world_model
        self.num_players     = num_players
        self.spec            = weight_spec
        self.init_state_fn   = initial_state_fn
        self.eval_games      = eval_games
        self.pool_size       = pool_size
        self.popsize         = popsize
        self.sigma_init      = sigma_init
        self.budget_s        = budget_s
        self.bot_kwargs      = bot_kwargs or {}
        self.seed            = seed
        self.verbose         = verbose
        self._ndim           = _free_dims(weight_spec)

    def run(
        self,
        n_gens: int,
        initial_weights: dict | None = None,
    ) -> dict:
        """Run CMA-ES for n_gens generations and return the best weight dict.

        Parameters
        ──────────
        n_gens          : number of CMA-ES generations to run
        initial_weights : starting weights (decoded to initial mean vector);
                          defaults to each spec's `default` value

        Returns
        ───────
        Best weight dict found (highest pool win rate).
        """
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        # Starting mean
        start_weights = {s.name: s.default for s in self.spec}
        if initial_weights:
            start_weights.update(initial_weights)
        x0 = _encode(start_weights, self.spec)

        # Boundary handling: all dims clipped to [0, 1]
        opts = cma.CMAOptions()
        opts["bounds"]    = [[0.0] * self._ndim, [1.0] * self._ndim]
        opts["popsize"]   = self.popsize
        opts["maxfevals"] = n_gens * self.popsize + 1
        opts["verbose"]   = -9   # suppress CMA-ES internal logging
        opts["seed"]      = self.seed or 0

        es = cma.CMAEvolutionStrategy(x0, self.sigma_init, opts)

        pool: list[dict] = [start_weights]  # start pool with initial weights
        best_weights     = start_weights
        best_wr          = 0.0
        gen              = 0

        if self.verbose:
            print(f"CMA-ES: {self._ndim}-dim search, popsize={self.popsize}, "
                  f"{n_gens} generations, {self.eval_games} games/candidate")
            print(f"{'Gen':>4}  {'Best WR':>7}  {'Pool WR':>7}  "
                  f"{'σ':>7}  Evals")

        while not es.stop() and gen < n_gens:
            candidates = es.ask()
            fitnesses  = []

            for vec in candidates:
                w  = _decode(vec, self.spec)
                wr = _eval_weights(
                    self.wm, w, pool,
                    self.init_state_fn, self.num_players,
                    self.eval_games, self.budget_s,
                    self.bot_kwargs, rng,
                )
                fitnesses.append(-wr)  # CMA-ES minimises

            es.tell(candidates, fitnesses)

            # Track best
            best_idx  = int(np.argmin(fitnesses))
            gen_best_wr = -fitnesses[best_idx]
            if gen_best_wr > best_wr:
                best_wr      = gen_best_wr
                best_weights = _decode(candidates[best_idx], self.spec)
                # Add to pool if room (or replace worst)
                if len(pool) < self.pool_size:
                    pool.append(copy.deepcopy(best_weights))
                else:
                    pool[gen % self.pool_size] = copy.deepcopy(best_weights)

            gen += 1
            if self.verbose and gen % max(1, n_gens // 20) == 0:
                pool_wr = _eval_weights(
                    self.wm, best_weights, pool,
                    self.init_state_fn, self.num_players,
                    self.eval_games * 2, self.budget_s,
                    self.bot_kwargs, rng,
                )
                print(f"{gen:>4}  {best_wr:>7.3f}  {pool_wr:>7.3f}  "
                      f"{es.sigma:>7.4f}  {es.result.evaluations}")

        if self.verbose:
            print(f"\nDone. Best pool win rate: {best_wr:.3f}")
            print(f"Best weights: {best_weights}")

        return best_weights
