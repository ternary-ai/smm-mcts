"""smm — Simultaneous-Move MCTS + CMA-ES self-play training library.

Plug any simultaneous-move game into this library by implementing the
WorldModel protocol.  The library provides:

  SMBot          — SM-MCTS agent with decoupled UCB (game-agnostic)
  CMAESTrainer   — CMA-ES weight optimizer via self-play
  WeightSpec     — descriptor for one weight dimension in the search space
  WorldModel     — Protocol your game engine must satisfy

Quick start
───────────
    from smm import SMBot, CMAESTrainer, WeightSpec, WorldModel

See smm/protocols.py for the full WorldModel interface.
See smm/mcts.py for SMBot configuration options.
See smm/cmaes.py for CMAESTrainer usage and the pool-training algorithm.
"""
from .mcts      import SMBot
from .cmaes     import CMAESTrainer, WeightSpec
from .protocols import WorldModel

__all__ = ["SMBot", "CMAESTrainer", "WeightSpec", "WorldModel"]
__version__ = "0.1.0"
