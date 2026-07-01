"""smm.protocols — WorldModel interface for the SM-MCTS / CMA-ES library.

Any game can plug into SMBot and CMAESTrainer by providing an object that
satisfies WorldModel.  The type is checked at runtime via isinstance() when
using @runtime_checkable, but duck-typing also works — just implement the
methods with the correct signatures.

Minimal example (Tic-tac-toe, sequential):
    class TicTacToeWM:
        def apply_joint_action(self, state, joint): ...
        def is_terminal(self, state): ...
        def terminal_values(self, state, num_players): ...
        def value_vector(self, state, num_players, weights): ...
        def action_candidates(self, state, player_id, **kw): ...
        def to_concrete(self, state, player_id, abstract): ...
        def state_signature(self, state): ...
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorldModel(Protocol):
    """Minimal contract a game engine must satisfy to use SMBot / CMAESTrainer.

    All methods receive and return plain Python objects; no numpy or torch
    dependencies are required at this layer.
    """

    def apply_joint_action(self, state: Any, joint_action: list) -> Any:
        """Apply joint_action and return the *new* successor state.

        joint_action[i] is the concrete action for player i (whatever format
        your game uses — list of fleet commands, integer move, etc.).
        The original state must NOT be mutated; return a new object.
        """
        ...

    def is_terminal(self, state: Any) -> bool:
        """Return True iff state is a terminal (game-over) position."""
        ...

    def terminal_values(self, state: Any, num_players: int) -> list[float]:
        """Return a length-num_players list of outcome values in [0, 1].

        Typically 1.0 for winner, 0.0 for loser, 0.5 for draw.
        Called only when is_terminal(state) is True.
        """
        ...

    def value_vector(
        self,
        state: Any,
        num_players: int,
        weights: dict,
    ) -> list[float]:
        """Heuristic evaluation of a non-terminal state.

        Returns a length-num_players list; each entry is the estimated value
        for that player in [0, 1].  weights is the dict being optimised by
        CMAESTrainer — your implementation reads whatever keys it needs.
        """
        ...

    def action_candidates(
        self,
        state: Any,
        player_id: int,
        **action_kwargs: Any,
    ) -> list:
        """Return the list of abstract actions available to player_id.

        Each entry must be **hashable** (tuple, int, string, frozenset, …)
        because it is used as a dict key inside SMBot's UCB tables.
        The list must always contain at least one entry (the no-op).
        action_kwargs come from SMBot.action_kwargs, letting callers tune the
        action-space width without changing the WorldModel implementation.
        """
        ...

    def to_concrete(
        self,
        state: Any,
        player_id: int,
        abstract_action: Any,
    ) -> Any:
        """Convert an abstract action to the format expected by apply_joint_action.

        Returns whatever type joint_action[i] should be for your game (e.g.
        a list of fleet-launch commands for Orbit Wars, or an integer for
        board games).
        """
        ...

    def state_signature(self, state: Any) -> Any:
        """Return a hashable signature for cross-turn subtree reuse.

        Two states with the same signature are considered equivalent for
        reuse purposes.  A tuple of (step, sorted planet/piece descriptors)
        works well.  Return None to disable subtree reuse.
        """
        ...
