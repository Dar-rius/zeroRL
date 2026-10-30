"""Transition-budget accounting for the scalability benchmark.

Budgets count *individual environment transitions*, never vectorized
``step()`` calls: a batch of N envs contributes N transitions.

Guarantees:
- An exact budget is either met exactly or the tracker records the actual
  count and flags the deviation — a different workload is never silently
  labelled as the requested one.
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when a batch would push the count past the target by more than
    a tolerated overshoot (i.e. the caller is not handling a remainder)."""


class BudgetTracker:
    """Counts transitions against a target budget.

    Args:
        target: requested number of individual env transitions.
        tolerate_overshoot: allow the last batch to overshoot the target by
            at most ``overshoot_tolerance`` transitions (relevant when a
            backend only produces fixed-size batches). The actual count and
            the deviation are always recorded and reported.
        overshoot_tolerance: max tolerated overshoot (defaults to one
            maximal batch, set by the caller).
    """

    def __init__(
        self,
        target: int,
        *,
        tolerate_overshoot: bool = False,
        overshoot_tolerance: int = 0,
    ) -> None:
        if target <= 0:
            raise ValueError(f"target must be > 0, got {target}")
        self.target = target
        self.actual = 0
        self.batches: list[int] = []
        self._tolerate = tolerate_overshoot
        self._tolerance = overshoot_tolerance

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.actual)

    @property
    def done(self) -> bool:
        return self.actual >= self.target

    @property
    def exact(self) -> bool:
        """True iff the recorded count matches the requested budget."""
        return self.actual == self.target

    @property
    def deviation(self) -> int:
        """actual - target (0 when exact)."""
        return self.actual - self.target

    def record(self, transitions: int) -> None:
        """Record a batch of individual transitions."""
        if transitions <= 0:
            raise ValueError(f"batch must record > 0 transitions, got {transitions}")
        projected = self.actual + transitions
        if projected > self.target and not self._tolerate:
            raise BudgetExceeded(
                f"batch of {transitions} overshoots target {self.target} "
                f"(actual {self.actual}); partial batches must be handled "
                f"explicitly by the caller"
            )
        if (
            projected > self.target
            and self._tolerate
            and projected - self.target > self._tolerance
        ):
            raise BudgetExceeded(
                f"overshoot {projected - self.target} exceeds tolerance "
                f"{self._tolerance}"
            )
        self.actual = projected
        self.batches.append(transitions)

    def summary(self) -> dict:
        """Return a JSON-safe summary for result rows."""
        return {
            "target": self.target,
            "actual": self.actual,
            "exact": self.exact,
            "deviation": self.deviation,
            "num_batches": len(self.batches),
        }


def remaining_batches(target: int, batch_size: int) -> int:
    """Number of full batches needed to reach an exact target.

    Raises ValueError when target is not divisible by batch_size — callers
    must then either adjust the batch size or switch to a tracked loop that
    reports the actual count.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if target % batch_size != 0:
        raise ValueError(
            f"target {target} is not divisible by batch_size {batch_size}; "
            f"handle the remainder explicitly"
        )
    return target // batch_size
