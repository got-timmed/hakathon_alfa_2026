from __future__ import annotations

import math
from time import monotonic
from typing import Callable


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(
        self,
        time_limit_seconds: float = 3 * 60 * 60,
        api_budget_rub: float = 600.0,
        finalization_reserve_seconds: float = 10 * 60,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if not math.isfinite(api_budget_rub) or api_budget_rub < 0:
            raise ValueError("api_budget_rub must be non-negative")
        if (
            not math.isfinite(finalization_reserve_seconds)
            or finalization_reserve_seconds < 0
        ):
            raise ValueError("finalization_reserve_seconds must be non-negative")
        if finalization_reserve_seconds >= time_limit_seconds:
            raise ValueError("finalization reserve must be less than the time limit")

        self.time_limit_seconds = time_limit_seconds
        self.api_budget_rub = api_budget_rub
        self.finalization_reserve_seconds = finalization_reserve_seconds
        self._clock = clock
        self._started_at = clock()
        self._api_charged_rub = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.time_limit_seconds - self.elapsed_seconds)

    @property
    def remaining_research_seconds(self) -> float:
        return max(0.0, self.remaining_seconds - self.finalization_reserve_seconds)

    @property
    def api_charged_rub(self) -> float:
        return self._api_charged_rub

    @property
    def remaining_api_budget_rub(self) -> float:
        return max(0.0, self.api_budget_rub - self._api_charged_rub)

    @property
    def research_time_available(self) -> bool:
        return self.remaining_research_seconds > 0

    def ensure_can_spend_api(self, estimated_cost_rub: float) -> None:
        if not math.isfinite(estimated_cost_rub) or estimated_cost_rub < 0:
            raise ValueError("estimated_cost_rub must be non-negative")
        if estimated_cost_rub > self.remaining_api_budget_rub:
            raise BudgetExceeded(
                f"Estimated API cost {estimated_cost_rub:.4f} RUB exceeds "
                f"remaining budget {self.remaining_api_budget_rub:.4f} RUB"
            )

    def charge_api_estimate(self, estimated_cost_rub: float) -> None:
        self.ensure_can_spend_api(estimated_cost_rub)
        self._api_charged_rub += estimated_cost_rub

    def require_research_time(self) -> float:
        remaining = self.remaining_research_seconds
        if remaining <= 0:
            raise BudgetExceeded("Research time limit reached")
        return remaining
