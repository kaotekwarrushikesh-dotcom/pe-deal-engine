"""The DCF engine: what the business is worth on its own cash flows.

Discounts unlevered free cash flow at WACC, adds a terminal value, and bridges enterprise
value to an implied share price. Every intermediate figure is retained rather than collapsed
into a total, because the project's transparency requirement means a user must be able to see
the present value of each individual year and reconcile it to the enterprise value by hand.

**Mid-year convention is offered and is off by default.** Cash arrives through the year rather
than in a lump on the last day, so discounting at the year end understates value by roughly
half a year of the discount rate. The convention is a real modelling choice with a defensible
answer either way, so it is exposed rather than decided silently.

**The Gordon growth formula breaks down as growth approaches the discount rate**, and the
failure is not graceful: the denominator goes to zero and the terminal value goes to infinity.
A model that clamps this produces a large number that looks like a valuation. This engine
refuses instead, because a terminal value computed at a growth rate near WACC is not a
conservative estimate of anything, it is an artefact of a division.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pe_engine.assumptions import DealAssumptions
from pe_engine.forecast import Forecast

# Below this gap between WACC and terminal growth the perpetuity is numerically unstable: a
# small change in either input swings the answer wildly, so the result is not a valuation.
MIN_WACC_GROWTH_GAP = 0.005


@dataclass(frozen=True)
class DCFResult:
    """A complete DCF, with every component kept for inspection."""

    wacc: float
    schedule: pd.DataFrame          # per-year FCF, discount factor and present value
    pv_forecast: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    implied_share_price: float
    current_share_price: float
    terminal_method: str
    currency: str = "USD"
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def upside(self) -> float:
        if self.current_share_price <= 0:
            return float("nan")
        return self.implied_share_price / self.current_share_price - 1.0

    @property
    def terminal_share(self) -> float:
        """How much of enterprise value sits beyond the explicit forecast.

        The most honest single read on how much of a DCF is actually modelled. Above roughly
        70% the answer is mostly a statement about a perpetuity formula rather than about the
        cash flows in the forecast.
        """
        if self.enterprise_value == 0:
            return float("nan")
        return self.pv_terminal / self.enterprise_value

    def calculation_trail(self) -> list[str]:
        """The enterprise value, reconstructed line by line.

        Exists so the number can be checked by hand rather than believed. The final line is an
        arithmetic identity: the components must sum to the total.
        """
        lines = [
            f"WACC = {self.wacc:.4%}",
            "",
            "Present value of forecast free cash flow:",
        ]
        for _, row in self.schedule.iterrows():
            lines.append(
                f"  Year {int(row['year'])}: UFCF {row['ufcf']:>12,.1f} "
                f"x discount factor {row['discount_factor']:.4f} = {row['pv']:>12,.1f}"
            )
        lines += [
            f"  Sum of forecast present values{' ' * 18}= {self.pv_forecast:>12,.1f}",
            "",
            f"Terminal value ({self.terminal_method}){' ' * 12}= {self.terminal_value:>12,.1f}",
            f"  discounted to today{' ' * 27}= {self.pv_terminal:>12,.1f}",
            "",
            f"Enterprise value = {self.pv_forecast:,.1f} + {self.pv_terminal:,.1f} "
            f"= {self.enterprise_value:,.1f}",
            f"  less net debt{' ' * 33}= {-self.net_debt:>12,.1f}",
            f"Equity value{' ' * 34}= {self.equity_value:>12,.1f}",
            f"  divided by {self.shares_outstanding:,.1f} shares",
            f"Implied share price{' ' * 27}= {self.implied_share_price:>12,.2f}",
        ]
        return lines


def discount_factors(years: int, wacc: float, mid_year: bool = False) -> np.ndarray:
    """One discount factor per forecast year.

    With the mid-year convention each cash flow is discounted from the middle of its year
    rather than the end, reflecting that revenue arrives continuously rather than on 31
    December.
    """
    periods = np.arange(1, years + 1, dtype=float)
    if mid_year:
        periods = periods - 0.5
    return 1.0 / (1.0 + wacc) ** periods


def terminal_value(
    assumptions: DealAssumptions, forecast: Forecast, wacc: float,
) -> tuple[float, str, list[str]]:
    """The value of everything beyond the forecast horizon, by the chosen method."""
    terminal = assumptions.terminal
    final = forecast.at(forecast.horizon)
    warnings: list[str] = []

    if terminal.method == "exit_multiple":
        value = float(final["ebitda"]) * terminal.exit_ebitda_multiple
        warnings.append(
            f"The terminal value applies today's {terminal.exit_ebitda_multiple:.1f}x multiple "
            f"to year {forecast.horizon} EBITDA. That imports current market pricing into a "
            "valuation of the future, which is circular in a way that is easy to forget: if "
            "multiples are high today, this method assumes they still are at exit."
        )
        return value, "exit multiple", warnings

    gap = wacc - terminal.terminal_growth
    if gap < MIN_WACC_GROWTH_GAP:
        raise ValueError(
            f"Terminal growth of {terminal.terminal_growth:.2%} is too close to the WACC of "
            f"{wacc:.2%}. The Gordon formula divides by their difference, which is "
            f"{gap:.2%} here, so the result is an artefact of the division rather than a "
            "valuation. Either lower terminal growth or use the exit multiple method."
        )

    final_ufcf = float(final["ufcf"])
    value = final_ufcf * (1.0 + terminal.terminal_growth) / gap

    if final_ufcf <= 0:
        warnings.append(
            "Terminal free cash flow is negative, so the perpetuity produces a negative "
            "terminal value. A business consuming cash in its final forecast year cannot be "
            "grown into perpetuity meaningfully; either extend the forecast until cash flow "
            "turns positive or use the exit multiple method."
        )
    if terminal.terminal_growth > 0.04:
        warnings.append(
            f"Terminal growth of {terminal.terminal_growth:.2%} exceeds long-run nominal GDP "
            "growth, which assumes the company grows faster than the economy forever and "
            "therefore eventually becomes the economy."
        )
    return value, "Gordon growth", warnings


def run(assumptions: DealAssumptions, forecast: Forecast,
        mid_year: bool = False, wacc_override: float | None = None) -> DCFResult:
    """Value the business on its unlevered cash flows.

    `wacc_override` exists for the sensitivity engine, which needs to sweep the discount rate
    without rebuilding the whole assumption tree for every cell in a grid.
    """
    wacc = assumptions.wacc.wacc if wacc_override is None else wacc_override
    warnings: list[str] = []
    notes: list[str] = []

    if wacc <= 0:
        raise ValueError(f"WACC must be positive, got {wacc:.4%}.")

    schedule = forecast.frame[["year", "revenue", "ebitda", "ebit", "nopat", "da",
                               "capex", "change_in_nwc", "ufcf"]].copy()
    factors = discount_factors(forecast.horizon, wacc, mid_year)
    schedule["discount_factor"] = factors
    schedule["pv"] = schedule["ufcf"] * factors

    pv_forecast = float(schedule["pv"].sum())
    tv, method, tv_warnings = terminal_value(assumptions, forecast, wacc)
    warnings.extend(tv_warnings)

    # The terminal value sits at the end of the final forecast year, so it is discounted at the
    # full horizon regardless of the mid-year convention applied to the flows themselves.
    pv_terminal = tv / (1.0 + wacc) ** forecast.horizon

    enterprise_value = pv_forecast + pv_terminal
    net_debt = assumptions.company.net_debt
    equity_value = enterprise_value - net_debt
    shares = assumptions.company.shares_outstanding
    implied = equity_value / shares if shares > 0 else float("nan")

    if equity_value < 0:
        warnings.append(
            "Equity value is negative: net debt exceeds the enterprise value. Under limited "
            "liability the shares cannot trade below zero, so the implied price is a measure "
            "of how far underwater the capital structure is rather than a price."
        )

    result = DCFResult(
        wacc=wacc, schedule=schedule, pv_forecast=pv_forecast, terminal_value=tv,
        pv_terminal=pv_terminal, enterprise_value=enterprise_value, net_debt=net_debt,
        equity_value=equity_value, shares_outstanding=shares, implied_share_price=implied,
        current_share_price=assumptions.company.share_price, terminal_method=method,
        currency=assumptions.company.currency,
    )

    if result.terminal_share > 0.75:
        warnings.append(
            f"{result.terminal_share:.0%} of enterprise value sits in the terminal value, so "
            "most of this valuation rests on assumptions beyond the forecast rather than on "
            "the modelled cash flows. The forecast years are doing less work than they appear "
            "to."
        )
    notes.append(
        "Free cash flow is unlevered: tax is charged on EBIT and interest never enters the "
        "cash flow, because WACC already carries the debt tax shield in the discount rate. "
        "Including it in both would value the same benefit twice."
    )
    if mid_year:
        notes.append(
            "Mid-year convention applied: flows are discounted from the middle of each year, "
            "since cash arrives through the year rather than on its final day."
        )

    return DCFResult(**{**result.__dict__,
                        "warnings": tuple(warnings), "notes": tuple(notes)})
