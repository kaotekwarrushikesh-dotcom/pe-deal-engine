"""Sensitivity grids for both models.

A point estimate from a model with a dozen assumptions is a single draw from a distribution the
user cannot see. These grids show the distribution across the two assumptions that matter most
for each method, which is the difference between "the answer is 47" and "the answer is between
38 and 61 depending on two things you cannot observe".

**Two dimensions at a time, not one.** A one-at-a-time table hides interaction, and for both
models the interaction is the point: in a DCF, a low discount rate and high terminal growth
compound rather than add, and in an LBO, entry and exit multiples move returns in opposite
directions so the diagonal is where the deal actually lives.

Every cell is a full model re-run rather than an approximation from a derivative. That is
slower and it means the grid reflects whatever the engine actually does, including its floors,
its refusals and its warnings, rather than a smooth surface that exists only in calculus.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pe_engine import dcf as dcf_engine
from pe_engine import lbo as lbo_engine
from pe_engine.assumptions import DealAssumptions
from pe_engine.forecast import Forecast
from pe_engine import forecast as forecast_engine


def _label(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "x":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def dcf_wacc_vs_growth(
    assumptions: DealAssumptions, forecast: Forecast,
    wacc_range: list[float] | None = None, growth_range: list[float] | None = None,
    output: str = "implied_share_price",
) -> pd.DataFrame:
    """Implied value across discount rate and terminal growth.

    The classic DCF sensitivity, and the one that most often surprises: the corner where WACC
    is low and growth is high is not slightly higher than the centre, it is dramatically so,
    because the two compound in the perpetuity denominator.

    Cells where terminal growth approaches WACC are left empty rather than filled. The engine
    refuses those combinations, and a grid that showed a number there would be presenting an
    artefact of a near-zero denominator as a valuation.
    """
    base_wacc = assumptions.wacc.wacc
    wacc_values = wacc_range or [base_wacc + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    growth_values = growth_range or [
        assumptions.terminal.terminal_growth + d for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]

    rows = []
    for w in wacc_values:
        row = []
        for g in growth_values:
            variant = assumptions.vary(**{"terminal.terminal_growth": g})
            try:
                result = dcf_engine.run(variant, forecast, wacc_override=w)
                row.append(getattr(result, output))
            except ValueError:
                row.append(float("nan"))
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=[_label(w, "pct") for w in wacc_values],
        columns=[_label(g, "pct") for g in growth_values],
    ).rename_axis(index="WACC", columns="Terminal growth")


def dcf_wacc_vs_exit_multiple(
    assumptions: DealAssumptions, forecast: Forecast,
    wacc_range: list[float] | None = None, multiple_range: list[float] | None = None,
) -> pd.DataFrame:
    """Implied share price across discount rate and exit multiple.

    Uses the exit-multiple terminal method, so this grid answers a different question from the
    Gordon one: what the business is worth if it is sold at a market multiple rather than grown
    in perpetuity.
    """
    base_wacc = assumptions.wacc.wacc
    base_multiple = assumptions.terminal.exit_ebitda_multiple
    wacc_values = wacc_range or [base_wacc + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    multiple_values = multiple_range or [base_multiple + d for d in (-2, -1, 0, 1, 2)]

    exit_based = assumptions.vary(**{"terminal.method": "exit_multiple"})
    rows = []
    for w in wacc_values:
        row = []
        for m in multiple_values:
            variant = exit_based.vary(**{"terminal.exit_ebitda_multiple": m})
            try:
                row.append(dcf_engine.run(variant, forecast, wacc_override=w).implied_share_price)
            except ValueError:
                row.append(float("nan"))
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=[_label(w, "pct") for w in wacc_values],
        columns=[_label(m, "x") for m in multiple_values],
    ).rename_axis(index="WACC", columns="Exit multiple")


def lbo_entry_vs_exit(
    assumptions: DealAssumptions, forecast: Forecast,
    entry_range: list[float] | None = None, exit_range: list[float] | None = None,
    metric: str = "irr",
) -> pd.DataFrame:
    """Sponsor returns across entry and exit multiples.

    The single most useful table in a buyout model. It shows the price at which the deal stops
    working, and the exit assumption required to rescue a price that is too high. `metric` is
    "irr" or "moic".
    """
    base_entry = assumptions.transaction.entry_ebitda_multiple
    base_exit = assumptions.transaction.exit_ebitda_multiple
    entry_values = entry_range or [base_entry + d for d in (-2, -1, 0, 1, 2)]
    exit_values = exit_range or [base_exit + d for d in (-2, -1, 0, 1, 2)]

    rows = []
    for entry in entry_values:
        row = []
        for exit_multiple in exit_values:
            try:
                result = lbo_engine.run(assumptions, forecast,
                                        entry_multiple_override=entry,
                                        exit_multiple_override=exit_multiple)
                row.append(result.irr if metric == "irr" else result.moic)
            except Exception:  # noqa: BLE001 - an unsolvable structure leaves the cell empty
                row.append(float("nan"))
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=[_label(v, "x") for v in entry_values],
        columns=[_label(v, "x") for v in exit_values],
    ).rename_axis(index="Entry multiple", columns="Exit multiple")


def lbo_leverage_vs_exit(
    assumptions: DealAssumptions, forecast: Forecast,
    leverage_range: list[float] | None = None, exit_range: list[float] | None = None,
    metric: str = "irr",
) -> pd.DataFrame:
    """Sponsor returns across total leverage and exit multiple.

    Leverage is applied by scaling every non-revolver tranche proportionally, which preserves
    the shape of the capital structure the user designed rather than replacing it with a single
    undifferentiated loan.
    """
    txn = assumptions.transaction
    base_turns = txn.total_turns
    leverage_values = leverage_range or [max(base_turns + d, 0.5) for d in (-2, -1, 0, 1, 2)]
    base_exit = txn.exit_ebitda_multiple
    exit_values = exit_range or [base_exit + d for d in (-2, -1, 0, 1, 2)]

    rows = []
    for turns in leverage_values:
        scale = turns / base_turns if base_turns > 0 else 0.0
        scaled = tuple(
            t if t.is_revolver else
            type(t)(name=t.name, turns_of_ebitda=t.turns_of_ebitda * scale,
                    interest_rate=t.interest_rate, amortisation_pct=t.amortisation_pct,
                    cash_sweep_priority=t.cash_sweep_priority, is_revolver=t.is_revolver)
            for t in txn.tranches)
        levered = assumptions.vary(**{"transaction.tranches": scaled})

        row = []
        for exit_multiple in exit_values:
            try:
                result = lbo_engine.run(levered, forecast,
                                        exit_multiple_override=exit_multiple)
                row.append(result.irr if metric == "irr" else result.moic)
            except Exception:  # noqa: BLE001
                row.append(float("nan"))
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=[f"{v:.1f}x" for v in leverage_values],
        columns=[_label(v, "x") for v in exit_values],
    ).rename_axis(index="Total leverage", columns="Exit multiple")


def dcf_value_drivers(assumptions: DealAssumptions,
                      shock: float = 0.01) -> pd.DataFrame:
    """How much each operating assumption moves the DCF, one at a time.

    A tornado analysis. Each driver is shocked by the same relative amount so the comparison is
    like for like, and the result is ranked by impact, which answers "what should I spend my
    diligence budget on" rather than "what is the answer".
    """
    base_forecast = forecast_engine.build(assumptions)
    base = dcf_engine.run(assumptions, base_forecast).implied_share_price

    drivers = {
        "Revenue growth": ("operating.revenue_growth",
                           tuple(g + shock for g in assumptions.operating.revenue_growth)),
        "EBITDA margin": ("operating.ebitda_margin",
                          tuple(m + shock for m in assumptions.operating.ebitda_margin)),
        "Capex intensity": ("operating.capex_pct_revenue",
                            tuple(c + shock for c in assumptions.operating.capex_pct_revenue)),
        "Working capital intensity": ("operating.receivables_pct_revenue",
                                      assumptions.operating.receivables_pct_revenue + shock),
        "Terminal growth": ("terminal.terminal_growth",
                            assumptions.terminal.terminal_growth + shock),
        "Tax rate": ("operating.tax_rate", assumptions.operating.tax_rate + shock),
    }

    rows = []
    for label, (path, value) in drivers.items():
        variant = assumptions.vary(**{path: value})
        try:
            shocked_forecast = forecast_engine.build(variant)
            shocked = dcf_engine.run(variant, shocked_forecast).implied_share_price
            impact = shocked - base
        except (ValueError, ZeroDivisionError):
            shocked, impact = float("nan"), float("nan")
        rows.append({
            "driver": label,
            "base_value": base,
            "shocked_value": shocked,
            "impact": impact,
            "impact_pct": impact / base if base else float("nan"),
        })

    frame = pd.DataFrame(rows)
    return frame.reindex(frame["impact"].abs().sort_values(ascending=False).index).reset_index(
        drop=True)
