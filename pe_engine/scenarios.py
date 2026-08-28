"""Bear, base and bull, each a full independent revaluation.

A downside case is not the base case multiplied by 0.8. A real downside compounds slower
growth, compressed margins, higher capital intensity, a higher discount rate and a lower exit
multiple, and those interact: lower EBITDA both reduces the exit value and slows the debt
paydown that would have cushioned it. A percentage haircut on the base answer cannot represent
that, and it systematically understates how bad a bad case is in a levered structure.

So each scenario builds its own assumption tree with `vary()` and runs the whole model from
first principles. Nothing is scaled from the base case.

**The scenario definitions are the user's, not the model's.** The defaults here are deliberate
shifts rather than a house view, and every one of them is overridable. A model that ships an
opinionated bear case is telling the user what to be worried about.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pe_engine import bridge as bridge_engine
from pe_engine import dcf as dcf_engine
from pe_engine import forecast as forecast_engine
from pe_engine import lbo as lbo_engine
from pe_engine.assumptions import DealAssumptions


@dataclass(frozen=True)
class ScenarioDefinition:
    """A named set of shifts from the base assumptions.

    Expressed as deltas rather than absolutes so a scenario stays meaningful when the base case
    changes. A bear case defined as "growth of 2%" becomes wrong the moment the user revises
    the base to 1%; defined as "two points below base" it stays correct.
    """

    name: str
    revenue_growth_delta: float = 0.0
    ebitda_margin_delta: float = 0.0
    capex_delta: float = 0.0
    wacc_delta: float = 0.0
    terminal_growth_delta: float = 0.0
    exit_multiple_delta: float = 0.0
    leverage_scale: float = 1.0


DEFAULT_SCENARIOS = (
    ScenarioDefinition(
        "Bear", revenue_growth_delta=-0.03, ebitda_margin_delta=-0.03, capex_delta=0.01,
        wacc_delta=0.01, terminal_growth_delta=-0.005, exit_multiple_delta=-2.0,
        leverage_scale=1.0),
    ScenarioDefinition("Base"),
    ScenarioDefinition(
        "Bull", revenue_growth_delta=0.02, ebitda_margin_delta=0.02, capex_delta=-0.005,
        wacc_delta=-0.01, terminal_growth_delta=0.005, exit_multiple_delta=2.0,
        leverage_scale=1.0),
)


def apply(assumptions: DealAssumptions, definition: ScenarioDefinition) -> DealAssumptions:
    """Build the assumption tree for one scenario.

    Margins are floored just above zero rather than allowed negative: a bear case that turns
    the business loss-making cannot be exited on an EBITDA multiple, and the model would refuse
    rather than produce a downside number. Flooring keeps the scenario informative and the
    floor is disclosed in the result.
    """
    op = assumptions.operating
    changes = {
        "operating.revenue_growth": tuple(
            g + definition.revenue_growth_delta for g in op.revenue_growth),
        "operating.ebitda_margin": tuple(
            max(m + definition.ebitda_margin_delta, 0.01) for m in op.ebitda_margin),
        "operating.capex_pct_revenue": tuple(
            max(c + definition.capex_delta, 0.0) for c in op.capex_pct_revenue),
        "terminal.terminal_growth": (
            assumptions.terminal.terminal_growth + definition.terminal_growth_delta),
        "transaction.exit_ebitda_multiple": max(
            assumptions.transaction.exit_ebitda_multiple + definition.exit_multiple_delta, 0.5),
        "label": definition.name,
    }

    varied = assumptions.vary(**changes)

    if definition.wacc_delta:
        # WACC is derived from its components, so the shift is applied to the cost of equity,
        # which is where a change in perceived risk would actually show up.
        varied = varied.vary(**{
            "wacc.equity_risk_premium": (
                assumptions.wacc.equity_risk_premium
                + definition.wacc_delta / max(assumptions.wacc.beta, 0.01))
        })

    if definition.leverage_scale != 1.0:
        scaled = tuple(
            t if t.is_revolver else
            type(t)(name=t.name,
                    turns_of_ebitda=t.turns_of_ebitda * definition.leverage_scale,
                    interest_rate=t.interest_rate, amortisation_pct=t.amortisation_pct,
                    cash_sweep_priority=t.cash_sweep_priority, is_revolver=t.is_revolver)
            for t in assumptions.transaction.tranches)
        varied = varied.vary(**{"transaction.tranches": scaled})

    return varied


@dataclass(frozen=True)
class ScenarioOutcome:
    """One scenario, fully computed."""

    name: str
    assumptions: DealAssumptions
    enterprise_value: float
    equity_value: float
    implied_share_price: float
    upside: float
    sponsor_equity: float
    moic: float
    irr: float
    exit_leverage: float
    error: str | None = None


def run_all(assumptions: DealAssumptions,
            definitions: tuple[ScenarioDefinition, ...] = DEFAULT_SCENARIOS,
            ) -> list[ScenarioOutcome]:
    """Run every scenario from first principles.

    A scenario that cannot be computed is returned with its error rather than dropped, because
    the fact that the bear case breaks the capital structure is the single most useful thing a
    downside scenario can tell you.
    """
    outcomes = []
    for definition in definitions:
        variant = apply(assumptions, definition)
        try:
            fc = forecast_engine.build(variant)
            valuation = dcf_engine.run(variant, fc)
            buyout = lbo_engine.run(variant, fc)
            outcomes.append(ScenarioOutcome(
                name=definition.name, assumptions=variant,
                enterprise_value=valuation.enterprise_value,
                equity_value=valuation.equity_value,
                implied_share_price=valuation.implied_share_price,
                upside=valuation.upside,
                sponsor_equity=buyout.sponsor_equity,
                moic=buyout.moic, irr=buyout.irr,
                exit_leverage=buyout.exit_leverage,
            ))
        except Exception as exc:  # noqa: BLE001 - a broken scenario is a finding
            outcomes.append(ScenarioOutcome(
                name=definition.name, assumptions=variant,
                enterprise_value=float("nan"), equity_value=float("nan"),
                implied_share_price=float("nan"), upside=float("nan"),
                sponsor_equity=float("nan"), moic=float("nan"), irr=float("nan"),
                exit_leverage=float("nan"), error=str(exc),
            ))
    return outcomes


def frame(outcomes: list[ScenarioOutcome]) -> pd.DataFrame:
    """The scenario table, for display."""
    return pd.DataFrame([{
        "Scenario": o.name,
        "DCF enterprise value": o.enterprise_value,
        "DCF equity value": o.equity_value,
        "Implied share price": o.implied_share_price,
        "Upside": o.upside,
        "Sponsor equity": o.sponsor_equity,
        "MOIC": o.moic,
        "IRR": o.irr,
        "Exit leverage": o.exit_leverage,
        "Status": o.error or "computed",
    } for o in outcomes])
