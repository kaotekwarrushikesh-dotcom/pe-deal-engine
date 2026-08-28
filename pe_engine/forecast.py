"""The operating forecast. One projection, read by both the DCF and the LBO.

This module builds revenue, EBITDA, D&A, EBIT, working capital and capex forward from the last
historical year. It does not discount anything and knows nothing about capital structure,
which is what allows the same forecast to feed an unlevered valuation and a levered buyout
without either contaminating the other.

**Two sign conventions are enforced here rather than trusted to the caller**, because both are
routine sources of a wrong model:

  Working capital   An *increase* in net working capital consumes cash. The forecast stores
                    the level and the change, and the change carries the sign, so a consumer
                    subtracts it and cannot get the direction wrong by accident.
  Capex             Always stored positive and always subtracted where it appears. Mixing a
                    negative capex convention with a subtraction is the classic way a model
                    quietly adds capital spending to cash flow.

**Working capital moves with the revenue increment, not the revenue level.** A company can
carry a large receivables balance indefinitely and consume no cash, provided the balance is not
growing. Applying the intensity to the change in revenue is what "growth consumes cash" means
arithmetically, and it avoids a phantom first-year outflow equal to the entire opening balance.
"""

from dataclasses import dataclass

import pandas as pd

from pe_engine.assumptions import DealAssumptions


@dataclass(frozen=True)
class Forecast:
    """The projected operating case, one row per forecast year."""

    frame: pd.DataFrame
    horizon: int
    base_revenue: float
    base_ebitda: float

    @property
    def years(self) -> list[int]:
        return list(self.frame["year"])

    def at(self, year: int) -> pd.Series:
        """One forecast year, 1-indexed as a model would label it."""
        if not 1 <= year <= self.horizon:
            raise ValueError(f"year {year} is outside the {self.horizon}-year forecast")
        return self.frame.iloc[year - 1]

    def ebitda_at(self, year: int) -> float:
        return float(self.at(year)["ebitda"])


def build(assumptions: DealAssumptions) -> Forecast:
    """Project the operating case forward from the last historical year."""
    op = assumptions.operating
    if assumptions.base_revenue <= 0:
        raise ValueError(
            "base revenue must be positive: the forecast grows from the last historical year, "
            "so there is nothing to grow from."
        )

    rows = []
    revenue = assumptions.base_revenue
    # The opening working capital balance, needed so the first forecast year charges only the
    # increment rather than the whole balance.
    previous_nwc = assumptions.base_revenue * op.net_working_capital_pct

    for i in range(op.horizon):
        revenue = revenue * (1.0 + op.revenue_growth[i])
        ebitda = revenue * op.ebitda_margin[i]
        da = revenue * op.da_pct_revenue[i]
        ebit = ebitda - da
        capex = revenue * op.capex_pct_revenue[i]

        nwc = revenue * op.net_working_capital_pct
        change_in_nwc = nwc - previous_nwc
        previous_nwc = nwc

        # Tax on unlevered EBIT. Interest is deliberately absent: the DCF's discount rate
        # already carries the debt tax shield, and charging it here as well would value the
        # same benefit twice. The LBO computes its own cash taxes on levered profit separately.
        unlevered_tax = max(ebit, 0.0) * op.tax_rate
        nopat = ebit - unlevered_tax

        rows.append({
            "year": i + 1,
            "revenue": revenue,
            "revenue_growth": op.revenue_growth[i],
            "ebitda": ebitda,
            "ebitda_margin": op.ebitda_margin[i],
            "da": da,
            "ebit": ebit,
            "ebit_margin": ebit / revenue if revenue else float("nan"),
            "unlevered_tax": unlevered_tax,
            "nopat": nopat,
            "capex": capex,
            "nwc": nwc,
            "change_in_nwc": change_in_nwc,
            # Unlevered free cash flow: NOPAT plus D&A (a non-cash charge already deducted to
            # reach EBIT), less capex and less the cash absorbed by working capital growth.
            "ufcf": nopat + da - capex - change_in_nwc,
        })

    return Forecast(
        frame=pd.DataFrame(rows), horizon=op.horizon,
        base_revenue=assumptions.base_revenue, base_ebitda=assumptions.base_ebitda,
    )


def historical_ratios(history: pd.DataFrame) -> pd.DataFrame:
    """Derive the ratios a user needs in order to choose forecast assumptions.

    Supplied so the forecast inputs can be anchored on what the business has actually done
    rather than guessed. Every column is optional: a user entering a partial history should
    get the ratios that history supports rather than an error.
    """
    out = history.copy().sort_values("year").reset_index(drop=True)

    def ratio(numerator: str, denominator: str, name: str):
        if numerator in out.columns and denominator in out.columns:
            out[name] = out[numerator] / out[denominator]

    if "revenue" in out.columns:
        out["revenue_growth"] = out["revenue"].pct_change()
    ratio("gross_profit", "revenue", "gross_margin")
    ratio("ebitda", "revenue", "ebitda_margin")
    ratio("ebit", "revenue", "ebit_margin")
    ratio("net_income", "revenue", "net_margin")
    ratio("capex", "revenue", "capex_pct_revenue")
    ratio("da", "revenue", "da_pct_revenue")

    components = [c for c in ("receivables", "inventory") if c in out.columns]
    if components and "payables" in out.columns:
        out["nwc"] = out[components].sum(axis=1) - out["payables"]
        if "revenue" in out.columns:
            out["nwc_pct_revenue"] = out["nwc"] / out["revenue"]
    return out


def suggest_assumptions(history: pd.DataFrame) -> dict:
    """Averages from history, offered as a starting point rather than applied.

    Returned to the user interface for pre-filling the input fields, so the user can see what
    the business has historically done and then decide. Nothing in the model reads this: the
    assumptions the model uses are whatever the user ends up entering, which is the difference
    between a suggestion and a hidden default.
    """
    ratios = historical_ratios(history)

    def mean_of(column: str) -> float | None:
        if column not in ratios.columns:
            return None
        series = ratios[column].dropna()
        return float(series.mean()) if len(series) else None

    return {
        "revenue_growth": mean_of("revenue_growth"),
        "ebitda_margin": mean_of("ebitda_margin"),
        "da_pct_revenue": mean_of("da_pct_revenue"),
        "capex_pct_revenue": mean_of("capex_pct_revenue"),
        "nwc_pct_revenue": mean_of("nwc_pct_revenue"),
    }
