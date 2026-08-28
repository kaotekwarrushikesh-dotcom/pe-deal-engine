"""The investment committee view: both models side by side, and what the gap means.

The DCF and the LBO are not two estimates of one number. They answer different questions, and
presenting them as competing valuations is the most common way this comparison is misread.

    DCF   What the business is worth to a permanent owner, on its own cash flows, discounted
          at the cost of capital. It has no view on who buys it or how it is financed.

    LBO   What a financial sponsor can pay and still earn a required return, given leverage,
          a finite holding period and an exit. It is a statement about the *buyer's* economics,
          not about intrinsic value.

The gap between them is informative in a specific direction. A sponsor who can pay more than
the DCF says the business is worth is not finding value the DCF missed; they are using
leverage and a shorter horizon to convert the same cash flows into a higher equity return. A
sponsor who cannot reach the DCF value is telling you the deal needs either cheaper debt, a
better operating case, or a higher exit multiple than the entry.

**Every conclusion in this module references calculated model outputs.** Nothing is generic,
and no sentence is produced that the numbers do not support.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pe_engine.assumptions import DealAssumptions
from pe_engine.bridge import ValueBridge
from pe_engine.dcf import DCFResult
from pe_engine.lbo import LBOResult


@dataclass(frozen=True)
class CommitteeView:
    """The comparison table, the conclusions and the deal test."""

    comparison: pd.DataFrame
    dcf_conclusion: str
    lbo_conclusion: str
    assessment: str
    max_entry: dict
    currency: str = "USD"


def compare(assumptions: DealAssumptions, dcf: DCFResult, lbo: LBOResult,
            bridge: ValueBridge, max_entry: dict) -> CommitteeView:
    """Build the committee view from computed model outputs only."""
    currency = assumptions.company.currency
    company = assumptions.company
    sources = lbo.sources_uses

    rows = [
        ("Enterprise value", f"{dcf.enterprise_value:,.0f}",
         f"{sources.entry_enterprise_value:,.0f}"),
        ("Equity value", f"{dcf.equity_value:,.0f}", f"{sources.sponsor_equity:,.0f}"),
        ("Implied share price",
         f"{dcf.implied_share_price:,.2f}" if not np.isnan(dcf.implied_share_price) else "n/a",
         "n/a"),
        ("Entry multiple",
         f"{sources.entry_enterprise_value / assumptions.base_ebitda:,.2f}x"
         if assumptions.base_ebitda else "n/a",
         f"{sources.entry_multiple:.2f}x"),
        ("Exit multiple",
         f"{assumptions.terminal.exit_ebitda_multiple:.2f}x"
         if assumptions.terminal.method == "exit_multiple" else "n/a (Gordon growth)",
         f"{lbo.exit_multiple:.2f}x"),
        ("Discount rate", f"{dcf.wacc:.2%}", "n/a (return is solved, not assumed)"),
        ("Terminal growth",
         f"{assumptions.terminal.terminal_growth:.2%}"
         if assumptions.terminal.method == "gordon" else "n/a (exit multiple)",
         "n/a"),
        ("Sponsor equity", "n/a", f"{lbo.sponsor_equity:,.0f}"),
        ("MOIC", "n/a", f"{lbo.moic:.2f}x" if not np.isnan(lbo.moic) else "n/a"),
        ("IRR", "n/a", f"{lbo.irr:.2%}" if not np.isnan(lbo.irr) else "n/a"),
        ("Holding period", "perpetual", f"{lbo.holding_period} years"),
    ]
    comparison = pd.DataFrame(rows, columns=["Metric", "DCF", "LBO"])

    # --- DCF conclusion -------------------------------------------------------------------------
    if company.share_price > 0 and not np.isnan(dcf.implied_share_price):
        direction = "upside" if dcf.upside >= 0 else "downside"
        dcf_conclusion = (
            f"The DCF values the equity at {currency} {dcf.equity_value:,.0f}, or "
            f"{currency} {dcf.implied_share_price:,.2f} per share against a market price of "
            f"{currency} {company.share_price:,.2f}, representing {abs(dcf.upside):.1%} "
            f"{direction}. {dcf.terminal_share:.0%} of enterprise value sits in the terminal "
            f"value, so that share of the answer rests on assumptions beyond the "
            f"{assumptions.operating.horizon}-year forecast."
        )
    else:
        dcf_conclusion = (
            f"The DCF values the enterprise at {currency} "
            f"{dcf.enterprise_value:,.0f} and the equity at {currency} {dcf.equity_value:,.0f}, "
            f"with {dcf.terminal_share:.0%} of enterprise value in the terminal value. No "
            "market price was entered, so there is no upside comparison."
        )

    # --- LBO conclusion --------------------------------------------------------------------------
    if np.isnan(lbo.irr) or lbo.sponsor_equity <= 0:
        lbo_conclusion = (
            "No sponsor return can be computed: the funding structure leaves no equity "
            "invested, so MOIC and IRR are undefined."
        )
    else:
        top = max(bridge.contribution_pct.items(), key=lambda kv: abs(kv[1]))
        lbo_conclusion = (
            f"At {sources.entry_multiple:.2f}x entry EBITDA the sponsor invests {currency} "
            f"{lbo.sponsor_equity:,.0f} alongside {currency} {sources.total_debt:,.0f} of debt, "
            f"or {lbo.entry_leverage:.1f}x leverage. Exiting at {lbo.exit_multiple:.2f}x after "
            f"{lbo.holding_period} years returns {currency} {lbo.exit_equity:,.0f}, a "
            f"{lbo.moic:.2f}x multiple and a {lbo.irr:.1%} IRR. Net leverage falls from "
            f"{lbo.entry_leverage:.1f}x to {lbo.exit_leverage:.1f}x. The largest single source "
            f"of value creation is {top[0].lower()} at {top[1]:.0%} of the total."
        )

    # --- The deal test ----------------------------------------------------------------------------
    target = assumptions.target_irr
    if np.isnan(lbo.irr):
        assessment = "The transaction cannot be assessed because no return could be computed."
    elif not max_entry.get("achievable", False):
        assessment = (
            f"The transaction returns {lbo.irr:.1%} against a {target:.0%} target, and the "
            f"target is not reachable at any entry price. {max_entry.get('reading', '')} The "
            "binding constraint is the operating case or the capital structure rather than the "
            "valuation, so negotiating price alone will not make this work."
        )
    else:
        margin = max_entry.get("margin_of_safety", float("nan"))
        clears = lbo.irr >= target
        verdict = "clears" if clears else "falls short of"
        assessment = (
            f"The transaction returns {lbo.irr:.1%} against a {target:.0%} target, which "
            f"{verdict} it. The maximum price consistent with the target is "
            f"{max_entry['max_entry_multiple']:.2f}x EBITDA, or {currency} "
            f"{max_entry['max_entry_enterprise_value']:,.0f}, against a proposed "
            f"{sources.entry_multiple:.2f}x. "
        )
        if not np.isnan(margin):
            assessment += (
                f"That leaves {margin:+.1%} of headroom on entry price. "
                if margin >= 0 else
                f"The proposed price is {abs(margin):.1%} above the maximum, so the deal only "
                "works at a lower entry or with a better operating case. "
            )
        if bridge.multiple_change > 0:
            assessment += (
                f"Note that {bridge.contribution_pct['Multiple change']:.0%} of value creation "
                "comes from multiple expansion, which is a bet on exit-market conditions rather "
                "than on the business."
            )
        elif bridge.debt_paydown > abs(bridge.total_created) * 0.5:
            assessment += (
                f"Note that {bridge.contribution_pct['Debt paydown']:.0%} of value creation "
                "comes from debt paydown, which depends on the business continuing to convert "
                "EBITDA into cash rather than on operational improvement."
            )

    return CommitteeView(
        comparison=comparison, dcf_conclusion=dcf_conclusion, lbo_conclusion=lbo_conclusion,
        assessment=assessment, max_entry=max_entry, currency=currency,
    )


def audit_trail(assumptions: DealAssumptions) -> pd.DataFrame:
    """Which input feeds which output, so a number can be traced back to its cause.

    Deliberately a static map of the model's structure rather than a runtime trace. The point
    is for a reader to understand what a change to one input will move before they change it.
    """
    rows = [
        ("Revenue growth", "operating.revenue_growth",
         "Revenue -> EBITDA -> EBIT -> UFCF", "DCF enterprise value, LBO exit EBITDA"),
        ("EBITDA margin", "operating.ebitda_margin",
         "EBITDA -> EBIT -> UFCF, and entry/exit valuation",
         "Both models, and the entry price itself"),
        ("D&A % revenue", "operating.da_pct_revenue",
         "EBIT (lower D&A raises EBIT and therefore tax)", "DCF value, LBO cash taxes"),
        ("Capex % revenue", "operating.capex_pct_revenue",
         "UFCF and cash available for debt service", "DCF value, LBO debt paydown"),
        ("Working capital %", "operating.*_pct_revenue",
         "Change in NWC -> UFCF and cash available", "DCF value, LBO debt paydown"),
        ("Tax rate", "operating.tax_rate",
         "Unlevered tax in the DCF, cash tax after interest in the LBO",
         "Both, by different routes"),
        ("Risk-free, beta, ERP", "wacc.*",
         "Cost of equity -> WACC -> discount factors", "DCF only"),
        ("Terminal growth", "terminal.terminal_growth",
         "Terminal value -> enterprise value", "DCF only, and heavily"),
        ("Entry multiple", "transaction.entry_ebitda_multiple",
         "Entry EV -> sources and uses -> sponsor equity",
         "LBO returns; the denominator of MOIC"),
        ("Debt turns", "transaction.tranches",
         "Debt raised -> sponsor equity, interest, debt schedule",
         "LBO returns, both directions"),
        ("Interest rates", "transaction.tranches",
         "Cash interest -> cash available -> debt paydown", "LBO exit net debt"),
        ("Exit multiple", "transaction.exit_ebitda_multiple",
         "Exit EV -> exit equity", "LBO returns; the numerator of MOIC"),
        ("Holding period", "transaction.holding_period",
         "Years of debt paydown and EBITDA growth before exit",
         "LBO IRR strongly, MOIC weakly"),
    ]
    return pd.DataFrame(rows, columns=["Input", "Field", "Flows through", "Affects"])
