"""Model validation: refuse to produce a number that cannot mean anything.

A financial model's most dangerous failure is not a crash. It is an output that looks like a
valuation, sits in the right cell, has the right number of decimal places, and is nonsense
because an input made the arithmetic meaningless. Every check here exists because there is a
specific way the model can produce such a number.

Checks are tiered the same way Module 3 tiers data quality, and for the same reason:

  BLOCKING   The result would be meaningless. The model refuses to run.
  WARNING    The result is computable but should be read with the warning attached.
  NOTE       A modelling choice worth surfacing, not a problem.

The distinction matters because a model that blocks on everything is unusable and one that
blocks on nothing is untrustworthy. WACC below terminal growth is blocking, since the
perpetuity is then negative or infinite. Leverage at seven turns is a warning, since it is
aggressive but perfectly computable.
"""

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["blocking", "warning", "note"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    field_name: str
    message: str

    def render(self) -> str:
        icon = {"blocking": "BLOCKING", "warning": "WARNING", "note": "NOTE"}[self.severity]
        return f"[{icon}] {self.field_name}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, severity: Severity, field_name: str, message: str) -> None:
        self.issues.append(Issue(severity, field_name, message))

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def notes(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "note"]

    @property
    def ok(self) -> bool:
        """Whether the model may run at all."""
        return not self.blocking

    def summary(self) -> str:
        if self.blocking:
            return (f"{len(self.blocking)} blocking issue(s) prevent the model from running, "
                    f"plus {len(self.warnings)} warning(s).")
        if self.warnings:
            return f"Model runs, with {len(self.warnings)} warning(s) to read alongside it."
        return "No issues found."


def validate(assumptions) -> ValidationReport:
    """Check an assumption tree before anything is computed from it."""
    report = ValidationReport()
    op = assumptions.operating
    wacc_inputs = assumptions.wacc
    terminal = assumptions.terminal
    txn = assumptions.transaction
    company = assumptions.company

    # --- The DCF's structural requirement ------------------------------------------------------
    wacc = wacc_inputs.wacc
    if wacc <= 0:
        report.add("blocking", "WACC",
                   f"WACC is {wacc:.2%}. A non-positive discount rate means future cash is "
                   "worth more than present cash, which inverts the entire valuation.")
    elif terminal.method == "gordon" and terminal.terminal_growth >= wacc:
        report.add("blocking", "Terminal growth",
                   f"Terminal growth of {terminal.terminal_growth:.2%} is at or above the WACC "
                   f"of {wacc:.2%}. The Gordon formula divides by their difference, so the "
                   "terminal value is negative or infinite rather than large.")
    elif terminal.method == "gordon" and (wacc - terminal.terminal_growth) < 0.005:
        report.add("warning", "Terminal growth",
                   f"Terminal growth is within {(wacc - terminal.terminal_growth):.2%} of the "
                   "WACC. The perpetuity is numerically unstable this close to the boundary: a "
                   "small change in either input swings the valuation wildly.")

    if terminal.method == "gordon" and terminal.terminal_growth > 0.04:
        report.add("warning", "Terminal growth",
                   f"Terminal growth of {terminal.terminal_growth:.2%} exceeds long-run nominal "
                   "GDP growth, which assumes the company eventually becomes the economy.")

    # --- Inputs that must exist for the arithmetic to mean anything ------------------------------
    if assumptions.base_revenue <= 0:
        report.add("blocking", "Base revenue",
                   "Base revenue must be positive: the forecast grows from the last historical "
                   "year, so there is nothing to grow from.")
    if assumptions.base_ebitda <= 0:
        report.add("blocking", "Base EBITDA",
                   "Entry EBITDA must be positive. A buyout is priced as a multiple of EBITDA, "
                   "so a loss-making business cannot be valued this way.")
    if company.shares_outstanding <= 0:
        report.add("warning", "Shares outstanding",
                   "Shares outstanding is zero, so no implied share price can be computed. "
                   "Enterprise and equity value are still valid.")
    if company.share_price <= 0:
        report.add("note", "Share price",
                   "No current share price entered, so upside against the market cannot be "
                   "shown. This is normal for a private company.")

    # --- Operating assumptions --------------------------------------------------------------------
    if any(m <= 0 for m in op.ebitda_margin):
        report.add("blocking", "EBITDA margin",
                   "At least one forecast year has a non-positive EBITDA margin. The LBO prices "
                   "and exits on EBITDA, so the model cannot proceed.")
    if any(m > 0.60 for m in op.ebitda_margin):
        report.add("warning", "EBITDA margin",
                   f"Peak forecast EBITDA margin is {max(op.ebitda_margin):.1%}. That is very "
                   "high outside software and licensing, and is worth checking against the "
                   "company's own history.")
    if any(g > 0.30 for g in op.revenue_growth):
        report.add("warning", "Revenue growth",
                   f"Peak forecast revenue growth is {max(op.revenue_growth):.1%}. Sustained "
                   "growth at that rate is rare and is usually the assumption a deal turns on.")
    if not 0.0 <= op.tax_rate < 1.0:
        report.add("blocking", "Tax rate",
                   f"Tax rate of {op.tax_rate:.1%} is outside the range 0% to 100%.")

    # --- Capital structure ----------------------------------------------------------------------------
    if not txn.tranches:
        report.add("warning", "Debt structure",
                   "No debt tranches defined, so this is an all-equity purchase rather than a "
                   "leveraged buyout. The model still runs and the returns are unlevered.")
    if txn.total_turns > 7.0:
        report.add("warning", "Leverage",
                   f"Entry leverage of {txn.total_turns:.1f}x EBITDA is aggressive. Above "
                   "roughly six turns, lenders normally impose covenants this model does not "
                   "represent, so the debt schedule may be more permissive than reality.")
    if txn.entry_ebitda_multiple <= 0:
        report.add("blocking", "Entry multiple",
                   "Entry multiple must be positive.")
    if txn.exit_ebitda_multiple <= 0:
        report.add("blocking", "Exit multiple",
                   "Exit multiple must be positive.")
    if txn.exit_ebitda_multiple > txn.entry_ebitda_multiple:
        report.add("note", "Exit multiple",
                   f"The exit multiple ({txn.exit_ebitda_multiple:.1f}x) is above the entry "
                   f"multiple ({txn.entry_ebitda_multiple:.1f}x), so the deal assumes multiple "
                   "expansion. That is a bet on the market rather than on the business, and the "
                   "value bridge shows how much of the return depends on it.")

    if txn.holding_period > op.horizon:
        report.add("blocking", "Holding period",
                   f"The {txn.holding_period}-year holding period exceeds the {op.horizon}-year "
                   "forecast, so there is no modelled EBITDA to exit on.")

    for tranche in txn.tranches:
        if tranche.interest_rate < 0:
            report.add("blocking", f"{tranche.name} interest",
                       "Interest rate cannot be negative.")
        if tranche.amortisation_pct < 0 or tranche.amortisation_pct > 1:
            report.add("blocking", f"{tranche.name} amortisation",
                       "Amortisation must be between 0% and 100% of original principal a year.")

    if txn.interest_basis == "average":
        report.add("note", "Interest basis",
                   "Average-balance interest is genuinely circular: interest depends on the "
                   "debt balance, which depends on repayments, which depend on interest. The "
                   "model iterates to convergence and reports whether it succeeded.")

    return report
