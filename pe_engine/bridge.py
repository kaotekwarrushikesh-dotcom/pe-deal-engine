"""The value creation bridge: where sponsor returns actually came from.

A sponsor turning 100 of equity into 250 has created 150, and the interesting question is which
of three levers did it. Growing the business, selling it at a higher multiple than was paid, and
using the company's own cash flow to repay debt are genuinely different achievements, and only
the first is unambiguously operational skill.

**The decomposition has four terms, not three, and the fourth is the one usually dropped.**

    Value created = Exit equity - Entry equity

    EBITDA growth        (Exit EBITDA - Entry EBITDA) x Entry multiple
    Multiple change      (Exit multiple - Entry multiple) x Entry EBITDA
    Interaction          (Exit EBITDA - Entry EBITDA) x (Exit multiple - Entry multiple)
    Debt paydown         Entry net debt - Exit net debt

Those four sum *exactly* to the value created, which is checked with an assertion rather than
asserted in prose. Drop the interaction term, as many published bridges do, and the components
no longer reconcile: the gap is silently absorbed into whichever bucket the author allocated it
to, usually EBITDA growth, which flatters the operational story.

The interaction term is real and has a plain meaning: it is the extra value from selling *more*
EBITDA at a *higher* multiple, which belongs to neither lever alone. Where a deal grows EBITDA
and the multiple expands, this term is positive and often material. This module reports it as
its own line, and offers an allocated view for presentation that says explicitly that it is
splitting a joint effect by convention rather than by measurement.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pe_engine.lbo import LBOResult

RECONCILIATION_TOLERANCE = 0.01  # millions


@dataclass(frozen=True)
class ValueBridge:
    """The attribution of sponsor value creation to its sources."""

    entry_equity: float
    exit_equity: float

    entry_ebitda: float
    exit_ebitda: float
    entry_multiple: float
    exit_multiple: float
    entry_net_debt: float
    exit_net_debt: float

    ebitda_growth: float
    multiple_change: float
    interaction: float
    debt_paydown: float

    currency: str = "USD"
    notes: tuple[str, ...] = ()

    @property
    def total_created(self) -> float:
        return self.exit_equity - self.entry_equity

    @property
    def components(self) -> dict[str, float]:
        return {
            "EBITDA growth": self.ebitda_growth,
            "Multiple change": self.multiple_change,
            "Growth and multiple together": self.interaction,
            "Debt paydown": self.debt_paydown,
        }

    @property
    def contribution_pct(self) -> dict[str, float]:
        """Each component as a share of total value created.

        Shares can exceed 100% or go negative, and that is not an error: a deal where the
        multiple contracted has a negative multiple contribution offset by more than 100% from
        the other levers. Normalising those away would hide the most important thing about the
        deal.
        """
        total = self.total_created
        if abs(total) < 1e-9:
            return {k: float("nan") for k in self.components}
        return {k: v / total for k, v in self.components.items()}

    @property
    def allocated_contribution_pct(self) -> dict[str, float]:
        """A three-way view, splitting the interaction between growth and multiple.

        For presentation only. The split is pro-rata to the two effects that produced it, which
        is a convention rather than a measurement: the joint term genuinely belongs to both and
        no allocation of it is more correct than another.
        """
        total = self.total_created
        if abs(total) < 1e-9:
            return {}

        growth, multiple = self.ebitda_growth, self.multiple_change
        magnitude = abs(growth) + abs(multiple)
        if magnitude < 1e-9:
            share_growth = share_multiple = 0.5
        else:
            share_growth = abs(growth) / magnitude
            share_multiple = abs(multiple) / magnitude

        return {
            "EBITDA growth": (growth + self.interaction * share_growth) / total,
            "Multiple change": (multiple + self.interaction * share_multiple) / total,
            "Debt paydown": self.debt_paydown / total,
        }

    def reconciles(self) -> bool:
        return abs(sum(self.components.values()) - self.total_created) < RECONCILIATION_TOLERANCE

    def frame(self) -> pd.DataFrame:
        rows = [{"component": "Entry sponsor equity", "value": self.entry_equity,
                 "share": float("nan")}]
        for name, value in self.components.items():
            rows.append({"component": name, "value": value,
                         "share": self.contribution_pct[name]})
        rows.append({"component": "Exit sponsor equity", "value": self.exit_equity,
                     "share": float("nan")})
        return pd.DataFrame(rows)

    def calculation_trail(self) -> list[str]:
        return [
            f"Entry: {self.entry_ebitda:,.1f} EBITDA x {self.entry_multiple:.2f}x "
            f"= {self.entry_ebitda * self.entry_multiple:,.1f} enterprise value",
            f"       less net debt {self.entry_net_debt:,.1f} "
            f"= {self.entry_equity:,.1f} sponsor equity",
            "",
            f"Exit:  {self.exit_ebitda:,.1f} EBITDA x {self.exit_multiple:.2f}x "
            f"= {self.exit_ebitda * self.exit_multiple:,.1f} enterprise value",
            f"       less net debt {self.exit_net_debt:,.1f} "
            f"= {self.exit_equity:,.1f} sponsor equity",
            "",
            f"Value created = {self.total_created:,.1f}",
            "",
            f"  EBITDA growth   ({self.exit_ebitda:,.1f} - {self.entry_ebitda:,.1f}) "
            f"x {self.entry_multiple:.2f}x = {self.ebitda_growth:>12,.1f}",
            f"  Multiple change ({self.exit_multiple:.2f}x - {self.entry_multiple:.2f}x) "
            f"x {self.entry_ebitda:,.1f} = {self.multiple_change:>12,.1f}",
            f"  Interaction     (EBITDA change) x (multiple change) "
            f"= {self.interaction:>12,.1f}",
            f"  Debt paydown    ({self.entry_net_debt:,.1f} - {self.exit_net_debt:,.1f}) "
            f"= {self.debt_paydown:>12,.1f}",
            f"  {'-' * 58}",
            f"  Sum{' ' * 52}= {sum(self.components.values()):>12,.1f}",
            "",
            "The four components sum exactly to the value created. Dropping the interaction "
            "term, as many bridges do, breaks that identity.",
        ]


def build(result: LBOResult) -> ValueBridge:
    """Decompose sponsor value creation into its four exact sources."""
    sources = result.sources_uses

    entry_ebitda = sources.entry_ebitda
    entry_multiple = sources.entry_multiple
    entry_ev = sources.entry_enterprise_value
    entry_net_debt = entry_ev - sources.sponsor_equity

    exit_ebitda = result.exit_ebitda
    exit_multiple = result.exit_multiple
    exit_net_debt = result.exit_net_debt

    delta_ebitda = exit_ebitda - entry_ebitda
    delta_multiple = exit_multiple - entry_multiple

    bridge = ValueBridge(
        entry_equity=sources.sponsor_equity,
        exit_equity=result.exit_equity,
        entry_ebitda=entry_ebitda, exit_ebitda=exit_ebitda,
        entry_multiple=entry_multiple, exit_multiple=exit_multiple,
        entry_net_debt=entry_net_debt, exit_net_debt=exit_net_debt,
        ebitda_growth=delta_ebitda * entry_multiple,
        multiple_change=delta_multiple * entry_ebitda,
        interaction=delta_ebitda * delta_multiple,
        debt_paydown=entry_net_debt - exit_net_debt,
        currency=result.currency,
    )

    # The decomposition is an algebraic identity, so a mismatch is a coding error rather than a
    # modelling judgement, and it should fail loudly instead of producing a plausible chart.
    assert bridge.reconciles(), (
        f"value bridge does not reconcile: components sum to "
        f"{sum(bridge.components.values()):,.4f} against value created of "
        f"{bridge.total_created:,.4f}"
    )

    notes = [
        "The four components sum exactly to the value created, which is asserted at runtime. "
        "A three-way bridge that omits the interaction term does not reconcile, and the "
        "difference is usually absorbed into EBITDA growth, flattering the operational story.",
    ]
    if bridge.interaction > abs(bridge.total_created) * 0.10:
        notes.append(
            f"The interaction term is {bridge.contribution_pct['Growth and multiple together']:.0%} "
            "of value created, which is material. It comes from selling more EBITDA at a higher "
            "multiple, and belongs to neither lever alone."
        )
    if bridge.multiple_change < 0:
        notes.append(
            "The exit multiple is below the entry multiple, so multiple contraction destroyed "
            "value and the other levers had to work harder to offset it. A deal that relies on "
            "multiple expansion is betting on the market rather than on the business."
        )
    if bridge.debt_paydown > abs(bridge.total_created) * 0.50:
        notes.append(
            "More than half of value creation came from debt paydown, which is leverage and "
            "cash generation rather than operational improvement. It is a legitimate source of "
            "return and a fragile one, since it depends on the business continuing to convert "
            "EBITDA into cash."
        )

    return ValueBridge(**{**bridge.__dict__, "notes": tuple(notes)})
