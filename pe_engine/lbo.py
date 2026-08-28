"""The LBO engine: what a sponsor can pay, and what they earn for it.

Where the DCF asks what the business is worth, this asks a different question entirely: given
a price, a capital structure and an exit, what return does the equity make? The two are not
competing estimates of the same quantity, and the comparison between them is the point of the
project rather than a tie-break.

**The circular reference is real and is handled rather than avoided.** Interest expense depends
on the debt balance; the debt balance depends on how much cash is swept to repay it; the cash
available depends on interest expense. Under a beginning-balance convention the loop breaks
naturally, because interest is fixed before any repayment happens. Under an average-balance
convention it does not, and the model iterates to convergence. Spreadsheets solve this by
enabling iterative calculation and hoping; this module iterates explicitly, caps the
iterations, and reports whether it converged.

**Sign conventions are enforced, not trusted.** Debt is positive. Repayments reduce it. Cash
available for debt service is after cash interest and cash taxes. A tranche can never repay
more than it owes, and the revolver is the only facility that may be redrawn.

**Cash taxes are computed on levered profit, unlike the DCF.** This is not an inconsistency
between the two models, it is the substantive difference between them: the sponsor genuinely
does deduct interest and genuinely does pay less tax, and that tax shield is a real source of
LBO returns. The DCF captures the same benefit through the discount rate instead. Counting it
in both places, or neither, would be the error.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pe_engine.assumptions import DealAssumptions, DebtTranche
from pe_engine.forecast import Forecast

MAX_ITERATIONS = 50
CONVERGENCE_TOLERANCE = 1e-7
BALANCE_TOLERANCE = 0.01  # millions; sources and uses must agree to within a rounding cent


@dataclass(frozen=True)
class SourcesAndUses:
    """The transaction funding table. Sources must equal uses, and this checks that they do."""

    entry_ebitda: float
    entry_multiple: float
    entry_enterprise_value: float

    purchase_equity: float
    refinanced_debt: float
    transaction_fees: float
    financing_fees: float
    minimum_cash: float

    debt_raised: dict[str, float]
    sponsor_equity: float
    existing_cash_used: float

    @property
    def total_uses(self) -> float:
        return (self.purchase_equity + self.refinanced_debt + self.transaction_fees
                + self.financing_fees + self.minimum_cash)

    @property
    def total_debt(self) -> float:
        return sum(self.debt_raised.values())

    @property
    def total_sources(self) -> float:
        return self.total_debt + self.sponsor_equity + self.existing_cash_used

    @property
    def balances(self) -> bool:
        return abs(self.total_sources - self.total_uses) < BALANCE_TOLERANCE

    @property
    def gap(self) -> float:
        return self.total_sources - self.total_uses

    def calculation_trail(self) -> list[str]:
        lines = [
            f"Entry enterprise value = {self.entry_ebitda:,.1f} EBITDA "
            f"x {self.entry_multiple:.2f}x = {self.entry_enterprise_value:,.1f}",
            "",
            "USES",
            f"  Purchase of equity{' ' * 12}{self.purchase_equity:>12,.1f}",
            f"  Refinance existing debt{' ' * 7}{self.refinanced_debt:>12,.1f}",
            f"  Transaction fees{' ' * 14}{self.transaction_fees:>12,.1f}",
            f"  Financing fees{' ' * 16}{self.financing_fees:>12,.1f}",
            f"  Minimum cash to balance sheet{' ' * 1}{self.minimum_cash:>12,.1f}",
            f"  Total uses{' ' * 20}{self.total_uses:>12,.1f}",
            "",
            "SOURCES",
        ]
        for name, amount in self.debt_raised.items():
            lines.append(f"  {name:<28}{amount:>12,.1f}")
        lines += [
            f"  Existing cash on balance sheet{self.existing_cash_used:>12,.1f}",
            f"  Sponsor equity{' ' * 16}{self.sponsor_equity:>12,.1f}",
            f"  Total sources{' ' * 17}{self.total_sources:>12,.1f}",
            "",
            f"Sources less uses = {self.gap:,.4f}"
            + ("  (balanced)" if self.balances else "  (DOES NOT BALANCE)"),
        ]
        return lines


@dataclass(frozen=True)
class LBOResult:
    """A complete buyout: funding, the debt schedule, the exit and the returns."""

    sources_uses: SourcesAndUses
    schedule: pd.DataFrame
    holding_period: int

    exit_ebitda: float
    exit_multiple: float
    exit_enterprise_value: float
    exit_net_debt: float
    exit_equity: float

    sponsor_equity: float
    moic: float
    irr: float

    converged: bool = True
    iterations: int = 1
    currency: str = "USD"
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def entry_leverage(self) -> float:
        entry = self.sources_uses
        return entry.total_debt / entry.entry_ebitda if entry.entry_ebitda > 0 else float("nan")

    @property
    def exit_leverage(self) -> float:
        return self.exit_net_debt / self.exit_ebitda if self.exit_ebitda > 0 else float("nan")

    def calculation_trail(self) -> list[str]:
        return self.sources_uses.calculation_trail() + [
            "",
            f"EXIT after {self.holding_period} years",
            f"  Exit EBITDA{' ' * 19}{self.exit_ebitda:>12,.1f}",
            f"  x exit multiple{' ' * 15}{self.exit_multiple:>12,.2f}x",
            f"  Exit enterprise value{' ' * 9}{self.exit_enterprise_value:>12,.1f}",
            f"  less net debt at exit{' ' * 9}{-self.exit_net_debt:>12,.1f}",
            f"  Sponsor exit equity{' ' * 11}{self.exit_equity:>12,.1f}",
            "",
            f"MOIC = {self.exit_equity:,.1f} / {self.sponsor_equity:,.1f} = {self.moic:.2f}x",
            f"IRR  = {self.moic:.2f}x over {self.holding_period} years = {self.irr:.2%}",
        ]


def build_sources_and_uses(assumptions: DealAssumptions) -> SourcesAndUses:
    """Size the transaction: what must be paid for, and what pays for it.

    Sponsor equity is the plug. Everything else is either determined by the deal terms or
    chosen by the user, and the sponsor writes a cheque for whatever remains. That is how a
    real funding table works and it is why sources always balance to uses by construction: the
    balance check exists to catch an arithmetic error, not to catch a user mistake.
    """
    txn = assumptions.transaction
    company = assumptions.company
    entry_ebitda = assumptions.base_ebitda

    if entry_ebitda <= 0:
        raise ValueError(
            "Entry EBITDA must be positive. A buyout is priced as a multiple of EBITDA, so a "
            "loss-making business cannot be valued this way."
        )

    entry_ev = entry_ebitda * txn.entry_ebitda_multiple

    # The sponsor buys the equity, which is enterprise value less what the lenders are owed
    # plus the cash already inside the business.
    purchase_equity = entry_ev - company.debt + company.cash
    refinanced = company.debt if txn.refinance_existing_debt else 0.0
    transaction_fees = entry_ev * txn.transaction_fee_pct

    debt_raised = {t.name: entry_ebitda * t.turns_of_ebitda
                   for t in txn.tranches if not t.is_revolver}
    financing_fees = sum(debt_raised.values()) * txn.financing_fee_pct

    # Cash already on the balance sheet funds part of the purchase, except the minimum the
    # business must retain to operate.
    existing_cash_used = max(company.cash - txn.minimum_cash, 0.0) if not \
        txn.refinance_existing_debt else company.cash

    uses = (purchase_equity + refinanced + transaction_fees + financing_fees + txn.minimum_cash)
    sponsor_equity = uses - sum(debt_raised.values()) - existing_cash_used

    return SourcesAndUses(
        entry_ebitda=entry_ebitda, entry_multiple=txn.entry_ebitda_multiple,
        entry_enterprise_value=entry_ev, purchase_equity=purchase_equity,
        refinanced_debt=refinanced, transaction_fees=transaction_fees,
        financing_fees=financing_fees, minimum_cash=txn.minimum_cash,
        debt_raised=debt_raised, sponsor_equity=sponsor_equity,
        existing_cash_used=existing_cash_used,
    )


def _available_for_debt(row: pd.Series, tax_rate: float, interest: float,
                        da: float) -> tuple[float, float]:
    """Cash generated in a year, after everything that must be paid before lenders.

    Returns the cash available for debt service and the cash tax charge. Taxes here are levered:
    interest is deducted before tax, which is the sponsor's genuine benefit from the capital
    structure and a real source of LBO returns.
    """
    ebit = float(row["ebit"])
    ebt = ebit - interest
    cash_tax = max(ebt, 0.0) * tax_rate
    # From EBITDA rather than EBIT, since D&A is non-cash and must be added back.
    available = (float(row["ebitda"]) - cash_tax - float(row["capex"])
                 - float(row["change_in_nwc"]) - interest)
    return available, cash_tax


def _run_schedule(assumptions: DealAssumptions, forecast: Forecast,
                  sources: SourcesAndUses) -> tuple[pd.DataFrame, bool, int]:
    """Roll the debt forward year by year, resolving the interest circularity if present.

    Under a beginning-balance convention interest is known before any repayment, so one pass
    suffices. Under an average-balance convention the answer depends on itself and the schedule
    is recomputed until the interest charge stops moving.
    """
    txn = assumptions.transaction
    tranches = [t for t in txn.tranches if not t.is_revolver]
    revolver = next((t for t in txn.tranches if t.is_revolver), None)
    horizon = min(txn.holding_period, forecast.horizon)

    # Interest guesses, refined on each pass when the basis is circular.
    interest_guess = np.zeros(horizon)
    converged, iterations = True, 1

    for iteration in range(1, MAX_ITERATIONS + 1):
        balances = {t.name: sources.debt_raised.get(t.name, 0.0) for t in tranches}
        original = dict(balances)
        revolver_balance = 0.0
        cash = txn.minimum_cash
        rows = []
        computed_interest = np.zeros(horizon)

        for i in range(horizon):
            row = forecast.at(i + 1)
            opening = dict(balances)
            opening_revolver = revolver_balance

            # --- interest -------------------------------------------------------------------
            if txn.interest_basis == "beginning":
                interest = sum(opening[t.name] * t.interest_rate for t in tranches)
                if revolver is not None:
                    interest += opening_revolver * revolver.interest_rate
            else:
                interest = float(interest_guess[i])

            available, cash_tax = _available_for_debt(
                row, assumptions.operating.tax_rate, interest, float(row["da"]))

            # --- mandatory amortisation -------------------------------------------------------
            mandatory_total = 0.0
            for t in tranches:
                due = min(original[t.name] * t.amortisation_pct, balances[t.name])
                balances[t.name] -= due
                mandatory_total += due

            remaining = available - mandatory_total

            # --- revolver draw if the year does not cover its obligations -----------------------
            if remaining < 0:
                shortfall = -remaining
                if revolver is not None:
                    capacity = max(
                        revolver.turns_of_ebitda * sources.entry_ebitda - revolver_balance, 0.0)
                    drawn = min(shortfall, capacity)
                    revolver_balance += drawn
                    shortfall -= drawn
                cash = max(cash - shortfall, 0.0)
                remaining = 0.0

            # --- optional repayment out of excess cash ------------------------------------------
            sweep_available = remaining
            if txn.sweep_policy == "mandatory_only":
                sweep_available = 0.0
            elif txn.sweep_policy == "partial_sweep":
                sweep_available = remaining * txn.sweep_pct

            optional_total = 0.0
            if revolver is not None and revolver_balance > 0 and sweep_available > 0:
                repay = min(revolver_balance, sweep_available)
                revolver_balance -= repay
                sweep_available -= repay
                optional_total += repay

            for t in sorted(tranches, key=lambda x: x.cash_sweep_priority):
                if sweep_available <= 0:
                    break
                repay = min(balances[t.name], sweep_available)
                balances[t.name] -= repay
                sweep_available -= repay
                optional_total += repay

            cash += remaining - optional_total
            computed_interest[i] = (
                sum((opening[t.name] + balances[t.name]) / 2 * t.interest_rate for t in tranches)
                + ((opening_revolver + revolver_balance) / 2 * revolver.interest_rate
                   if revolver is not None else 0.0)
            )

            rows.append({
                "year": i + 1,
                "ebitda": float(row["ebitda"]),
                "cash_taxes": cash_tax,
                "capex": float(row["capex"]),
                "change_in_nwc": float(row["change_in_nwc"]),
                "cash_interest": interest,
                "cash_available": available,
                "mandatory_repayment": mandatory_total,
                "optional_repayment": optional_total,
                "opening_debt": sum(opening.values()) + opening_revolver,
                "closing_debt": sum(balances.values()) + revolver_balance,
                "revolver": revolver_balance,
                "cash_balance": cash,
                "net_debt": sum(balances.values()) + revolver_balance - cash,
                "leverage": ((sum(balances.values()) + revolver_balance - cash)
                             / float(row["ebitda"]) if row["ebitda"] else float("nan")),
            })

        if txn.interest_basis == "beginning":
            break

        shift = float(np.max(np.abs(computed_interest - interest_guess)))
        interest_guess = computed_interest
        iterations = iteration
        if shift < CONVERGENCE_TOLERANCE:
            break
    else:
        converged = False

    return pd.DataFrame(rows), converged, iterations


def irr(cash_flows: list[float]) -> float:
    """Annualised internal rate of return, by bisection on net present value.

    Bisection rather than a library call so the failure modes are explicit. An IRR requires at
    least one sign change: a stream of only outflows or only inflows has no rate at which the
    present value is zero, and returning a number there would be inventing one.
    """
    flows = np.asarray(cash_flows, dtype=float)
    if len(flows) < 2:
        return float("nan")
    if not (np.any(flows > 0) and np.any(flows < 0)):
        return float("nan")

    def npv(rate: float) -> float:
        periods = np.arange(len(flows), dtype=float)
        return float(np.sum(flows / (1.0 + rate) ** periods))

    low, high = -0.9999, 10.0
    npv_low, npv_high = npv(low), npv(high)
    if npv_low * npv_high > 0:
        return float("nan")

    for _ in range(200):
        mid = (low + high) / 2
        value = npv(mid)
        if abs(value) < 1e-9:
            return mid
        if value * npv_low > 0:
            low, npv_low = mid, value
        else:
            high = mid
    return (low + high) / 2


def run(assumptions: DealAssumptions, forecast: Forecast,
        entry_multiple_override: float | None = None,
        exit_multiple_override: float | None = None) -> LBOResult:
    """Model the buyout end to end and compute sponsor returns.

    The overrides exist for the sensitivity engine and the maximum-price solver, both of which
    sweep multiples without rebuilding the assumption tree for every evaluation.
    """
    working = assumptions
    if entry_multiple_override is not None:
        working = working.vary(**{"transaction.entry_ebitda_multiple": entry_multiple_override})
    if exit_multiple_override is not None:
        working = working.vary(**{"transaction.exit_ebitda_multiple": exit_multiple_override})

    txn = working.transaction
    warnings: list[str] = []
    notes: list[str] = []

    sources = build_sources_and_uses(working)
    if not sources.balances:
        warnings.append(
            f"Sources and uses do not balance: the gap is {sources.gap:,.2f}. This is an "
            "arithmetic error in the model rather than a user input problem, since sponsor "
            "equity is the balancing item."
        )

    schedule, converged, iterations = _run_schedule(working, forecast, sources)
    if not converged:
        warnings.append(
            f"The interest calculation did not converge within {MAX_ITERATIONS} iterations. "
            "Average-balance interest is genuinely circular, and a non-converging schedule "
            "usually means the capital structure cannot service itself at these assumptions."
        )

    holding = min(txn.holding_period, forecast.horizon)
    final = schedule.iloc[holding - 1]
    exit_ebitda = forecast.ebitda_at(holding)
    exit_ev = exit_ebitda * txn.exit_ebitda_multiple
    exit_net_debt = float(final["net_debt"])
    exit_equity = exit_ev - exit_net_debt

    sponsor_equity = sources.sponsor_equity
    moic = exit_equity / sponsor_equity if sponsor_equity > 0 else float("nan")

    flows = [-sponsor_equity] + [0.0] * (holding - 1) + [exit_equity]
    rate = irr(flows)

    if exit_equity <= 0:
        warnings.append(
            "Sponsor equity is wiped out at exit: net debt exceeds the exit enterprise value. "
            "The return is a total loss rather than a negative percentage, and the IRR is "
            "reported as -100%."
        )
    if sponsor_equity <= 0:
        warnings.append(
            "Sponsor equity is zero or negative, meaning the debt raised exceeds the funds "
            "required. MOIC and IRR are undefined because there is no invested capital to "
            "return."
        )

    peak_leverage = float(schedule["leverage"].max())
    if peak_leverage > 6.0:
        warnings.append(
            f"Net leverage peaks at {peak_leverage:.1f}x EBITDA. Above roughly 6x, lenders "
            "typically require covenants that this model does not represent, so the schedule "
            "may be more permissive than a real credit agreement would allow."
        )
    if (schedule["cash_available"] < 0).any():
        years = list(schedule.loc[schedule["cash_available"] < 0, "year"])
        warnings.append(
            f"Cash available for debt service is negative in year(s) {years}. The business "
            "does not cover its own interest and capital spending in those years, which is "
            "funded here by the revolver or by drawing down cash."
        )

    notes.append(
        "Cash taxes are computed on profit after interest, unlike the DCF's unlevered tax. "
        "That is the substantive difference between the two models rather than an "
        "inconsistency: the sponsor genuinely deducts interest, and that shield is a real "
        "source of returns. The DCF captures the same benefit inside WACC instead."
    )
    notes.append(
        f"Interest is charged on the {txn.interest_basis} balance"
        + (f", resolved iteratively in {iterations} passes because an average balance depends "
           "on the repayments it determines." if txn.interest_basis == "average"
           else ", which avoids the circular reference entirely.")
    )

    return LBOResult(
        sources_uses=sources, schedule=schedule, holding_period=holding,
        exit_ebitda=exit_ebitda, exit_multiple=txn.exit_ebitda_multiple,
        exit_enterprise_value=exit_ev, exit_net_debt=exit_net_debt,
        exit_equity=exit_equity, sponsor_equity=sponsor_equity,
        moic=moic, irr=rate if exit_equity > 0 else -1.0,
        converged=converged, iterations=iterations, currency=working.company.currency,
        warnings=tuple(warnings), notes=tuple(notes),
    )


def maximum_entry_multiple(assumptions: DealAssumptions, forecast: Forecast,
                           target_irr: float | None = None,
                           low: float = 1.0, high: float = 30.0) -> dict:
    """The highest entry multiple that still clears a target return.

    This is the question a sponsor actually asks, and it runs the model backwards: rather than
    computing the return from a price, it solves for the price that produces a required return.

    Bisection relies on IRR falling as the entry multiple rises, which holds because paying
    more for the same business and the same exit can only reduce the return. Where the target
    is unreachable even at the lowest searchable multiple, the function says so rather than
    returning the boundary as though it were a solution.

    **The lower bound has to be found, not assumed.** Below a certain entry price the debt
    raised exceeds the total funding requirement, so sponsor equity goes negative and there is
    no invested capital to earn a return on. That is an infeasible structure rather than an
    attractive one, and a solver that treated the resulting undefined IRR as "target not
    achievable" would report the opposite of the truth: it would reject deals that work at a
    sensible price because they break at an absurd one. So the search starts by walking up from
    `low` until the structure is fundable.
    """
    target = assumptions.target_irr if target_irr is None else target_irr

    def irr_at(multiple: float) -> float:
        try:
            result = run(assumptions, forecast, entry_multiple_override=multiple)
            if result.sponsor_equity <= 0:
                return float("nan")
            return result.irr
        except Exception:  # noqa: BLE001 - an unsolvable structure is not a crash
            return float("nan")

    step = max((high - low) / 200.0, 0.01)
    probe = low
    irr_low = irr_at(probe)
    while np.isnan(irr_low) and probe < high:
        probe += step
        irr_low = irr_at(probe)
    low = probe

    if np.isnan(irr_low):
        return {
            "target_irr": target, "max_entry_multiple": float("nan"),
            "max_entry_enterprise_value": float("nan"), "achievable": False,
            "reading": (
                f"No entry multiple between {low:.1f}x and {high:.1f}x produces a fundable "
                "structure with positive sponsor equity. The debt raised exceeds the purchase "
                "price across the whole range, so leverage needs reducing before a price can "
                "be solved for."
            ),
        }

    irr_high = irr_at(high)
    if irr_low < target:
        return {
            "target_irr": target, "max_entry_multiple": float("nan"),
            "max_entry_enterprise_value": float("nan"), "achievable": False,
            "reading": (
                f"A {target:.0%} IRR is not achievable even at the lowest fundable entry "
                f"multiple of {low:.2f}x, where the return is {irr_low:.1%}. The operating "
                "case and capital structure cannot produce this return regardless of price, "
                "so the constraint is the business rather than the valuation."
            ),
        }
    if not np.isnan(irr_high) and irr_high > target:
        return {
            "target_irr": target, "max_entry_multiple": high,
            "max_entry_enterprise_value": high * assumptions.base_ebitda, "achievable": True,
            "reading": (
                f"Even at {high:.1f}x the IRR exceeds {target:.0%}, so the search range is the "
                "binding constraint rather than the deal economics. Raise the upper bound to "
                "find the true maximum."
            ),
        }

    for _ in range(80):
        mid = (low + high) / 2
        value = irr_at(mid)
        if np.isnan(value):
            high = mid
            continue
        if value > target:
            low = mid
        else:
            high = mid

    solved = (low + high) / 2
    entry_ev = solved * assumptions.base_ebitda
    actual = assumptions.transaction.entry_ebitda_multiple
    margin = (solved - actual) / actual if actual > 0 else float("nan")

    return {
        "target_irr": target,
        "max_entry_multiple": solved,
        "max_entry_enterprise_value": entry_ev,
        "actual_entry_multiple": actual,
        "margin_of_safety": margin,
        "achievable": True,
        "reading": (
            f"At a {target:.0%} target return the sponsor can pay up to {solved:.2f}x EBITDA, "
            f"an enterprise value of {entry_ev:,.0f}. The proposed entry is {actual:.2f}x, "
            f"which leaves {margin:+.1%} of headroom against the maximum."
        ),
    }
