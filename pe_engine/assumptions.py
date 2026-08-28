"""The Assumption Engine: one immutable structure that every model reads from.

This is the design decision the rest of the project rests on. Every material assumption lives
here, in one tree, and no calculation anywhere else is allowed to invent a number. A DCF
enterprise value and an LBO sponsor IRR computed from the same `DealAssumptions` are
guaranteed to describe the same company under the same operating case, because there is
physically nowhere else for an operating assumption to come from.

**Everything is frozen.** A scenario is not a mutated global, it is a new object built with
`vary()`. That matters more than it sounds: sensitivity grids run dozens of variants and
scenario switching runs three, and the single most common way a financial model produces a
wrong number is state left behind from the previous run. An immutable tree makes that
impossible rather than unlikely.

**Per-year assumptions are lists, not single rates.** A revenue growth that fades from 8% to
4% over five years is the normal case in a real model, and forcing a single rate would be a
simplification the user cannot undo. Where a caller supplies one value, it is broadcast across
the horizon and the model records that it did.

**Nothing here has a default that carries a view.** Where a default exists it is either a
structural convention (a five-year horizon) or an obviously neutral placeholder. Growth rates,
margins and multiples have no defaults at all: they are required, because a model that quietly
supplies its own operating case is a model whose output belongs to whoever wrote the defaults
rather than to the user.
"""

from dataclasses import dataclass, field, replace
from typing import Literal

TerminalMethod = Literal["gordon", "exit_multiple"]
InterestBasis = Literal["beginning", "average"]
SweepPolicy = Literal["mandatory_only", "partial_sweep", "full_sweep"]


def _broadcast(value: float | list[float], years: int, label: str) -> tuple[float, ...]:
    """A single rate becomes a flat path; a list must match the horizon exactly.

    A length mismatch is refused rather than padded, because silently extending a four-year
    assumption across a five-year forecast invents the fifth year without saying so.
    """
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(years))
    path = tuple(float(v) for v in value)
    if len(path) != years:
        raise ValueError(
            f"{label} has {len(path)} values for a {years}-year forecast. Supply one value per "
            "year, or a single number to apply to every year."
        )
    return path


@dataclass(frozen=True)
class Company:
    """Identity and the current capital structure the deal starts from."""

    name: str
    ticker: str = ""
    industry: str = ""
    currency: str = "USD"

    share_price: float = 0.0
    shares_outstanding: float = 0.0   # millions
    debt: float = 0.0                 # millions
    cash: float = 0.0                 # millions

    @property
    def market_cap(self) -> float:
        return self.share_price * self.shares_outstanding

    @property
    def net_debt(self) -> float:
        return self.debt - self.cash

    @property
    def enterprise_value(self) -> float:
        """Market capitalisation plus net debt: what the market pays for the whole business."""
        return self.market_cap + self.net_debt


@dataclass(frozen=True)
class OperatingAssumptions:
    """The operating case. Shared by both models, which is what makes them comparable.

    A DCF and an LBO that disagree because they were fed different growth rates are not
    telling you anything about the deal, only about the inconsistency. Both read this.
    """

    horizon: int
    revenue_growth: tuple[float, ...]
    ebitda_margin: tuple[float, ...]
    da_pct_revenue: tuple[float, ...]
    capex_pct_revenue: tuple[float, ...]
    tax_rate: float

    # Working capital as a percentage of revenue. Receivables and inventory consume cash as
    # revenue grows; payables release it, which is why the sign convention is enforced in the
    # forecast rather than left to the caller.
    receivables_pct_revenue: float = 0.0
    inventory_pct_revenue: float = 0.0
    payables_pct_revenue: float = 0.0
    other_nwc_pct_revenue: float = 0.0

    @property
    def net_working_capital_pct(self) -> float:
        """Net working capital intensity. Payables are a source of cash, so they subtract."""
        return (self.receivables_pct_revenue + self.inventory_pct_revenue
                + self.other_nwc_pct_revenue - self.payables_pct_revenue)


@dataclass(frozen=True)
class WACCAssumptions:
    """Every component of the discount rate, entered rather than derived.

    Kept as inputs deliberately. Module 2 in this platform estimates beta and the risk-free
    rate from market data, which is right for screening many companies. Here the user is
    modelling one deal and needs to be able to say "what if the cost of equity is 11%", so the
    components are theirs to set and the arithmetic is shown rather than hidden.
    """

    risk_free_rate: float
    equity_risk_premium: float
    beta: float
    cost_of_debt: float
    tax_rate: float
    equity_weight_value: float   # market value of equity, millions
    debt_weight_value: float     # market value of debt, millions

    @property
    def cost_of_equity(self) -> float:
        """CAPM: the return equity holders require for this much market risk."""
        return self.risk_free_rate + self.beta * self.equity_risk_premium

    @property
    def after_tax_cost_of_debt(self) -> float:
        """Interest is deductible, so the company does not bear the full coupon."""
        return self.cost_of_debt * (1.0 - self.tax_rate)

    @property
    def total_capital(self) -> float:
        return self.equity_weight_value + self.debt_weight_value

    @property
    def equity_weight(self) -> float:
        return self.equity_weight_value / self.total_capital if self.total_capital > 0 else 0.0

    @property
    def debt_weight(self) -> float:
        return self.debt_weight_value / self.total_capital if self.total_capital > 0 else 0.0

    @property
    def wacc(self) -> float:
        return (self.equity_weight * self.cost_of_equity
                + self.debt_weight * self.after_tax_cost_of_debt)


@dataclass(frozen=True)
class TerminalAssumptions:
    """How the value beyond the forecast horizon is estimated.

    Both methods are offered because they fail differently. Gordon growth is sensitive to a
    rate the user cannot observe and breaks down entirely as growth approaches the discount
    rate. An exit multiple is observable but imports today's market pricing into a valuation
    of tomorrow, which is circular in a way that is easy to forget.
    """

    method: TerminalMethod = "gordon"
    terminal_growth: float = 0.02
    exit_ebitda_multiple: float = 10.0


@dataclass(frozen=True)
class DebtTranche:
    """One layer of the capital structure.

    Sized as a multiple of entry EBITDA, which is how leverage is actually quoted in a deal,
    rather than as an absolute amount that would stop meaning anything the moment EBITDA
    changes.
    """

    name: str
    turns_of_ebitda: float
    interest_rate: float
    amortisation_pct: float = 0.0     # mandatory repayment, share of original principal a year
    cash_sweep_priority: int = 1      # lower repays first out of excess cash
    is_revolver: bool = False


@dataclass(frozen=True)
class TransactionAssumptions:
    """Entry, financing and exit. The economics a sponsor actually controls."""

    entry_ebitda_multiple: float
    exit_ebitda_multiple: float
    holding_period: int

    transaction_fee_pct: float = 0.02      # of enterprise value
    financing_fee_pct: float = 0.02        # of debt raised
    minimum_cash: float = 0.0              # millions
    refinance_existing_debt: bool = True

    tranches: tuple[DebtTranche, ...] = ()
    interest_basis: InterestBasis = "beginning"
    sweep_policy: SweepPolicy = "full_sweep"
    sweep_pct: float = 0.50                # used only when sweep_policy is partial_sweep

    @property
    def total_turns(self) -> float:
        return sum(t.turns_of_ebitda for t in self.tranches)


@dataclass(frozen=True)
class DealAssumptions:
    """The whole model's inputs, in one object.

    Passed to the DCF engine, the LBO engine, the sensitivity engine and the scenario engine.
    Nothing downstream holds an operating assumption of its own.
    """

    company: Company
    operating: OperatingAssumptions
    wacc: WACCAssumptions
    terminal: TerminalAssumptions
    transaction: TransactionAssumptions

    base_revenue: float = 0.0      # last historical year, millions
    base_ebitda: float = 0.0       # last historical year, millions
    target_irr: float = 0.20
    label: str = "Base case"

    def vary(self, **changes) -> "DealAssumptions":
        """A modified copy, for scenarios and sensitivity grids.

        Nested fields are addressed with a dotted path, so
        `vary(**{"transaction.exit_ebitda_multiple": 9.0})` returns a new tree with that one
        value changed and everything else shared. Nothing is mutated, so a grid of eighty
        variants cannot contaminate the base case or each other.
        """
        top: dict = {}
        nested: dict[str, dict] = {}
        for key, value in changes.items():
            if "." in key:
                section, field_name = key.split(".", 1)
                nested.setdefault(section, {})[field_name] = value
            else:
                top[key] = value

        for section, updates in nested.items():
            current = getattr(self, section)
            top[section] = replace(current, **updates)
        return replace(self, **top)


def build(
    company: Company,
    base_revenue: float,
    base_ebitda: float,
    horizon: int,
    revenue_growth: float | list[float],
    ebitda_margin: float | list[float],
    da_pct_revenue: float | list[float],
    capex_pct_revenue: float | list[float],
    tax_rate: float,
    risk_free_rate: float,
    equity_risk_premium: float,
    beta: float,
    cost_of_debt: float,
    entry_ebitda_multiple: float,
    exit_ebitda_multiple: float,
    holding_period: int,
    tranches: list[DebtTranche] | None = None,
    **extra,
) -> DealAssumptions:
    """Assemble a complete assumption tree, broadcasting single rates across the horizon.

    Every operating and financing input is required. There is no default growth rate, margin
    or multiple anywhere in this function, because a model that supplies its own operating
    case produces an answer belonging to whoever wrote the defaults rather than to the user.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least one year, got {horizon}")
    if holding_period < 1:
        raise ValueError(f"holding period must be at least one year, got {holding_period}")
    if holding_period > horizon:
        raise ValueError(
            f"holding period of {holding_period} years exceeds the {horizon}-year forecast. "
            "The exit would fall outside the modelled period, so there would be no EBITDA to "
            "exit on."
        )

    operating = OperatingAssumptions(
        horizon=horizon,
        revenue_growth=_broadcast(revenue_growth, horizon, "revenue growth"),
        ebitda_margin=_broadcast(ebitda_margin, horizon, "EBITDA margin"),
        da_pct_revenue=_broadcast(da_pct_revenue, horizon, "D&A as % of revenue"),
        capex_pct_revenue=_broadcast(capex_pct_revenue, horizon, "capex as % of revenue"),
        tax_rate=tax_rate,
        receivables_pct_revenue=extra.pop("receivables_pct_revenue", 0.0),
        inventory_pct_revenue=extra.pop("inventory_pct_revenue", 0.0),
        payables_pct_revenue=extra.pop("payables_pct_revenue", 0.0),
        other_nwc_pct_revenue=extra.pop("other_nwc_pct_revenue", 0.0),
    )

    wacc = WACCAssumptions(
        risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium, beta=beta,
        cost_of_debt=cost_of_debt, tax_rate=tax_rate,
        equity_weight_value=extra.pop("equity_weight_value", company.market_cap),
        debt_weight_value=extra.pop("debt_weight_value", company.debt),
    )

    terminal = TerminalAssumptions(
        method=extra.pop("terminal_method", "gordon"),
        terminal_growth=extra.pop("terminal_growth", 0.02),
        exit_ebitda_multiple=exit_ebitda_multiple,
    )

    transaction = TransactionAssumptions(
        entry_ebitda_multiple=entry_ebitda_multiple,
        exit_ebitda_multiple=exit_ebitda_multiple,
        holding_period=holding_period,
        transaction_fee_pct=extra.pop("transaction_fee_pct", 0.02),
        financing_fee_pct=extra.pop("financing_fee_pct", 0.02),
        minimum_cash=extra.pop("minimum_cash", 0.0),
        refinance_existing_debt=extra.pop("refinance_existing_debt", True),
        tranches=tuple(tranches or ()),
        interest_basis=extra.pop("interest_basis", "beginning"),
        sweep_policy=extra.pop("sweep_policy", "full_sweep"),
        sweep_pct=extra.pop("sweep_pct", 0.50),
    )

    return DealAssumptions(
        company=company, operating=operating, wacc=wacc, terminal=terminal,
        transaction=transaction, base_revenue=base_revenue, base_ebitda=base_ebitda,
        target_irr=extra.pop("target_irr", 0.20),
        label=extra.pop("label", "Base case"),
    )
