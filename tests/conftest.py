"""Shared fixtures: one plausible mid-market industrial buyout.

Deliberately a synthetic company rather than a real one. Every number in a test needs to be
stable, and a fixture that fetched live market data would make the suite fail for reasons that
have nothing to do with the code. The figures are chosen to be realistic enough that the
resulting IRR and MOIC land in ranges a practitioner would recognise, which is what makes a
test failure informative rather than merely red.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pe_engine.assumptions import Company, DebtTranche, build


@pytest.fixture
def company() -> Company:
    return Company(
        name="Meridian Industrial", ticker="MRDN", industry="Industrials", currency="USD",
        share_price=42.50, shares_outstanding=50.0, debt=300.0, cash=60.0,
    )


@pytest.fixture
def tranches() -> list[DebtTranche]:
    return [
        DebtTranche("Senior Term Loan", turns_of_ebitda=3.5, interest_rate=0.075,
                    amortisation_pct=0.05, cash_sweep_priority=1),
        DebtTranche("Mezzanine", turns_of_ebitda=1.5, interest_rate=0.115,
                    amortisation_pct=0.0, cash_sweep_priority=2),
        DebtTranche("Revolver", turns_of_ebitda=0.5, interest_rate=0.065, is_revolver=True),
    ]


@pytest.fixture
def deal(company, tranches):
    """The base case every test varies from."""
    return build(
        company=company, base_revenue=1200.0, base_ebitda=240.0, horizon=5,
        revenue_growth=[0.08, 0.07, 0.06, 0.05, 0.04],
        ebitda_margin=[0.205, 0.21, 0.215, 0.22, 0.22],
        da_pct_revenue=0.04, capex_pct_revenue=0.05, tax_rate=0.25,
        risk_free_rate=0.042, equity_risk_premium=0.055, beta=1.15, cost_of_debt=0.075,
        entry_ebitda_multiple=9.0, exit_ebitda_multiple=9.5, holding_period=5,
        tranches=tranches,
        receivables_pct_revenue=0.14, inventory_pct_revenue=0.11, payables_pct_revenue=0.09,
        terminal_growth=0.025, minimum_cash=25.0, target_irr=0.20,
    )
