"""Tests for the assumption engine, forecast and DCF.

The properties pinned here are the ones a financial model gets quietly wrong: a mutated
assumption leaking between scenario runs, a working capital sign flip, interest double-counted
through both the cash flow and the discount rate, and a terminal value produced by dividing by
something close to zero.
"""

import numpy as np
import pandas as pd
import pytest

from pe_engine import dcf as dcf_engine
from pe_engine import forecast as forecast_engine
from pe_engine import validation
from pe_engine.assumptions import Company, DebtTranche, build


# --- The assumption engine ---------------------------------------------------------------------

def test_a_single_rate_is_broadcast_across_the_horizon(deal):
    assert len(deal.operating.revenue_growth) == deal.operating.horizon
    assert deal.operating.capex_pct_revenue == (0.05,) * 5


def test_a_mismatched_assumption_path_is_refused_rather_than_padded(company, tranches):
    """Silently extending a four-year assumption over five years invents the fifth year."""
    with pytest.raises(ValueError, match="4 values for a 5-year forecast"):
        build(company=company, base_revenue=1000.0, base_ebitda=200.0, horizon=5,
              revenue_growth=[0.05, 0.05, 0.05, 0.05], ebitda_margin=0.2,
              da_pct_revenue=0.04, capex_pct_revenue=0.05, tax_rate=0.25,
              risk_free_rate=0.04, equity_risk_premium=0.055, beta=1.0, cost_of_debt=0.07,
              entry_ebitda_multiple=8.0, exit_ebitda_multiple=8.0, holding_period=5)


def test_varying_returns_a_new_tree_and_leaves_the_original_untouched(deal):
    """The single most common way a financial model produces a wrong number is state left
    behind from the previous run. Immutability makes that impossible rather than unlikely."""
    original_multiple = deal.transaction.entry_ebitda_multiple
    variant = deal.vary(**{"transaction.entry_ebitda_multiple": 12.0})

    assert variant.transaction.entry_ebitda_multiple == 12.0
    assert deal.transaction.entry_ebitda_multiple == original_multiple
    assert variant is not deal


def test_varying_one_field_shares_everything_else(deal):
    variant = deal.vary(**{"terminal.terminal_growth": 0.03})
    assert variant.operating is deal.operating
    assert variant.company is deal.company


def test_a_holding_period_beyond_the_forecast_is_refused(company, tranches):
    """There would be no modelled EBITDA to exit on."""
    with pytest.raises(ValueError, match="exceeds the 5-year forecast"):
        build(company=company, base_revenue=1000.0, base_ebitda=200.0, horizon=5,
              revenue_growth=0.05, ebitda_margin=0.2, da_pct_revenue=0.04,
              capex_pct_revenue=0.05, tax_rate=0.25, risk_free_rate=0.04,
              equity_risk_premium=0.055, beta=1.0, cost_of_debt=0.07,
              entry_ebitda_multiple=8.0, exit_ebitda_multiple=8.0, holding_period=7,
              tranches=tranches)


# --- WACC, component by component ------------------------------------------------------------------

def test_cost_of_equity_follows_capm(deal):
    w = deal.wacc
    assert w.cost_of_equity == pytest.approx(w.risk_free_rate + w.beta * w.equity_risk_premium)


def test_cost_of_debt_is_reduced_by_the_tax_shield(deal):
    w = deal.wacc
    assert w.after_tax_cost_of_debt == pytest.approx(w.cost_of_debt * (1 - w.tax_rate))
    assert w.after_tax_cost_of_debt < w.cost_of_debt


def test_wacc_is_the_weighted_average_of_its_two_costs(deal):
    w = deal.wacc
    expected = (w.equity_weight * w.cost_of_equity
                + w.debt_weight * w.after_tax_cost_of_debt)
    assert w.wacc == pytest.approx(expected)
    assert w.equity_weight + w.debt_weight == pytest.approx(1.0)


def test_wacc_sits_between_the_two_component_costs(deal):
    w = deal.wacc
    assert min(w.cost_of_equity, w.after_tax_cost_of_debt) <= w.wacc
    assert w.wacc <= max(w.cost_of_equity, w.after_tax_cost_of_debt)


# --- The forecast -------------------------------------------------------------------------------------

def test_revenue_compounds_at_the_given_growth_rates(deal):
    f = forecast_engine.build(deal)
    expected = deal.base_revenue
    for i, growth in enumerate(deal.operating.revenue_growth):
        expected *= (1 + growth)
        assert f.at(i + 1)["revenue"] == pytest.approx(expected)


def test_ebit_is_ebitda_less_depreciation(deal):
    f = forecast_engine.build(deal)
    for _, row in f.frame.iterrows():
        assert row["ebit"] == pytest.approx(row["ebitda"] - row["da"])


def test_growing_working_capital_consumes_cash(deal):
    """An increase in net working capital is a cash outflow, so it subtracts from free cash
    flow. Getting this sign backwards is a routine source of a wrong model."""
    f = forecast_engine.build(deal)
    row = f.at(1)
    assert row["change_in_nwc"] > 0
    reconstructed = row["nopat"] + row["da"] - row["capex"] - row["change_in_nwc"]
    assert row["ufcf"] == pytest.approx(reconstructed)


def test_working_capital_charges_only_the_increment_not_the_balance(deal):
    """Applying the intensity to the revenue level would produce a phantom first-year outflow
    equal to the entire opening balance."""
    f = forecast_engine.build(deal)
    first = f.at(1)
    intensity = deal.operating.net_working_capital_pct
    revenue_increase = first["revenue"] - deal.base_revenue
    assert first["change_in_nwc"] == pytest.approx(revenue_increase * intensity)
    assert first["change_in_nwc"] < first["nwc"]


def test_shrinking_working_capital_releases_cash(company, tranches):
    """A business shrinking its receivables generates cash, so the change is negative and
    therefore adds to free cash flow."""
    shrinking = build(
        company=company, base_revenue=1000.0, base_ebitda=200.0, horizon=3,
        revenue_growth=-0.10, ebitda_margin=0.2, da_pct_revenue=0.04,
        capex_pct_revenue=0.05, tax_rate=0.25, risk_free_rate=0.04,
        equity_risk_premium=0.055, beta=1.0, cost_of_debt=0.07,
        entry_ebitda_multiple=8.0, exit_ebitda_multiple=8.0, holding_period=3,
        tranches=tranches, receivables_pct_revenue=0.20)
    f = forecast_engine.build(shrinking)
    assert f.at(1)["change_in_nwc"] < 0


def test_unlevered_free_cash_flow_contains_no_interest(deal):
    """Tax is charged on EBIT and interest never enters the cash flow, because WACC already
    carries the debt tax shield. Including it in both values the same benefit twice."""
    f = forecast_engine.build(deal)
    row = f.at(1)
    assert row["unlevered_tax"] == pytest.approx(row["ebit"] * deal.operating.tax_rate)
    # An interest charge would make NOPAT depend on the capital structure, and it does not.
    levered = deal.vary(**{"transaction.tranches": ()})
    assert forecast_engine.build(levered).at(1)["ufcf"] == pytest.approx(row["ufcf"])


def test_a_forecast_needs_positive_base_revenue(deal):
    with pytest.raises(ValueError, match="base revenue must be positive"):
        forecast_engine.build(deal.vary(base_revenue=0.0))


# --- The DCF --------------------------------------------------------------------------------------------

def test_present_values_reconcile_to_the_enterprise_value(deal):
    """The transparency requirement made testable: the components must sum to the total."""
    f = forecast_engine.build(deal)
    result = dcf_engine.run(deal, f)
    assert result.pv_forecast + result.pv_terminal == pytest.approx(result.enterprise_value)
    assert result.schedule["pv"].sum() == pytest.approx(result.pv_forecast)


def test_each_discount_factor_matches_the_formula(deal):
    f = forecast_engine.build(deal)
    result = dcf_engine.run(deal, f)
    for _, row in result.schedule.iterrows():
        expected = 1.0 / (1.0 + result.wacc) ** row["year"]
        assert row["discount_factor"] == pytest.approx(expected)


def test_the_equity_bridge_subtracts_net_debt(deal):
    f = forecast_engine.build(deal)
    result = dcf_engine.run(deal, f)
    assert result.equity_value == pytest.approx(
        result.enterprise_value - deal.company.net_debt)
    assert result.implied_share_price == pytest.approx(
        result.equity_value / deal.company.shares_outstanding)


def test_a_higher_discount_rate_lowers_the_valuation(deal):
    f = forecast_engine.build(deal)
    low = dcf_engine.run(deal, f, wacc_override=0.08).enterprise_value
    high = dcf_engine.run(deal, f, wacc_override=0.12).enterprise_value
    assert high < low


def test_higher_terminal_growth_raises_the_valuation(deal):
    f = forecast_engine.build(deal)
    low = dcf_engine.run(deal.vary(**{"terminal.terminal_growth": 0.01}), f).enterprise_value
    high = dcf_engine.run(deal.vary(**{"terminal.terminal_growth": 0.03}), f).enterprise_value
    assert high > low


def test_terminal_growth_at_the_discount_rate_is_refused(deal):
    """The Gordon formula divides by the difference, so the result would be an artefact of the
    division rather than a valuation."""
    f = forecast_engine.build(deal)
    broken = deal.vary(**{"terminal.terminal_growth": deal.wacc.wacc})
    with pytest.raises(ValueError, match="too close to the WACC"):
        dcf_engine.run(broken, f)


def test_the_exit_multiple_method_uses_final_year_ebitda(deal):
    f = forecast_engine.build(deal)
    variant = deal.vary(**{"terminal.method": "exit_multiple",
                           "terminal.exit_ebitda_multiple": 10.0})
    result = dcf_engine.run(variant, f)
    assert result.terminal_value == pytest.approx(f.ebitda_at(5) * 10.0)
    assert any("imports current market pricing" in w for w in result.warnings)


def test_the_mid_year_convention_raises_the_valuation(deal):
    """Cash arrives through the year rather than on its final day, so discounting from the
    year end understates value by roughly half a year of the discount rate."""
    f = forecast_engine.build(deal)
    year_end = dcf_engine.run(deal, f, mid_year=False).pv_forecast
    mid_year = dcf_engine.run(deal, f, mid_year=True).pv_forecast
    assert mid_year > year_end


def test_the_terminal_share_reports_how_much_is_assumption(deal):
    f = forecast_engine.build(deal)
    result = dcf_engine.run(deal, f)
    assert 0 < result.terminal_share < 1
    assert result.terminal_share == pytest.approx(
        result.pv_terminal / result.enterprise_value)


def test_the_calculation_trail_reconstructs_the_answer(deal):
    """The transparency requirement: a user must be able to check the number by hand."""
    f = forecast_engine.build(deal)
    result = dcf_engine.run(deal, f)
    trail = "\n".join(result.calculation_trail())
    assert "Enterprise value" in trail
    assert "Implied share price" in trail
    assert len(result.calculation_trail()) > deal.operating.horizon


# --- Validation ------------------------------------------------------------------------------------------

def test_a_clean_deal_produces_no_blocking_issues(deal):
    assert validation.validate(deal).ok


def test_growth_above_the_discount_rate_blocks_the_model(deal):
    report = validation.validate(deal.vary(**{"terminal.terminal_growth": 0.15}))
    assert not report.ok
    assert any("at or above the WACC" in i.message for i in report.blocking)


def test_a_negative_ebitda_margin_blocks_the_model(deal):
    report = validation.validate(deal.vary(**{"operating.ebitda_margin": (-0.05,) * 5}))
    assert not report.ok


def test_aggressive_leverage_warns_without_blocking(deal, tranches):
    heavy = list(tranches)
    heavy[0] = DebtTranche("Senior Term Loan", 7.0, 0.075, amortisation_pct=0.05)
    report = validation.validate(deal.vary(**{"transaction.tranches": tuple(heavy)}))
    assert report.ok
    assert any("aggressive" in i.message for i in report.warnings)


def test_multiple_expansion_is_noted_rather_than_warned(deal):
    """Assuming a higher exit than entry multiple is a bet on the market, which is worth
    surfacing but is not an error."""
    report = validation.validate(deal)
    assert any("multiple expansion" in i.message for i in report.notes)
