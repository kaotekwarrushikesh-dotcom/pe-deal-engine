"""Tests for the LBO engine, the value bridge and the scenario layer.

The properties pinned here are the ones that separate a working buyout model from a plausible
one: sources balancing to uses by construction, debt never going negative, the circular
interest reference converging, IRR refusing to invent a rate where none exists, the value
bridge reconciling exactly, and the maximum-price solver distinguishing an infeasible structure
from an unattractive one.
"""

import numpy as np
import pandas as pd
import pytest

from pe_engine import bridge as bridge_engine
from pe_engine import committee as committee_engine
from pe_engine import dcf as dcf_engine
from pe_engine import forecast as forecast_engine
from pe_engine import lbo as lbo_engine
from pe_engine import scenarios as scenario_engine
from pe_engine import sensitivity
from pe_engine.assumptions import DebtTranche


@pytest.fixture
def built(deal):
    forecast = forecast_engine.build(deal)
    return deal, forecast, lbo_engine.run(deal, forecast)


# --- Sources and uses -----------------------------------------------------------------------

def test_sources_balance_to_uses(built):
    """Sponsor equity is the plug, so this holds by construction. The check exists to catch an
    arithmetic error in the funding table, not a user mistake."""
    _, _, result = built
    assert result.sources_uses.balances
    assert result.sources_uses.gap == pytest.approx(0.0, abs=0.01)


def test_entry_enterprise_value_is_ebitda_times_the_multiple(built):
    deal, _, result = built
    sources = result.sources_uses
    assert sources.entry_enterprise_value == pytest.approx(
        deal.base_ebitda * deal.transaction.entry_ebitda_multiple)


def test_debt_is_sized_in_turns_of_entry_ebitda(built):
    deal, _, result = built
    for tranche in deal.transaction.tranches:
        if tranche.is_revolver:
            continue
        assert result.sources_uses.debt_raised[tranche.name] == pytest.approx(
            deal.base_ebitda * tranche.turns_of_ebitda)


def test_a_higher_entry_price_requires_more_sponsor_equity(deal):
    """Debt is fixed in turns of EBITDA, so every extra pound of price comes from the sponsor."""
    forecast = forecast_engine.build(deal)
    cheap = lbo_engine.run(deal, forecast, entry_multiple_override=8.0).sponsor_equity
    dear = lbo_engine.run(deal, forecast, entry_multiple_override=11.0).sponsor_equity
    assert dear > cheap


def test_loss_making_businesses_cannot_be_priced_on_an_ebitda_multiple(deal):
    forecast = forecast_engine.build(deal)
    with pytest.raises(ValueError, match="Entry EBITDA must be positive"):
        lbo_engine.run(deal.vary(base_ebitda=-50.0), forecast)


# --- The debt schedule -------------------------------------------------------------------------

def test_debt_never_goes_negative(built):
    """A tranche cannot repay more than it owes."""
    _, _, result = built
    assert (result.schedule["closing_debt"] >= -1e-9).all()
    assert (result.schedule["revolver"] >= -1e-9).all()


def test_debt_rolls_forward_from_one_year_to_the_next(built):
    _, _, result = built
    schedule = result.schedule
    for i in range(1, len(schedule)):
        assert schedule.iloc[i]["opening_debt"] == pytest.approx(
            schedule.iloc[i - 1]["closing_debt"])


def test_repayments_reconcile_the_opening_balance_to_the_closing_one(built):
    """Opening less mandatory less optional equals closing, unless the revolver was drawn."""
    _, _, result = built
    for _, row in result.schedule.iterrows():
        repaid = row["mandatory_repayment"] + row["optional_repayment"]
        movement = row["opening_debt"] - row["closing_debt"]
        assert movement <= repaid + 1e-6


def test_a_cash_sweep_repays_more_debt_than_mandatory_amortisation_alone(deal):
    forecast = forecast_engine.build(deal)
    swept = lbo_engine.run(deal.vary(**{"transaction.sweep_policy": "full_sweep"}), forecast)
    minimal = lbo_engine.run(
        deal.vary(**{"transaction.sweep_policy": "mandatory_only"}), forecast)
    assert swept.exit_net_debt < minimal.exit_net_debt
    assert swept.irr > minimal.irr


def test_a_partial_sweep_sits_between_the_two_extremes(deal):
    forecast = forecast_engine.build(deal)
    full = lbo_engine.run(deal.vary(**{"transaction.sweep_policy": "full_sweep"}), forecast)
    half = lbo_engine.run(deal.vary(**{"transaction.sweep_policy": "partial_sweep",
                                       "transaction.sweep_pct": 0.5}), forecast)
    none = lbo_engine.run(deal.vary(**{"transaction.sweep_policy": "mandatory_only"}), forecast)
    assert full.exit_net_debt <= half.exit_net_debt <= none.exit_net_debt


def test_leverage_falls_over_the_holding_period_when_cash_is_swept(built):
    _, _, result = built
    assert result.exit_leverage < result.entry_leverage


def test_cash_taxes_are_charged_on_profit_after_interest(built):
    """The substantive difference from the DCF: the sponsor genuinely deducts interest, and
    that shield is a real source of returns."""
    deal, forecast, result = built
    first = result.schedule.iloc[0]
    ebit = float(forecast.at(1)["ebit"])
    expected = max(ebit - first["cash_interest"], 0.0) * deal.operating.tax_rate
    assert first["cash_taxes"] == pytest.approx(expected)
    # And it is lower than the unlevered tax the DCF charges on the same year.
    assert first["cash_taxes"] < float(forecast.at(1)["unlevered_tax"])


# --- The circular interest reference ----------------------------------------------------------------

def test_beginning_balance_interest_needs_only_one_pass(deal):
    """No circularity: interest is fixed before any repayment happens."""
    forecast = forecast_engine.build(deal)
    result = lbo_engine.run(deal.vary(**{"transaction.interest_basis": "beginning"}), forecast)
    assert result.converged
    assert result.iterations == 1


def test_average_balance_interest_converges_and_says_how_many_passes(deal):
    """Genuinely circular: interest depends on the balance, which depends on the repayments
    that the interest determines."""
    forecast = forecast_engine.build(deal)
    result = lbo_engine.run(deal.vary(**{"transaction.interest_basis": "average"}), forecast)
    assert result.converged
    assert result.iterations > 1
    assert any("iteratively" in n for n in result.notes)


def test_average_balance_interest_is_lower_than_beginning_balance(deal):
    """Debt falls through the year, so the average balance is below the opening one and the
    interest charge is smaller."""
    forecast = forecast_engine.build(deal)
    beginning = lbo_engine.run(deal.vary(**{"transaction.interest_basis": "beginning"}), forecast)
    average = lbo_engine.run(deal.vary(**{"transaction.interest_basis": "average"}), forecast)
    assert average.schedule["cash_interest"].sum() < beginning.schedule["cash_interest"].sum()


# --- Returns -------------------------------------------------------------------------------------------

def test_moic_is_exit_equity_over_invested_equity(built):
    _, _, result = built
    assert result.moic == pytest.approx(result.exit_equity / result.sponsor_equity)


def test_irr_matches_the_compound_rate_implied_by_the_moic(built):
    """With a single entry and a single exit and no interim flows, IRR is the geometric rate."""
    _, _, result = built
    expected = result.moic ** (1 / result.holding_period) - 1
    assert result.irr == pytest.approx(expected, rel=1e-4)


def test_irr_refuses_to_invent_a_rate_without_a_sign_change():
    """A stream of only outflows has no rate at which the present value is zero."""
    assert np.isnan(lbo_engine.irr([-100.0, -50.0, -20.0]))
    assert np.isnan(lbo_engine.irr([100.0, 50.0]))
    assert np.isnan(lbo_engine.irr([-100.0]))


def test_irr_solves_a_known_case():
    """Doubling money in three years is a 25.99% annual rate."""
    assert lbo_engine.irr([-100.0, 0.0, 0.0, 200.0]) == pytest.approx(0.2599, abs=1e-3)


def test_a_higher_exit_multiple_raises_returns(deal):
    forecast = forecast_engine.build(deal)
    low = lbo_engine.run(deal, forecast, exit_multiple_override=8.0).irr
    high = lbo_engine.run(deal, forecast, exit_multiple_override=11.0).irr
    assert high > low


def test_a_higher_entry_multiple_lowers_returns(deal):
    forecast = forecast_engine.build(deal)
    cheap = lbo_engine.run(deal, forecast, entry_multiple_override=8.0).irr
    dear = lbo_engine.run(deal, forecast, entry_multiple_override=11.0).irr
    assert cheap > dear


def test_a_wiped_out_sponsor_reports_a_total_loss_rather_than_a_percentage(deal):
    """Under limited liability the equity cannot go below zero, so the return floors at -100%."""
    forecast = forecast_engine.build(deal)
    result = lbo_engine.run(deal, forecast, exit_multiple_override=2.0)
    if result.exit_equity <= 0:
        assert result.irr == -1.0
        assert any("wiped out" in w for w in result.warnings)


# --- The maximum entry price solver -----------------------------------------------------------------------

def test_the_maximum_price_falls_as_the_target_return_rises(deal):
    forecast = forecast_engine.build(deal)
    lenient = lbo_engine.maximum_entry_multiple(deal, forecast, target_irr=0.15)
    demanding = lbo_engine.maximum_entry_multiple(deal, forecast, target_irr=0.25)
    assert lenient["max_entry_multiple"] > demanding["max_entry_multiple"]


def test_paying_the_maximum_price_produces_the_target_return(deal):
    """The defining property of the solver, checked by running the model at its answer."""
    forecast = forecast_engine.build(deal)
    solved = lbo_engine.maximum_entry_multiple(deal, forecast, target_irr=0.20)
    at_max = lbo_engine.run(deal, forecast,
                            entry_multiple_override=solved["max_entry_multiple"])
    assert at_max.irr == pytest.approx(0.20, abs=1e-3)


def test_an_infeasible_structure_is_not_mistaken_for_an_unachievable_target(deal):
    """Below a certain entry price the debt raised exceeds the funding requirement and sponsor
    equity goes negative. That is an impossible structure, not an attractive one, and a solver
    that read the resulting undefined IRR as 'target unreachable' would reject deals that work
    at a sensible price because they break at an absurd one."""
    forecast = forecast_engine.build(deal)

    # The base deal returns above 20%, so a 20% target must be reported as achievable even
    # though the IRR is undefined at the bottom of the default search range.
    assert lbo_engine.run(deal, forecast).irr > 0.20
    assert lbo_engine.run(deal, forecast, entry_multiple_override=1.0).sponsor_equity < 0

    solved = lbo_engine.maximum_entry_multiple(deal, forecast, target_irr=0.20)
    assert solved["achievable"]
    assert solved["max_entry_multiple"] > 1.0


def test_an_unreachable_target_is_reported_honestly(deal):
    forecast = forecast_engine.build(deal)
    solved = lbo_engine.maximum_entry_multiple(deal, forecast, target_irr=5.0)
    assert not solved["achievable"]
    assert np.isnan(solved["max_entry_multiple"])
    assert "not achievable" in solved["reading"]


def test_the_margin_of_safety_compares_the_maximum_to_the_proposed_price(deal):
    forecast = forecast_engine.build(deal)
    solved = lbo_engine.maximum_entry_multiple(deal, forecast, target_irr=0.15)
    expected = ((solved["max_entry_multiple"] - deal.transaction.entry_ebitda_multiple)
                / deal.transaction.entry_ebitda_multiple)
    assert solved["margin_of_safety"] == pytest.approx(expected)


# --- The value creation bridge ------------------------------------------------------------------------------

def test_the_four_components_sum_exactly_to_value_created(built):
    """An algebraic identity. Dropping the interaction term breaks it, and the difference is
    usually absorbed into EBITDA growth, flattering the operational story."""
    _, _, result = built
    bridge = bridge_engine.build(result)
    assert bridge.reconciles()
    assert sum(bridge.components.values()) == pytest.approx(bridge.total_created)


def test_value_created_is_exit_equity_less_entry_equity(built):
    _, _, result = built
    bridge = bridge_engine.build(result)
    assert bridge.total_created == pytest.approx(result.exit_equity - result.sponsor_equity)


def test_the_interaction_term_is_the_product_of_both_changes(built):
    _, _, result = built
    bridge = bridge_engine.build(result)
    expected = ((bridge.exit_ebitda - bridge.entry_ebitda)
                * (bridge.exit_multiple - bridge.entry_multiple))
    assert bridge.interaction == pytest.approx(expected)


def test_dropping_the_interaction_term_breaks_the_reconciliation(built):
    """Demonstrates why the fourth component exists rather than asserting it."""
    _, _, result = built
    bridge = bridge_engine.build(result)
    three_way = bridge.ebitda_growth + bridge.multiple_change + bridge.debt_paydown
    assert three_way != pytest.approx(bridge.total_created)
    assert abs(three_way - bridge.total_created) == pytest.approx(abs(bridge.interaction))


def test_a_flat_multiple_produces_no_multiple_or_interaction_contribution(deal):
    forecast = forecast_engine.build(deal)
    flat = lbo_engine.run(deal, forecast, entry_multiple_override=9.0,
                          exit_multiple_override=9.0)
    bridge = bridge_engine.build(flat)
    assert bridge.multiple_change == pytest.approx(0.0)
    assert bridge.interaction == pytest.approx(0.0)
    assert bridge.reconciles()


def test_multiple_contraction_shows_as_negative_value_and_is_flagged(deal):
    forecast = forecast_engine.build(deal)
    contracted = lbo_engine.run(deal, forecast, entry_multiple_override=10.0,
                                exit_multiple_override=8.0)
    bridge = bridge_engine.build(contracted)
    assert bridge.multiple_change < 0
    assert any("multiple contraction destroyed value" in n for n in bridge.notes)


def test_contribution_shares_sum_to_one(built):
    _, _, result = built
    bridge = bridge_engine.build(result)
    assert sum(bridge.contribution_pct.values()) == pytest.approx(1.0)


def test_the_allocated_view_also_sums_to_one_and_has_three_parts(built):
    """A presentation convention rather than a measurement, so it is offered separately."""
    _, _, result = built
    allocated = bridge_engine.build(result).allocated_contribution_pct
    assert len(allocated) == 3
    assert sum(allocated.values()) == pytest.approx(1.0)


# --- Scenarios -----------------------------------------------------------------------------------------------

def test_scenarios_are_recomputed_rather_than_scaled(deal):
    """A downside is not the base case multiplied by a haircut: lower EBITDA both reduces the
    exit value and slows the debt paydown that would have cushioned it."""
    outcomes = scenario_engine.run_all(deal)
    by_name = {o.name: o for o in outcomes}
    assert set(by_name) == {"Bear", "Base", "Bull"}

    bear, base, bull = by_name["Bear"], by_name["Base"], by_name["Bull"]
    assert bear.enterprise_value < base.enterprise_value < bull.enterprise_value
    assert bear.irr < base.irr < bull.irr


def test_a_scenario_leaves_the_base_assumptions_untouched(deal):
    original = deal.operating.revenue_growth
    scenario_engine.run_all(deal)
    assert deal.operating.revenue_growth == original


def test_a_broken_scenario_is_reported_rather_than_dropped(deal):
    """That the bear case breaks the capital structure is the most useful thing a downside can
    tell you, so it must not vanish from the table."""
    brutal = scenario_engine.ScenarioDefinition("Collapse", revenue_growth_delta=-0.5,
                                                ebitda_margin_delta=-0.5,
                                                terminal_growth_delta=0.5)
    outcomes = scenario_engine.run_all(deal, definitions=(brutal,))
    assert len(outcomes) == 1
    assert outcomes[0].error is not None


def test_scenario_deltas_are_relative_so_they_survive_a_changed_base(deal):
    """A bear case defined as an absolute rate becomes wrong the moment the base is revised."""
    definition = scenario_engine.ScenarioDefinition("Bear", revenue_growth_delta=-0.03)
    applied = scenario_engine.apply(deal, definition)
    for original, shifted in zip(deal.operating.revenue_growth, applied.operating.revenue_growth):
        assert shifted == pytest.approx(original - 0.03)


# --- Sensitivity -----------------------------------------------------------------------------------------------

def test_the_dcf_grid_rises_toward_low_wacc_and_high_growth(deal):
    forecast = forecast_engine.build(deal)
    grid = sensitivity.dcf_wacc_vs_growth(deal, forecast)
    assert grid.shape == (5, 5)
    top_left = grid.iloc[0, -1]      # lowest WACC, highest growth
    bottom_right = grid.iloc[-1, 0]  # highest WACC, lowest growth
    assert top_left > bottom_right


def test_the_lbo_grid_rises_with_exit_and_falls_with_entry(deal):
    forecast = forecast_engine.build(deal)
    grid = sensitivity.lbo_entry_vs_exit(deal, forecast, metric="irr")
    assert grid.shape == (5, 5)
    values = grid.to_numpy()
    assert values[0, -1] > values[-1, 0]   # cheap entry, rich exit beats the reverse
    assert np.all(np.diff(values, axis=1) > 0)   # rising exit multiple raises IRR
    assert np.all(np.diff(values, axis=0) < 0)   # rising entry multiple lowers IRR


def test_the_driver_analysis_ranks_by_impact(deal):
    frame = sensitivity.dcf_value_drivers(deal)
    impacts = frame["impact"].abs().to_numpy()
    assert np.all(np.diff(impacts) <= 1e-9)
    assert len(frame) >= 5


# --- The committee view ---------------------------------------------------------------------------------------------

def test_the_conclusions_reference_computed_outputs(deal):
    """No generic statements: every sentence must cite a number the model produced."""
    forecast = forecast_engine.build(deal)
    valuation = dcf_engine.run(deal, forecast)
    buyout = lbo_engine.run(deal, forecast)
    bridge = bridge_engine.build(buyout)
    max_entry = lbo_engine.maximum_entry_multiple(deal, forecast)
    view = committee_engine.compare(deal, valuation, buyout, bridge, max_entry)

    assert f"{buyout.moic:.2f}x" in view.lbo_conclusion
    assert f"{buyout.irr:.1%}" in view.lbo_conclusion
    assert f"{valuation.terminal_share:.0%}" in view.dcf_conclusion
    assert f"{deal.target_irr:.0%}" in view.assessment
    assert len(view.comparison) > 8


def test_the_audit_trail_maps_inputs_to_outputs(deal):
    trail = committee_engine.audit_trail(deal)
    assert set(trail.columns) == {"Input", "Field", "Flows through", "Affects"}
    assert len(trail) >= 10
