"""Private Equity Valuation and Deal Analysis Engine, as an app.

Every material assumption is a user input. Nothing here holds a view: change a growth rate and
the forecast, the DCF, the debt schedule, the returns, the sensitivity grids, the scenarios and
the committee conclusion all move together, because they all read the same assumption tree.

The interface layer contains no financial calculations. It collects inputs, calls the engine,
and renders what comes back. That separation is deliberate: the engine has 75 tests and the
interface has none, so anything that matters has to live in the engine.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from pe_engine import bridge as bridge_engine
from pe_engine import committee as committee_engine
from pe_engine import dcf as dcf_engine
from pe_engine import forecast as forecast_engine
from pe_engine import lbo as lbo_engine
from pe_engine import scenarios as scenario_engine
from pe_engine import sensitivity
from pe_engine import validation
from pe_engine.assumptions import Company, DebtTranche, build

st.set_page_config(page_title="PE Valuation & Deal Analysis Engine", page_icon="📐",
                   layout="wide")

GREEN, AMBER, RED, BLUE, GREY = "#1e7a4b", "#b8860b", "#a32020", "#2b6cb0", "#8a8a8a"


def pct(v, dp=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{dp}%}"


def num(v, dp=0):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{dp}f}"


def mult(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}x"


def chart(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      template="plotly_white", legend=dict(orientation="h", y=1.12, x=0),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def heatmap(frame: pd.DataFrame, fmt: str = ".1%", reverse: bool = False):
    fig = go.Figure(data=go.Heatmap(
        z=frame.to_numpy(), x=list(frame.columns), y=list(frame.index),
        colorscale="RdYlGn" if not reverse else "RdYlGn_r",
        text=frame.to_numpy(), texttemplate="%{text:" + fmt + "}", showscale=False))
    fig.update_layout(xaxis_title=frame.columns.name, yaxis_title=frame.index.name)
    chart(fig, 330)


st.title("Private Equity Valuation & Deal Analysis Engine")
st.caption(
    "One company, two institutional methodologies. The DCF asks what the business is worth to "
    "a permanent owner on its own cash flows. The LBO asks what a sponsor can pay and still "
    "earn a required return. They answer different questions, and the gap between them is the "
    "point rather than a tie-break. Every assumption below is yours to set."
)

# =============================== SIDEBAR: THE ASSUMPTION ENGINE =================================

with st.sidebar:
    st.header("Assumptions")
    st.caption("Everything the model knows. Nothing is hard-coded.")

    with st.expander("Company", expanded=True):
        name = st.text_input("Name", "Meridian Industrial")
        ticker = st.text_input("Ticker", "MRDN")
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "INR"], index=0)
        share_price = st.number_input("Current share price", value=42.50, step=0.50)
        shares = st.number_input("Shares outstanding (m)", value=50.0, step=1.0)
        existing_debt = st.number_input("Existing debt (m)", value=300.0, step=10.0)
        existing_cash = st.number_input("Cash (m)", value=60.0, step=10.0)

    with st.expander("Last historical year", expanded=True):
        base_revenue = st.number_input("Revenue (m)", value=1200.0, step=50.0)
        base_ebitda = st.number_input("EBITDA (m)", value=240.0, step=10.0)
        st.caption(f"Implied margin {base_ebitda / base_revenue:.1%}" if base_revenue else "")

    with st.expander("Operating forecast", expanded=True):
        horizon = st.slider("Forecast years", 3, 10, 5)
        st.caption("Per-year rates. A fading path is the normal case in a real model.")
        growth_text = st.text_input(
            "Revenue growth by year (%)",
            ", ".join(["8", "7", "6", "5", "4"][:horizon] + ["4"] * max(0, horizon - 5)))
        margin_text = st.text_input(
            "EBITDA margin by year (%)",
            ", ".join(["20.5", "21", "21.5", "22", "22"][:horizon] + ["22"] * max(0, horizon - 5)))
        da_pct = st.number_input("D&A (% of revenue)", value=4.0, step=0.25) / 100
        capex_pct = st.number_input("Capex (% of revenue)", value=5.0, step=0.25) / 100
        tax_rate = st.number_input("Tax rate (%)", value=25.0, step=1.0) / 100

    with st.expander("Working capital (% of revenue)"):
        receivables = st.number_input("Receivables", value=14.0, step=0.5) / 100
        inventory = st.number_input("Inventory", value=11.0, step=0.5) / 100
        payables = st.number_input("Payables", value=9.0, step=0.5) / 100

    with st.expander("Cost of capital"):
        risk_free = st.number_input("Risk-free rate (%)", value=4.2, step=0.1) / 100
        erp = st.number_input("Equity risk premium (%)", value=5.5, step=0.1) / 100
        beta = st.number_input("Beta", value=1.15, step=0.05)
        cost_of_debt = st.number_input("Cost of debt (%)", value=7.5, step=0.25) / 100

    with st.expander("Terminal value"):
        terminal_method = st.radio("Method", ["gordon", "exit_multiple"],
                                   format_func=lambda m: "Gordon growth" if m == "gordon"
                                   else "Exit multiple")
        terminal_growth = st.number_input("Terminal growth (%)", value=2.5, step=0.1) / 100
        mid_year = st.checkbox("Mid-year convention", value=False,
                               help="Cash arrives through the year rather than on its last day.")

    with st.expander("Transaction", expanded=True):
        entry_multiple = st.number_input("Entry EBITDA multiple", value=9.0, step=0.25)
        exit_multiple = st.number_input("Exit EBITDA multiple", value=9.5, step=0.25)
        holding_period = st.slider("Holding period (years)", 1, horizon,
                                   min(5, horizon))
        target_irr = st.number_input("Target IRR (%)", value=20.0, step=1.0) / 100
        transaction_fee = st.number_input("Transaction fees (% of EV)", value=2.0, step=0.5) / 100
        financing_fee = st.number_input("Financing fees (% of debt)", value=2.0, step=0.5) / 100
        minimum_cash = st.number_input("Minimum cash (m)", value=25.0, step=5.0)

    with st.expander("Debt structure", expanded=True):
        senior_turns = st.number_input("Senior turns of EBITDA", value=3.5, step=0.25)
        senior_rate = st.number_input("Senior rate (%)", value=7.5, step=0.25) / 100
        senior_amort = st.number_input("Senior amortisation (% a year)", value=5.0, step=1.0) / 100
        mezz_turns = st.number_input("Mezzanine turns", value=1.5, step=0.25)
        mezz_rate = st.number_input("Mezzanine rate (%)", value=11.5, step=0.25) / 100
        revolver_turns = st.number_input("Revolver capacity (turns)", value=0.5, step=0.25)
        revolver_rate = st.number_input("Revolver rate (%)", value=6.5, step=0.25) / 100
        interest_basis = st.radio("Interest on", ["beginning", "average"],
                                  format_func=lambda b: f"{b.capitalize()} balance")
        sweep = st.radio("Cash sweep", ["full_sweep", "partial_sweep", "mandatory_only"],
                         format_func=lambda s: {"full_sweep": "Full excess cash sweep",
                                                "partial_sweep": "Partial sweep",
                                                "mandatory_only": "Mandatory only"}[s])
        sweep_pct = st.slider("Sweep percentage", 0.0, 1.0, 0.5) if sweep == "partial_sweep" else 0.5


def parse_path(text: str, years: int, label: str) -> list[float]:
    values = [float(v.strip()) / 100 for v in text.split(",") if v.strip()]
    if len(values) == 1:
        return values * years
    if len(values) != years:
        raise ValueError(f"{label}: {len(values)} values for a {years}-year forecast")
    return values


try:
    growth = parse_path(growth_text, horizon, "Revenue growth")
    margins = parse_path(margin_text, horizon, "EBITDA margin")
except ValueError as exc:
    st.error(f"Could not read the assumptions: {exc}")
    st.stop()

company = Company(name=name, ticker=ticker, currency=currency, share_price=share_price,
                  shares_outstanding=shares, debt=existing_debt, cash=existing_cash)

tranches = [
    DebtTranche("Senior Term Loan", senior_turns, senior_rate,
                amortisation_pct=senior_amort, cash_sweep_priority=1),
    DebtTranche("Mezzanine", mezz_turns, mezz_rate, cash_sweep_priority=2),
    DebtTranche("Revolver", revolver_turns, revolver_rate, is_revolver=True),
]

try:
    deal = build(
        company=company, base_revenue=base_revenue, base_ebitda=base_ebitda, horizon=horizon,
        revenue_growth=growth, ebitda_margin=margins, da_pct_revenue=da_pct,
        capex_pct_revenue=capex_pct, tax_rate=tax_rate, risk_free_rate=risk_free,
        equity_risk_premium=erp, beta=beta, cost_of_debt=cost_of_debt,
        entry_ebitda_multiple=entry_multiple, exit_ebitda_multiple=exit_multiple,
        holding_period=holding_period, tranches=tranches,
        receivables_pct_revenue=receivables, inventory_pct_revenue=inventory,
        payables_pct_revenue=payables, terminal_method=terminal_method,
        terminal_growth=terminal_growth, transaction_fee_pct=transaction_fee,
        financing_fee_pct=financing_fee, minimum_cash=minimum_cash, target_irr=target_irr,
        interest_basis=interest_basis, sweep_policy=sweep, sweep_pct=sweep_pct,
    )
except ValueError as exc:
    st.error(f"These assumptions cannot build a model: {exc}")
    st.stop()

report = validation.validate(deal)
if not report.ok:
    st.error("**The model cannot run on these assumptions.**")
    for issue in report.blocking:
        st.error(issue.render())
    st.stop()

forecast = forecast_engine.build(deal)
valuation = dcf_engine.run(deal, forecast, mid_year=mid_year)
buyout = lbo_engine.run(deal, forecast)
value_bridge = bridge_engine.build(buyout)
max_entry = lbo_engine.maximum_entry_multiple(deal, forecast)
view = committee_engine.compare(deal, valuation, buyout, value_bridge, max_entry)

tabs = st.tabs(["Dashboard", "Forecast", "DCF", "LBO", "Value creation",
                "Sensitivity", "Scenarios", "Investment committee", "Audit"])

# =============================== DASHBOARD ======================================================

with tabs[0]:
    st.markdown(f"### {deal.company.name} · {currency}")

    left, right = st.columns(2)
    with left:
        st.markdown("**DCF: what the business is worth**")
        a = st.columns(3)
        a[0].metric("Enterprise value", num(valuation.enterprise_value))
        a[1].metric("Equity value", num(valuation.equity_value))
        a[2].metric("Implied price", f"{valuation.implied_share_price:,.2f}",
                    delta=pct(valuation.upside), delta_color="normal")
    with right:
        st.markdown("**LBO: what a sponsor earns**")
        b = st.columns(3)
        b[0].metric("Sponsor equity", num(buyout.sponsor_equity))
        b[1].metric("MOIC", mult(buyout.moic))
        b[2].metric("IRR", pct(buyout.irr),
                    delta=f"target {deal.target_irr:.0%}", delta_color="off")

    st.divider()
    st.markdown("**Deal test**")
    c = st.columns(4)
    c[0].metric("Target IRR", pct(deal.target_irr, 0))
    c[1].metric("Maximum entry", mult(max_entry.get("max_entry_multiple")))
    c[2].metric("Proposed entry", mult(deal.transaction.entry_ebitda_multiple))
    margin = max_entry.get("margin_of_safety", float("nan"))
    c[3].metric("Margin of safety", pct(margin) if not np.isnan(margin) else "n/a",
                delta="headroom" if margin >= 0 else "overpaying", delta_color="off")

    box = st.success if buyout.irr >= deal.target_irr else st.warning
    box(view.assessment)

    for issue in report.warnings:
        st.warning(issue.render())

# =============================== FORECAST =======================================================

with tabs[1]:
    st.markdown("#### Operating forecast")
    st.caption("One projection, read by both models. A DCF and an LBO that disagree because "
               "they were fed different growth rates tell you nothing about the deal.")

    display = forecast.frame.copy()
    for col in ("revenue_growth", "ebitda_margin", "ebit_margin"):
        display[col] = display[col].map(lambda v: f"{v:.1%}")
    for col in ("revenue", "ebitda", "da", "ebit", "nopat", "capex", "nwc",
                "change_in_nwc", "ufcf", "unlevered_tax"):
        display[col] = display[col].map(lambda v: f"{v:,.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=forecast.frame["year"], y=forecast.frame["revenue"],
                         name="Revenue", marker_color=BLUE, opacity=0.35))
    fig.add_trace(go.Bar(x=forecast.frame["year"], y=forecast.frame["ebitda"],
                         name="EBITDA", marker_color=GREEN))
    fig.add_trace(go.Scatter(x=forecast.frame["year"], y=forecast.frame["ufcf"],
                             name="Unlevered FCF", line=dict(color=RED, width=3)))
    fig.update_layout(barmode="overlay", xaxis_title="Forecast year")
    chart(fig)

    st.caption(
        "Free cash flow is unlevered: tax is charged on EBIT and interest never enters, "
        "because the DCF's discount rate already carries the debt tax shield. Including it in "
        "both would value the same benefit twice. The LBO computes its own cash taxes on "
        "profit after interest, which is the substantive difference between the two models."
    )

# =============================== DCF ============================================================

with tabs[2]:
    st.markdown("#### Discounted cash flow")

    w = deal.wacc
    cols = st.columns(5)
    cols[0].metric("Cost of equity", pct(w.cost_of_equity, 2))
    cols[1].metric("After-tax cost of debt", pct(w.after_tax_cost_of_debt, 2))
    cols[2].metric("Equity weight", pct(w.equity_weight, 0))
    cols[3].metric("Debt weight", pct(w.debt_weight, 0))
    cols[4].metric("WACC", pct(valuation.wacc, 2))
    st.caption(f"Cost of equity = {w.risk_free_rate:.2%} + {w.beta:.2f} x "
               f"{w.equity_risk_premium:.2%} = {w.cost_of_equity:.2%}. "
               f"After-tax cost of debt = {w.cost_of_debt:.2%} x (1 - {w.tax_rate:.0%}) "
               f"= {w.after_tax_cost_of_debt:.2%}.")

    st.divider()
    v = st.columns(5)
    v[0].metric("PV of forecast", num(valuation.pv_forecast))
    v[1].metric("PV of terminal", num(valuation.pv_terminal))
    v[2].metric("Enterprise value", num(valuation.enterprise_value))
    v[3].metric("Equity value", num(valuation.equity_value))
    v[4].metric("In terminal value", pct(valuation.terminal_share, 0))

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * horizon + ["relative", "total", "relative", "total"],
        x=[f"Y{int(y)}" for y in valuation.schedule["year"]] +
          ["Terminal", "Enterprise value", "Net debt", "Equity value"],
        y=list(valuation.schedule["pv"]) + [valuation.pv_terminal, 0,
                                            -valuation.net_debt, 0],
        connector={"line": {"color": GREY}},
    ))
    chart(fig, 360)

    for warning in valuation.warnings:
        st.warning(warning)
    for note in valuation.notes:
        st.caption(f"ℹ️ {note}")

    with st.expander("Show calculation"):
        st.code("\n".join(valuation.calculation_trail()), language="text")

# =============================== LBO ============================================================

with tabs[3]:
    st.markdown("#### Leveraged buyout")

    sources = buyout.sources_uses
    if not sources.balances:
        st.error(f"Sources and uses do not balance: gap of {sources.gap:,.2f}")
    else:
        st.success(f"Sources and uses balance at {num(sources.total_uses)}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Uses**")
        st.dataframe(pd.DataFrame([
            {"Item": "Purchase of equity", "Amount": sources.purchase_equity},
            {"Item": "Refinance existing debt", "Amount": sources.refinanced_debt},
            {"Item": "Transaction fees", "Amount": sources.transaction_fees},
            {"Item": "Financing fees", "Amount": sources.financing_fees},
            {"Item": "Minimum cash", "Amount": sources.minimum_cash},
            {"Item": "Total uses", "Amount": sources.total_uses},
        ]).style.format({"Amount": "{:,.1f}"}), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Sources**")
        rows = [{"Item": k, "Amount": v} for k, v in sources.debt_raised.items()]
        rows += [{"Item": "Existing cash", "Amount": sources.existing_cash_used},
                 {"Item": "Sponsor equity", "Amount": sources.sponsor_equity},
                 {"Item": "Total sources", "Amount": sources.total_sources}]
        st.dataframe(pd.DataFrame(rows).style.format({"Amount": "{:,.1f}"}),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Debt schedule**")
    schedule = buyout.schedule.copy()
    for col in schedule.columns:
        if col not in ("year",):
            schedule[col] = schedule[col].map(lambda v: f"{v:,.1f}")
    st.dataframe(schedule, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=buyout.schedule["year"], y=buyout.schedule["closing_debt"],
                         name="Total debt", marker_color=RED, opacity=0.7))
    fig.add_trace(go.Scatter(x=buyout.schedule["year"], y=buyout.schedule["leverage"],
                             name="Net debt / EBITDA", yaxis="y2",
                             line=dict(color=BLUE, width=3)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Turns"),
                      xaxis_title="Year")
    chart(fig)

    r = st.columns(5)
    r[0].metric("Entry leverage", f"{buyout.entry_leverage:.2f}x")
    r[1].metric("Exit leverage", f"{buyout.exit_leverage:.2f}x")
    r[2].metric("Exit equity", num(buyout.exit_equity))
    r[3].metric("MOIC", mult(buyout.moic))
    r[4].metric("IRR", pct(buyout.irr))

    for warning in buyout.warnings:
        st.warning(warning)
    for note in buyout.notes:
        st.caption(f"ℹ️ {note}")

    with st.expander("Show calculation"):
        st.code("\n".join(buyout.calculation_trail()), language="text")

# =============================== VALUE CREATION =================================================

with tabs[4]:
    st.markdown("#### Where the returns came from")
    st.caption(
        "Growing the business, selling it at a higher multiple than was paid, and using the "
        "company's own cash flow to repay debt are genuinely different achievements. Only the "
        "first is unambiguously operational skill."
    )

    b = value_bridge
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["Entry equity", "EBITDA growth", "Multiple change",
           "Growth x multiple", "Debt paydown", "Exit equity"],
        y=[b.entry_equity, b.ebitda_growth, b.multiple_change, b.interaction,
           b.debt_paydown, 0],
        connector={"line": {"color": GREY}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
    ))
    chart(fig, 380)

    contributions = pd.DataFrame([
        {"Source": k, "Value": v, "Share": b.contribution_pct[k]}
        for k, v in b.components.items()])
    st.dataframe(contributions.style.format({"Value": "{:,.1f}", "Share": "{:.1%}"}),
                 use_container_width=True, hide_index=True)

    st.info(
        "**The bridge has four components, not three.** The interaction term is the extra "
        "value from selling more EBITDA at a higher multiple, and it belongs to neither lever "
        "alone. Many published bridges drop it, which means their components no longer "
        "reconcile to the total and the gap is silently absorbed into EBITDA growth, "
        "flattering the operational story. Here the four sum exactly to value created, and "
        "that identity is asserted at runtime rather than trusted."
    )

    with st.expander("A three-way view for presentation"):
        allocated = b.allocated_contribution_pct
        st.dataframe(pd.DataFrame([{"Source": k, "Share": v} for k, v in allocated.items()])
                     .style.format({"Share": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
        st.caption(
            "The interaction is split pro-rata between growth and multiple. That is a "
            "convention rather than a measurement: the joint term genuinely belongs to both, "
            "and no allocation of it is more correct than another."
        )

    for note in b.notes:
        st.caption(f"ℹ️ {note}")

    with st.expander("Show calculation"):
        st.code("\n".join(b.calculation_trail()), language="text")

# =============================== SENSITIVITY ====================================================

with tabs[5]:
    st.markdown("#### Sensitivity")
    st.caption("Two dimensions at a time. A one-at-a-time table hides the interaction, and for "
               "both models the interaction is the point.")

    st.markdown("**DCF: implied share price across WACC and terminal growth**")
    if terminal_method == "gordon":
        heatmap(sensitivity.dcf_wacc_vs_growth(deal, forecast), fmt=",.1f")
        st.caption("Empty cells are combinations where terminal growth approaches WACC. The "
                   "engine refuses those rather than showing an artefact of a near-zero "
                   "denominator.")
    else:
        heatmap(sensitivity.dcf_wacc_vs_exit_multiple(deal, forecast), fmt=",.1f")

    st.markdown("**LBO: sponsor IRR across entry and exit multiples**")
    heatmap(sensitivity.lbo_entry_vs_exit(deal, forecast, metric="irr"))

    st.markdown("**LBO: sponsor IRR across leverage and exit multiple**")
    heatmap(sensitivity.lbo_leverage_vs_exit(deal, forecast, metric="irr"))

    st.markdown("**What the DCF actually depends on**")
    drivers = sensitivity.dcf_value_drivers(deal)
    fig = go.Figure(go.Bar(
        x=drivers["impact"], y=drivers["driver"], orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in drivers["impact"]]))
    fig.update_layout(xaxis_title="Change in implied share price from a 1 point shift")
    chart(fig, 300)
    st.caption("Each driver shocked by the same amount, ranked by impact. This answers where "
               "to spend diligence, not what the answer is.")

# =============================== SCENARIOS ======================================================

with tabs[6]:
    st.markdown("#### Bear, base and bull")
    st.caption(
        "Each scenario is a full independent revaluation, not a percentage haircut on the "
        "base case. A real downside compounds slower growth, weaker margins, a higher discount "
        "rate and a lower exit together, and in a levered structure lower EBITDA also slows "
        "the debt paydown that would have cushioned it."
    )

    outcomes = scenario_engine.run_all(deal)
    table = scenario_engine.frame(outcomes)
    st.dataframe(
        table.style.format({
            "DCF enterprise value": "{:,.0f}", "DCF equity value": "{:,.0f}",
            "Implied share price": "{:,.2f}", "Upside": "{:+.1%}",
            "Sponsor equity": "{:,.0f}", "MOIC": "{:.2f}x", "IRR": "{:.1%}",
            "Exit leverage": "{:.2f}x"}),
        use_container_width=True, hide_index=True)

    computed = [o for o in outcomes if o.error is None]
    if computed:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[o.name for o in computed],
                             y=[o.irr for o in computed], name="IRR",
                             marker_color=[GREEN if o.irr >= deal.target_irr else AMBER
                                           for o in computed]))
        fig.add_hline(y=deal.target_irr, line=dict(color=RED, dash="dash"),
                      annotation_text=f"target {deal.target_irr:.0%}")
        fig.update_layout(yaxis_tickformat=".0%")
        chart(fig, 300)

    for outcome in outcomes:
        if outcome.error:
            st.warning(f"**{outcome.name} could not be computed:** {outcome.error}")

# =============================== INVESTMENT COMMITTEE ===========================================

with tabs[7]:
    st.markdown("#### Investment committee")
    st.dataframe(view.comparison, use_container_width=True, hide_index=True)

    st.markdown("**DCF conclusion**")
    st.info(view.dcf_conclusion)
    st.markdown("**LBO conclusion**")
    st.info(view.lbo_conclusion)
    st.markdown("**Assessment**")
    (st.success if buyout.irr >= deal.target_irr else st.warning)(view.assessment)

    st.caption(
        "The two are not competing estimates of one number. A sponsor who can pay more than "
        "the DCF says the business is worth has not found value the DCF missed; they are using "
        "leverage and a finite horizon to convert the same cash flows into a higher equity "
        "return. A sponsor who cannot reach it is being told the deal needs cheaper debt, a "
        "better operating case, or a higher exit than entry."
    )

# =============================== AUDIT ==========================================================

with tabs[8]:
    st.markdown("#### Audit trail")
    st.caption("Which input feeds which output. Read this before changing an assumption, to "
               "know what it will move.")
    st.dataframe(committee_engine.audit_trail(deal), use_container_width=True, hide_index=True)

    st.markdown("**Validation**")
    st.write(report.summary())
    for issue in report.issues:
        {"blocking": st.error, "warning": st.warning, "note": st.caption}[issue.severity](
            issue.render())

    st.markdown("**Model architecture**")
    st.code(
        "Assumption engine  (immutable tree; every input lives here)\n"
        "        |\n"
        "   Forecast        (one operating case, shared)\n"
        "      /    \\\n"
        "   DCF      LBO    (unlevered value | levered returns)\n"
        "      \\    /\n"
        "  Sensitivity, scenarios, value bridge\n"
        "        |\n"
        "  Investment committee",
        language="text")
    st.caption(
        "The interface layer contains no financial calculations: it collects inputs, calls the "
        "engine and renders results. The engine carries 75 tests and the interface carries "
        "none, so anything that matters lives in the engine."
    )
