# Private Equity Valuation & Deal Analysis Engine

> One company, two institutional methodologies, one set of assumptions. DCF and LBO side by
> side, with every material input under the user's control.

**Status: the engine and interface are built and tested.** Assumption engine, operating
forecast, DCF with both terminal methods, full LBO with a multi-tranche debt schedule and cash
sweep, sponsor returns, maximum entry price, the value creation bridge, sensitivity grids,
scenarios, validation and the investment committee view all run. 75 tests.

The DCF asks what the business is worth to a permanent owner on its own cash flows. The LBO
asks what a financial sponsor can pay and still earn a required return. **They answer different
questions**, and presenting them as competing estimates of one number is the most common way the
comparison is misread. The gap between them is the deliverable.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest tests/ -q
```

```bash
.venv/bin/streamlit run app.py
```

## The design decision everything rests on

There is **one assumption tree**, it is **immutable**, and every model reads from it.

```
Assumption engine  (every input lives here, nowhere else)
        |
   Forecast        (one operating case, shared)
      /    \
   DCF      LBO    (unlevered value | levered returns)
      \    /
  Sensitivity, scenarios, value bridge
        |
  Investment committee
```

A DCF and an LBO that disagree because they were fed different growth rates tell you nothing
about the deal, only about the inconsistency. Both read the same `DealAssumptions`, so that
cannot happen.

Immutability is not stylistic. A sensitivity grid runs eighty variants and a scenario run does
three, and **the single most common way a financial model produces a wrong number is state left
behind from the previous run**. A scenario is a new object built with `vary()`, never a mutated
global, which makes that class of bug impossible rather than unlikely.

## The two models are genuinely independent, and here is the proof

Changing one assumption and watching what moves is the clearest test that the models are
correctly separated:

| Change | DCF enterprise value | DCF price | LBO IRR | MOIC |
|---|---:|---:|---:|---:|
| Base case | 2,253 | 40.26 | 20.3% | 2.52x |
| Revenue growth +2 points a year | 2,389 | 42.97 | 23.4% | 2.86x |
| EBITDA margin -3 points | 1,818 | 31.56 | 13.8% | 1.91x |
| **Beta 1.15 to 1.60** | **1,728** | **29.77** | **20.3%** | **2.52x** |
| **Leverage 5.0x to 6.5x** | **2,253** | **40.26** | **25.5%** | **3.11x** |
| Exit multiple 9.5x to 8.0x | 2,253 | 40.26 | 15.0% | 2.01x |

The two bold rows are the point. **Beta moves the DCF and leaves the IRR untouched**, because
the LBO never uses a discount rate: its return is solved, not assumed. **Leverage moves the IRR
and leaves the DCF untouched**, because unlevered free cash flow is independent of capital
structure by construction. Operating assumptions move both. If either orthogonality broke, one
model would be contaminating the other.

## Four things that are easy to get quietly wrong

### 1. The circular reference is real, and is resolved rather than avoided

Interest expense depends on the debt balance. The debt balance depends on how much cash is
swept to repay it. The cash available depends on interest expense.

Under a **beginning-balance** convention the loop breaks naturally, since interest is fixed
before any repayment happens. Under an **average-balance** convention it does not, and the
schedule is recomputed until the interest charge stops moving. Spreadsheets solve this by
enabling iterative calculation and hoping; this engine iterates explicitly, caps the iterations,
and reports whether it converged. A test asserts that beginning-balance needs one pass and
average-balance needs more than one.

### 2. The value creation bridge has four components, not three

    Value created = Exit equity - Entry equity

    EBITDA growth    (Exit EBITDA - Entry EBITDA) x Entry multiple
    Multiple change  (Exit multiple - Entry multiple) x Entry EBITDA
    Interaction      (Exit EBITDA - Entry EBITDA) x (Exit multiple - Entry multiple)
    Debt paydown     Entry net debt - Exit net debt

Those four sum **exactly** to value created, and the engine asserts it at runtime. Drop the
interaction term, as many published bridges do, and the components no longer reconcile: the gap
is silently absorbed into whichever bucket the author chose, usually EBITDA growth, which
flatters the operational story. On the base case here the interaction is 3.5% of value created,
and a test demonstrates that removing it breaks the identity by exactly its own size.

The interaction has a plain meaning: it is the extra value from selling **more** EBITDA at a
**higher** multiple, and it belongs to neither lever alone. A three-way allocated view is offered
for presentation, labelled as a convention rather than a measurement.

### 3. The maximum entry price solver has to find the feasible range first

Solving for the highest price that still clears a target IRR relies on IRR falling as the entry
multiple rises. That holds. What does not hold is that IRR is *defined* across the whole search
range.

**Below roughly 4.5x on the base case, the debt raised exceeds the total funding requirement**,
so sponsor equity goes negative and there is no invested capital to earn a return on. That is an
infeasible structure, not an attractive one. A solver that read the resulting undefined IRR as
"target unreachable" would report the exact opposite of the truth: it would reject deals that
work at a sensible price because they break at an absurd one. This was a real bug found during
the build, and the fix is to walk the lower bound up until the structure is fundable before
bisecting. A test pins it.

### 4. Tax is treated differently in the two models, on purpose

The DCF charges tax on **unlevered EBIT** and interest never enters the cash flow, because WACC
already carries the debt tax shield in the discount rate. Including it in both would value the
same benefit twice, which is the most common way a DCF flatters itself and always moves the
answer up.

The LBO charges cash tax on **profit after interest**, because the sponsor genuinely does deduct
it and that shield is a real source of buyout returns. This is not an inconsistency between the
models, it is the substantive difference between them. Counting it in both places, or neither,
would be the error. A test asserts the LBO's cash tax is below the DCF's unlevered tax in the
same year.

## Validation

Checks are tiered, because a model that blocks on everything is unusable and one that blocks on
nothing is untrustworthy.

| Tier | Meaning | Example |
|---|---|---|
| **Blocking** | The result would be meaningless, so the model refuses to run | Terminal growth at or above WACC: the Gordon formula divides by their difference |
| **Warning** | Computable, but read the number with this attached | Leverage above 7x, where real lenders impose covenants this model does not represent |
| **Note** | A modelling choice worth surfacing | Exit multiple above entry, so the deal assumes multiple expansion |

## Transparency

Every headline number has a **Show calculation** panel that reconstructs it line by line, so it
can be checked by hand rather than believed. The DCF trail lists each year's cash flow, its
discount factor and its present value, then the terminal value and the bridge to a share price.
The LBO trail lists the full sources and uses table, then the exit and the return arithmetic.

An **audit trail** maps each input to what it flows through and what it ultimately affects, so a
user can see what an assumption will move before changing it.

## What is in the engine

| Module | Responsibility |
|---|---|
| `assumptions.py` | The immutable assumption tree, `vary()`, and the WACC components |
| `forecast.py` | Revenue to unlevered free cash flow, one case shared by both models |
| `dcf.py` | Discounting, both terminal methods, the enterprise-to-equity bridge |
| `lbo.py` | Sources and uses, debt schedule, cash sweep, returns, maximum entry price |
| `bridge.py` | The four-component value creation decomposition |
| `sensitivity.py` | Two-dimensional grids for both models, plus a driver tornado |
| `scenarios.py` | Bear, base and bull, each fully recomputed |
| `validation.py` | Tiered model checks |
| `committee.py` | The comparison table, conclusions and audit trail |

The interface layer contains no financial calculations. It collects inputs, calls the engine and
renders results. The engine carries all 75 tests and the interface carries none, so anything
that matters lives in the engine.

## Known limitations

1. **No circularity between the DCF and the capital structure.** WACC uses the weights the user
   enters rather than the post-transaction structure. That is correct for valuing the business
   as it stands, and it means the DCF does not reflect the LBO's leverage, which is deliberate.
2. **The debt schedule has no covenants.** Real credit agreements impose maintenance tests that
   would block some of the structures this model will happily run. Leverage above 7x raises a
   warning rather than a refusal.
3. **Interim sponsor cash flows are supported by the IRR function but not yet by the interface.**
   Dividend recapitalisations would change returns materially and are not modelled.
4. **Scenarios shift assumptions by fixed deltas.** A genuinely correlated downside, where
   margin compression and multiple contraction arrive together because they share a cause, is
   more realistic than independent shifts.
5. **Historical financials are entered rather than fetched.** The sibling modules in this
   platform fetch filings automatically; here the user supplies the base year, because the point
   of the tool is that the assumptions are theirs.
