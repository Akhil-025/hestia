# Pluto: personal finance

Pluto logs expenses and investments, answers questions about them, and (new) helps you plan.
Everything below is stored in one local SQLite file, `data/pluto/pluto.db`. The market-data,
portfolio, backtest and forecast features also use Postgres/Redis/Qdrant and live data sources
and are described in their own module docstrings.

## What you can say

| Say | Intent | What happens |
|---|---|---|
| "log 500 for groceries" | `pluto_log_expense` | Saves it. Accepts "₹1,500", "5k", "2 lakh". Rejects zero, negatives, nan, absurd values. |
| "give me my budget summary" | `pluto_get_budget_summary` | Totals by category (where the money went). |
| "set a food budget of 8000" | `pluto_set_budget` | A monthly limit for one category. "Remove my food budget" deletes it. |
| "am I over budget?" | `pluto_budget_status` | Spending this month against each limit. |
| "what subscriptions am I paying for?" | `pluto_recurring_expenses` | Regular charges found in your log, with monthly cost. |
| "how much can I spend today?" | `pluto_safe_to_spend` | What is left per remaining day. |
| "my monthly income is 85000" | `pluto_set_income` | Needed for the health score and as a fallback for safe-to-spend. |
| "how healthy are my finances?" | `pluto_financial_health` | A 0-100 score with the parts shown. |
| "what if I invest 10000 a month for 15 years at 12 percent?" | `pluto_scenario_plan` | Projection with a plus-or-minus 3 point range. |
| "export my expenses for tax filing" | `pluto_export_tax` | Two CSVs in `data/pluto/exports/`. |

## How the planning features decide things

**Recurring charges.** The same payee (case, month names and years ignored) at least three times,
on different days, with gaps close to weekly, fortnightly, monthly, quarterly or yearly, and amounts
within 25% of each other. Two charges are not a pattern. A charge is "stopped" once it is more than
a cycle overdue; a latest amount more than 5% off the usual one is flagged as a price change.

**Budgets.** Used 80% or more is a warning; over the limit is "over". The heartbeat speaks one alert
per category per month for each state, never between 22:00 and 07:00 (held, not lost), and skips
the warning if you jumped straight past the limit.

**Safe to spend.** (Budget - spent this month - recurring bills still due this month) divided by the
days left including today. With category budgets it counts only those categories; with no budgets it
uses your monthly income against all spending.

**Health score.** Weighted average of whichever of these can be computed (at least two are needed):
savings rate against income (40%, full marks at 20% saved), steadiness of weekly spending (30%),
diversification of holdings by cost (30%, full marks at five equal holdings). The reply names any
part it could not compute. It is a rule of thumb, not financial advice.

**Scenario.** Contributions at the start of each month, monthly compounding at the rate equivalent to
the yearly figure, constant return. 1-60 years, return between -50% and 100%.

**Tax export.** Indian financial year (April to March), expenses and investments to CSV. The
`tax_hint` column only notes where a category often matters (health, education, ELSS/PPF/NPS).
It does not decide what is deductible. Cells starting with `= + - @` are prefixed with `'` so a
spreadsheet cannot run them as formulas.

## Known limits

* Dates are stored in UTC (SQLite `CURRENT_TIMESTAMP`), so an expense logged just after midnight in
  India can land on the previous day, and in rare cases the previous month.
* Budgets and recurring detection depend on categories the keyword matcher and the model assign.
* The score and scenarios are illustrations from simple formulas.
* Not built: broker sync (#131), price/news alerts (#132), multi-currency net worth (#136),
  forecast ranges (#139), explain-this-holding (#140), receipt photos (#141), rebalancing (#142).

## Code map

* `planning.py` the six features above; pure functions plus `PlanningManager`.
* `personal_finance.py` expenses, investments, spending report, currency conversion.
* `db.py` `PlutoDB` (SQLite: expenses, investments, budgets, settings, alerts_sent).
* `engine.py` routing; `check_budget_alerts()` is the heartbeat hook.
* Tests: `tests/test_pluto_planning.py`, `tests/test_pluto_planning_property.py`, `tests/test_pluto.py`.
