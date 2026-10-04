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
| "forecast my spending" | `pluto_forecast_spending` | Next 7 days with an 80% range for each day and the total. |
| "set my target allocation to 60 stocks, 30 mutual funds, 10 crypto" | `pluto_set_target_allocation` | Saves targets (by asset class or by holding). "Clear my target allocation" removes them. |
| "do I need to rebalance?" / "rebalance with 20000 new money" | `pluto_rebalance_portfolio` | Drift from your targets; what to sell and buy, or a buy-only split of new money. |
| "explain my Infosys holding" | `pluto_explain_holding` | Your position, price behaviour, headlines, fundamentals. |
| "why is the quant score for TCS what it is" | `pluto_explain_quant_score` | The parts that make up the score. |
| "try different moving average windows on Reliance" | `pluto_backtest_sweep` | Grid of fast/slow windows with equity curves (also in the Portfolio tab). |
| "log this receipt /path/photo.jpg" | `pluto_log_receipt` | Reads the photo and logs the total. |
| "is the market data being rate limited?" | `pluto_data_source_status` | Requests, 429s, cooldowns and retry settings per source. |

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

**Forecast range.** The model is re-fitted on a growing window and asked to predict each of the last 14
days it had not seen; the root-mean-square of those misses is the typical one-day error. A day's range is
the prediction plus or minus 1.28 times that error, widened 10% for each day ahead (a rule of thumb). The
total's range adds the daily errors as if independent. On synthetic data the 80% range covered about 90%.
Spending is lumpy, so treat it as a rough band.

**Rebalancing.** Targets must add up to 100 and cover everything you hold; a held class with no target is an
error, never "sell it all". Holdings are valued at the live price, or at cost when none is available (the reply
says which). Buy-only mode spreads new money over under-weight areas by their shortfall and says how far that got.
Ignores tax, brokerage and lot sizes.

**Receipts.** A total is trusted only on a line labelled like a total (grand total, amount payable, total...);
subtotal, tax, discount and change lines never count. Without one it shows what it read and asks. The same
amount and shop on the same day is reported as a likely duplicate. Needs `pytesseract`, `pillow` and the
Tesseract program.

**Throttling.** An HTTP 429 from Yahoo or CoinGecko is not retried. It starts a cooldown (the server's
Retry-After, up to 15 minutes, else 60 seconds); during it Yahoo history calls fail at once without a request.
Counts reset when Pluto restarts.

**Backtest sweep.** Up to 8 fast and 8 slow windows (64 pairs, fast below slow), one price download. Uses its
own simulator (long only, all-in, 0.1% fee per side), not vectorbt. The best cell is best on the prices it was
tested on; read the median and how many cells beat buy-and-hold.

## Known limits

* Dates are stored in UTC (SQLite `CURRENT_TIMESTAMP`), so an expense logged just after midnight in
  India can land on the previous day, and in rare cases the previous month.
* Budgets and recurring detection depend on categories the keyword matcher and the model assign.
* The score and scenarios are illustrations from simple formulas.
* Not built: broker sync (#131), price/news alerts (#132), multi-currency net worth (#136).
* Not tested against the live services: Yahoo news and quote-summary, SEC EDGAR, real Tesseract OCR.
* Fundamentals for Indian tickers rely on a Yahoo cookie-and-crumb handshake that Yahoo can change.
* No web upload for receipts and no web view for the quant-score breakdown yet (chat and JSON only).
* Rebalancing and the sweep are in one currency; mixed currencies are not handled.

## Code map

* `planning.py` the six features above; pure functions plus `PlanningManager`.
* `personal_finance.py` expenses, investments, spending report, currency conversion.
* `throttle.py`, `rebalance.py`, `receipts.py`, `holding_explainer.py`, `explain.py` the #139-#145 features;
  `forecasting.py` and `backtest.py` gained ranges and the sweep.
* `db.py` `PlutoDB` (SQLite: expenses, investments, budgets, settings, alerts_sent).
* `engine.py` routing; `check_budget_alerts()` is the heartbeat hook.
* Tests: `tests/test_pluto_extras.py` (201), `tests/test_pluto_planning.py`, `tests/test_pluto_planning_property.py`, `tests/test_pluto.py`.
