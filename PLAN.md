# Shelf Space Allocator: build plan for Claude Code

Goal: a small, polished, LIVE demo at **https://shelf.manarattar.com** that shows
a junior data scientist's end-to-end skills on a retail commerce problem
(inspired by Albert Heijn's "which products get shelf space in each store").
Someone should understand it in 3 minutes and see real engineering behind it.

Owner: Manar (he/him). Repo: `manarattar/shelf-space` (his own GitHub, never anyone else's).
Working dir: `E:\Manar\claude_projects\shelf-space`.

## Principles
- **Simple to go through.** One page, three steps, one number that matters (expected uplift with a confidence interval).
- **Honest.** Public data has no real shelf-space data, so the space-to-sales response is a stated ASSUMPTION. The app and README say so plainly. Never present simulated results as real-world findings.
- **Packages, not notebooks.** Installable Python package with tests and CI. Notebooks only for exploration, if at all.
- **Baselines first.** Every model is reported against a naive baseline.
- **Low memory.** Manar's PC has 7.8 GB RAM. Run Spark in local mode with a capped driver memory, use a SUBSET of the data (see below), and no long background loops.
- **No LLM features.** This demo shows classical data science rigour.

## Read before starting
- Memory: `C:\Users\manar\.claude\projects\C--Users-manar\memory\project_contabo_vps.md`, `reference_dns_vercel.md`, `project_claude_projects_deploy.md` (deploy and DNS conventions).
- Look at one existing deployed project (e.g. `argus-lite` or `workflow-studio`) and copy its Docker + Caddy + deploy pattern instead of inventing a new one.

## Data
- **M5 Walmart forecasting** (Kaggle): `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`.
- Manar must download it himself (needs his Kaggle login). Ask him to place the files in `data/raw/`. Do not try to log in to Kaggle or create accounts.
- **Subset:** 3 stores from one state, 3 categories (FOODS, HOUSEHOLD, HOBBIES), about 300 items total, last ~2 years. Keep processed data under ~50 MB so it ships with the repo or the container.
- **Fallback if M5 is not available yet:** `scripts/make_synthetic.py` generates a realistic synthetic retail dataset with the same schema (weekly seasonality, yearly seasonality, price effects, promotions). Build everything against this first so work is never blocked. State in the README which data a given deployment uses.

## Architecture (keep it small)
```
shelf-space/
  pyproject.toml
  README.md
  data/raw/            (gitignored)   data/processed/ (small parquet, committed)
  src/shelfspace/
    data.py            PySpark: load M5, clean, build features -> parquet
    forecast.py        seasonal-naive baseline + LightGBM, WAPE / MASE
    response.py        shelf-space -> sales response curves (ASSUMED, documented)
    optimise.py        space allocation under constraints (scipy or cvxpy)
    experiment.py      simulated A/B test: power analysis, uplift, CI
  app/streamlit_app.py the one-page demo
  tests/               pytest for each module
  .github/workflows/ci.yml
  Dockerfile, docker-compose.yml (or the pattern used by other projects)
```
Heavy work (Spark + training) runs OFFLINE once and writes small result files.
The live app only loads those files and runs the fast optimiser. No Spark in production.

## Phases (finish each, run tests, commit, then continue)

### Phase 1: Data + forecast (Spark)
- PySpark local mode (`spark.driver.memory=3g`): load, melt wide sales to long, join calendar + prices, create features (lags, rolling means, price, day of week, event flags).
- Write features to parquet.
- Forecast: seasonal naive baseline, then LightGBM. Hold out the last 28 days.
- Report **WAPE and MASE** vs the baseline, per category. Save metrics to `results/metrics.json`.
- Tests: feature builder on a tiny fixture, metric functions on known values.

### Phase 2: Shelf-space response and optimiser
- `response.py`: per category, sales = demand * f(space) with diminishing returns (e.g. `f(s) = s**alpha`, alpha per category, clearly labelled assumption, configurable).
- `optimise.py`: per store, allocate a fixed total number of facings across products. Constraints: min 1 facing, max N per product, total budget. Objective: maximise expected sales (or margin if price available). Use scipy or cvxpy, deterministic.
- Compare against the "current" allocation (proportional to historical sales share or equal split).
- Tests: constraints always satisfied, optimiser never worse than the baseline on the objective.

### Phase 3: Simulated experiment
- `experiment.py`: split stores into control/treatment, simulate N weeks with noise, compute uplift, 95% CI, p-value, and a power calculation (how many store-weeks are needed).
- Output: uplift +x% (95% CI a to b), and a plain sentence on whether it is significant.
- Tests: with zero true effect the false-positive rate is about 5% (seeded).

### Phase 4: The demo app (one page, three steps)
Streamlit, simple layout, light theme consistent with manarattar.com:
1. **Pick** a store and a category (sidebar or top selects).
2. **See** the forecast accuracy vs baseline (one small chart + the WAPE number) and a side-by-side bar chart: current vs recommended facings per product.
3. **Result** card: expected uplift with CI, and the power-analysis sentence.
Plus:
- A slider for total shelf budget (re-runs the optimiser live, it is fast).
- A clearly visible "Assumptions and limits" expander (response curve is simulated, public data, not AH data).
- A short "How it works" section linking to the repo.
- No login, no accounts, no external API calls.

### Phase 5: Package, CI, docs
- `pyproject.toml` installable with `pip install -e .`.
- GitHub Actions: lint (ruff), pytest, and a tiny quality gate (forecast must beat the baseline on the fixture data).
- README: problem, approach, results table, screenshot, honest limitations, "what I would do with real planogram data", run instructions.
- Add a one-page `docs/for-category-managers.md`: the result explained without jargon.

### Phase 6: Deploy live
- Dockerise the Streamlit app (small image, only runtime deps, processed data baked in).
- Deploy to the Contabo VPS under `/srv/stack` following the existing pattern (Caddy reverse proxy, docker compose). Use the same conventions as the other projects' compose services and keep memory use small (set a container memory limit around 512 MB).
- DNS: add the `shelf` A record for manarattar.com to `194.163.176.183` via the Vercel CLI (per memory `reference_dns_vercel.md`). Do not ask Manar to do it.
- Verify: `https://shelf.manarattar.com` loads over HTTPS, all three steps work, the budget slider responds, no errors in container logs. Use a screenshot or browser check of the live URL, not just a 200.
- Push the repo to `manarattar/shelf-space` (public) with CI green.

### Phase 7: Show it on the portfolio
- Add a project card to the portfolio (`E:\Manar\claude_projects\portfolio`) following how other cards are done: title "Shelf Space Allocator", one-line description, tech tags (PySpark, LightGBM, optimisation, A/B testing), links to live demo and repo.
- Commit, and deploy the portfolio the usual way (see `project_claude_projects_deploy.md`; Vercel does not auto-deploy).
- Update the Claude Projects memory index entry.

## Definition of done
- Live at https://shelf.manarattar.com, working, fast (<3 s per interaction).
- Repo public, CI green, README with screenshot and honest limitations.
- Forecast beats the naive baseline and the result is reported with the metric names (WAPE, MASE).
- Package installs cleanly and tests pass from a fresh clone.
- Portfolio card live.
- The demo is understandable by a non-technical person in under 3 minutes.

## Do not
- Do not invent real-world claims (no "increases AH sales by X%").
- Do not create accounts (Kaggle, Databricks, etc.). Ask Manar for files or use the synthetic fallback.
- Do not push to any repo Manar does not own.
- Do not use the Hajzi VPS. This project goes on Contabo only.
- Do not add Spark to the production container.
- Do not leave notebooks as the main deliverable.

## Optional stretch (only if everything above is done)
- Run the Spark job on Azure Databricks (Azure for Students) and note it in the README as the production path.
- Add category-level margin instead of sales as the optimisation objective.
