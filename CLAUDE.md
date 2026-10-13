# CLAUDE.md

Operating rules for this repository. `README.md` maps the code and says where each line of
work stands. `CLAUDE_CODE_BRIEFING.md` has project background and measured findings.

## Autonomy

**Work unattended. Do not ask questions or wait for approval** unless the user is plainly
present and the question changes what you do next. Sessions are often started and checked
hours later.

- Where a task leaves something open, decide it yourself from the measured findings, and
  **log the decision** (below).
- Never pause for confirmation before a long run. Start it.
- If something fails, diagnose and work around it. If a whole stage is impossible, skip it,
  log why, and continue.
- If a result contradicts the task's assumptions, follow the evidence rather than the
  document, and log the divergence prominently.

The one thing to stop for: if continuing would **destroy measured data** — overwriting a
cache, deleting a `runs_*/` directory — don't. Work around it and log it.

## Decision log

Append to `DECISIONS.md` whenever you make a choice a task document did not specify.

```
## [stage] short title
**Chose:** what you did
**Over:** the alternative(s)
**Because:** the reasoning, citing measured data where relevant
**Reversible:** yes/no - what would need re-running to change it
```

Keep it terse. This is the first thing the user reads on returning.

## Python and MATLAB

Use `/opt/anaconda3/bin/python`. `matlab.engine` is installed there and nowhere else.

```bash
/opt/anaconda3/bin/python -c "import matlab.engine; print('ok')"
```

**Run from the repo root.** `bgn_vars.mat` is written to the working directory by MATLAB and
read back by Python, and `start_engine` resolves `matlab/` and `gating/` against the cwd.

**Simulations must run serially.** `bgn_vars.mat` is one shared file rewritten every step, so
concurrent MATLAB engines corrupt each other.

Prefer scripts over notebooks for long runs, and use `python -u` so output streams.

**Restart the kernel after editing any `.py` file.** `%autoreload` leaves objects bound to
stale classes and produces `AttributeError` on methods you can see in the file.

## Crash resilience

Sessions run for hours. Assume interruption.

- Cache every simulation result **as you go**, not at stage end.
- Never re-run a cached measurement. Check the cache first, always.
- Write partial results to the report file as each stage completes, so an interrupted session
  still leaves something readable.

## Statistics

Per-pull reward noise is roughly ±0.02 — but **check it for the arm you are using**. At the
155 Hz / 450 µA probe arm it is 0.0068, three times tighter, and that is what makes a short
probe sufficient. Assuming the global figure where it does not apply wastes simulations.

- n ≥ 10 behind any reported number, with a standard error.
- Never report a difference smaller than its error bar as a finding.
- Several findings in this project have been retracted after failing at higher n or after a
  measurement was corrected. **State null results plainly** rather than hunting for a framing
  that makes them look positive. Both results documents record their retractions as
  prominently as their findings; keep that standard.

## Known traps

Each of these cost real debugging time. Do not rediscover them.

- **Grid-edge optima.** This project has produced an apparent optimum sitting on the boundary
  of the search grid at least five times — 1000 µA, 130 Hz, 80 Hz, 180 Hz, and the selector's
  frequency result, which was retracted when extending the grid moved the answer with it.
  After any selection, check whether the chosen arm sits at a grid edge in either dimension and
  flag it. A metric that improves when you widen the search is measuring the search.
- **`dbs_t3p.CURR_VALUES` starts at 1000 µA**, skipping the 300–550 µA region where the useful
  settings are. Check which grid a result came from before quoting it.
- **Biomarker:** use `calculate_pb` / `normalize_pb`. The `*_absolute` variants separate healthy
  from PD by only 1.24× against 2.83× and make severities look nearly identical.
- **Units:** `bgn_step` takes index units, not milliseconds. `dt = 0.01` ms, so the third
  argument is `tmax * 100`.
- **Window length:** `calculate_pb` sums FFT **bin indices** 13:35, which only correspond to
  13–35 Hz at a 1000 ms window. At 500 ms the same function measures 26–70 Hz.
- **Dtype:** cast MATLAB arrays to float64 before arithmetic. `dbs ** 2` in the native integer
  dtype overflows, wraps negative, and silently corrupts RMS.
- **Seeding:** `bgn_init(pd, tmax, seed)` reseeds every call, so every simulation is a different
  brain. Single measurements mean nothing.
- **Two different rewards exist in the tree.** `dbs_t3p` uses the published three-term reward on
  beta power: `-0.7*r1(P_beta) + 0.1*r3(no-stim) - 0.2*r2(RMS)`. `dbs_core` uses a two-term
  reward on `sgis`. Numbers from the two paths are not comparable.
- **Saturation above ~350 µA** makes several arms statistically indistinguishable. Report tied
  sets rather than a single best arm.

## Files

**Live.** `dbs_t3p.py` is the bandit and the published reward. `dbs_selector.py` and
`run_selector.py` are the minimum-effective-dose selector with severity inference.
`neural_*.py` is the per-band warm-start. `pd_switch.py`, `experiments.py` and
`recheck_stim.py` are the change-detection work. Notebooks `01_`–`04_` run in that order.

**Superseded but documented.** `dbs_core.py` holds the live environment and state features
*plus* the retired TD3 agent — read its header, the file is half and half.
`dbs_reward_model.py` and `dbs_plots.py` belong to approaches that were replaced. They are
kept and commented because the reasons they were dropped are themselves findings. Prefer
`dbs_t3p` for new work, but these are not off-limits.

**Do not edit** `helpers.py` — biomarker and reward primitives shared by everything.

**Never edit** `bgn_vars.mat` — MATLAB state, rewritten every step.

`archive/` holds the dead halves of the superseded notebooks. Nothing live imports from it.

## Caches

`runs_selector/`, `runs_neural/` and the detection caches are **committed**, because they are
small and expensive to regenerate — the selector's alone is about 2.5 hours of MATLAB. Check
for a cache before running anything.
