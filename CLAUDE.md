# CLAUDE.md

Operating rules for this workspace. Read `NEURAL_T3P_TASK.md` for the current task and
`claude-briefing.md` for project background and measured findings.

## Autonomy

**Work unattended. Do not ask questions or wait for approval.** The user starts a session and
returns hours later expecting results.

- Where the task document leaves something open, decide it yourself using the measured
  findings as evidence, and **log the decision** (see below).
- Never pause for confirmation before a long run. Start it.
- If something fails, diagnose and work around it. If a whole stage is impossible, skip it,
  log why, and continue to the next stage.
- If a result contradicts the task's assumptions, follow the evidence rather than the
  document, and log the divergence prominently.

The one thing to stop for: if continuing would **destroy measured data** (overwriting a
cache, deleting `runs_*/`), don't. Work around it and log it.

## Decision log

Maintain `DECISIONS.md` at the workspace root. Append an entry whenever you make a choice the
task document did not specify. Format:

```
## [stage] short title
**Chose:** what you did
**Over:** the alternative(s)
**Because:** the reasoning, citing measured data where relevant
**Reversible:** yes/no - what would need re-running to change it
```

Keep it terse. This is the first thing the user reads on returning.

## Budget

**Hard cap: 4 hours of MATLAB simulation time** for the whole session. One simulation is
about 3.9 seconds, so roughly 3,700 simulations.

Track your spend and report it. If the full plan exceeds budget, cut scope in this priority
order and log the cut:

1. Data collection for training (required - everything depends on it)
2. Experiment 1, cross-band prediction accuracy (cheap, uses cached data)
3. Experiment 2, warm-start convergence speedup (the headline claim)
4. Experiment 3, wrong-band penalty

Reduce repeats or the number of pd values before dropping a whole experiment.

## Crash resilience

Sessions run for hours. Assume interruption.

- Cache every simulation result to `runs_neural/` **as you go**, not at stage end.
- Never re-run a cached measurement. Check the cache first, always.
- Write partial results to the report file as each stage completes, so an interrupted session
  still leaves something readable.
- Use `python -u` so output streams rather than buffering.

## Python

Use `/opt/anaconda3/bin/python`. `matlab.engine` is installed there and nowhere else.

```bash
/opt/anaconda3/bin/python -c "import matlab.engine; print('ok')"
```

Run everything as scripts, not notebooks. Simulations must run **serially** - `bgn_vars.mat`
is a single shared file in the working directory, so concurrent MATLAB engines corrupt each
other.

## Statistics

Per-pull reward noise is roughly **plus/minus 0.02**.

- n >= 10 behind any reported number, with a standard error.
- Never report a difference smaller than its error bar as a finding.
- Three findings in this project were retracted after failing at higher n. State null results
  plainly rather than hunting for a framing that makes them look positive.

## Known traps

Each of these cost real debugging time. Do not rediscover them.

- **Biomarker:** use `calculate_pb` / `normalize_pb`. The `*_absolute` variants separate
  healthy from PD by only 1.24x versus 2.83x and make severities look identical.
- **Units:** `bgn_step` takes index units, not milliseconds. `dt = 0.01` ms, so the third
  argument is `tmax * 100`.
- **Dtype:** cast MATLAB arrays to float64 before arithmetic. `dbs ** 2` in the native integer
  dtype overflows and wraps negative, silently corrupting RMS.
- **Reward weights** follow the paper: `-0.7*r1(P_beta) + 0.1*r3(no-stim) - 0.2*r2(RMS)`.
- **Seeding:** `bgn_init(pd, tmax, seed)` reseeds every pull, so every simulation is a
  different brain realization. Single measurements mean nothing.
- **Boundary results:** this project has three times produced an "optimum" that was really the
  edge of the search grid (1000 uA, then 130 Hz, then 80 Hz). After any selection, check
  whether the chosen arm sits at a grid edge in either dimension and flag it if so.

## Files

`dbs_t3p.py` is the live module - `T3PBandit`, `MatlabBackend`, `compute_reward`.
`helpers.py` holds biomarker and reward primitives; do not edit it.
`dbs_selector.py` has the lexicographic rule from a previous task - reusable but not required.

`dbs_core.py` (TD3) and `dbs_reward_model.py` are abandoned approaches. Do not build on them.

`bgn_vars.mat` is MATLAB state, rewritten every step. Never edit it.