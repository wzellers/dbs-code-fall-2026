# DBS Controller Selection

Selecting patient-specific stimulation controllers for deep brain stimulation, using the
T3P multi-armed bandit as the selection mechanism.

## Project goal

Train separate controllers at different Parkinson's severities, then use T3P to select the
right controller for a given patient. `pd = 0` is a healthy brain; severity scales upward
from there. Milder patients should end up with less aggressive stimulation.

## Requirements

- MATLAB with the Python engine installed (`matlab.engine`)
- Python 3.12, PyTorch, NumPy, SciPy, matplotlib
- The BGT model `.m` files must be on the MATLAB path

## Layout

```
dbs_core.py           Config, TD3 agent, MatlabEnv/MockEnv, training loop, checkpointing
dbs_checks.py         Verification (offline + live), calibration, fixed-policy baselines
dbs_plots.py          Training, evaluation, and run-comparison figures
dbs_reward_model.py   Uniform data collection, supervised fitting, cross-evaluation
dbs_t3p.py            T3P bandit selector (in progress)

matlab/               BGT model — bgn_init.m, bgn_step.m and dependencies
notebooks/            Driver notebooks
datasets/             Collected samples (expensive to regenerate — keep, don't commit)
runs/                 Training outputs and checkpoints
```

## Important: run from the repo root

`bgn_vars.mat` is written to the working directory by MATLAB and read back by Python. Launch
Jupyter from the repo root, or pass an explicit path:

```python
menv = MatlabEnv(cfg, mat_path="/full/path/to/bgn_vars.mat")
```

## Units and conventions

These caused real bugs. Do not change without re-reading this section.

- `dt = 0.01` ms. The simulation index `i` advances 100 units per millisecond.
- `sim_time_per_step = 100` means **100 ms**. `cfg.window_samples()` converts it to the
  10,000 index units that `bgn_step` and array slicing actually need. **`bgn_step` takes
  index units, not milliseconds.**
- An episode is `steps_per_episode * window_samples() == tmax / dt`. With the defaults:
  15 x 10000 = 150000 = 1500 / 0.01. If that identity breaks, something is wrong.
- Cast MATLAB arrays to `float64` before arithmetic. `dbs ** 2` in the native integer dtype
  overflows and wraps negative, silently corrupting RMS.
- Hjorth features use **per-neuron** variances (matching the calibration formula). The
  bounds in `Config` were measured with that formula.

## Workflow

Always in this order. Skipping the checks has cost days.

1. `dbs_checks.versions()` — confirms all modules are the same version. Stale files fail
   loudly here rather than mysteriously later.
2. `dbs_checks.offline(hidden_dim)` — 13 checks, no MATLAB, ~1 min.
3. `dbs_checks.live(env, cfg)` — 11 checks against the real model, ~5 min. **Never start a
   long run if these fail.**
4. `dbs_checks.baseline(env, cfg)` — fixed-policy results. This is the number any learned
   controller has to beat.
5. Short run (~30 episodes) to confirm the loop survives.
6. Full run.

## After editing a `.py` file

Restart the kernel. Do not rely on `%autoreload` — it leaves existing objects bound to stale
classes, which produces confusing `AttributeError`s on methods that visibly exist in the file.

## Current state

**Working:** environment verified, 11 live checks passing at pd = 0.3 / 0.6 / 1.0. Two
independent measurement methods agree on the optimum (~177 Hz / 1000 µA).

**Established:** network capacity does not affect performance (401 to 17,537 parameters
perform identically). Controller specialization by severity is real — cross-evaluation R²
degrades with severity distance.

**Not working:** TD3 lost to fixed open-loop DBS in 6/6 runs, caused by a missing uniform
random exploration phase (the untrained actor never samples below ~1600 µA, where the
optimum lies). The reward model located the optimum in its data but could not resolve it
from 400 uniform samples.

**Next:** grid-based collection with repeats, then T3P as the selector over per-severity
controllers.

## Known cost

- TD3 training: ~70 min per run
- Data collection: ~35 min per severity, reusable across any number of architectures
- Fitting a reward model: seconds