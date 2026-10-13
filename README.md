# DBS Controller Selection

Patient-specific stimulation control for deep brain stimulation, on a Rubin–Terman style
basal-ganglia–thalamic model in MATLAB. Severity is a parameter `pd` from 0 (healthy) to 1
(severe).

The question across all of it: a patient's symptom severity is not fixed, so can we choose
the right stimulation for the severity they are in right now — and notice when it changes?

---

## Where the work stands

Four lines, three of them live. Each has its own results document with the numbers.

| line | status | code | results |
|---|---|---|---|
| **Controller selection** | working | `dbs_selector.py`, `run_selector.py` | `SELECTOR_RESULTS.md` |
| **Neural warm-start** | working | `neural_*.py` | `NEURAL_RESULTS.md` |
| **Change detection** | working | `pd_switch.py`, `experiments.py`, `recheck_stim.py` | `detection_results.ipynb` |
| **TD3 / reward models** | superseded | `dbs_core.py`, `dbs_reward_model.py` | this README, below |

**Controller selection** picks stimulation by minimising power subject to a floor on symptom
control, with a hard safety constraint, and infers severity from a 155 Hz probe rather than
assuming it is known. 2,353 simulations. Amplitude increases with severity; the harmful
sub-threshold band is real at 33σ; 8 probes are enough to infer severity well enough to
prescribe correctly 4/4.

**Neural warm-start** trains reward-predicting networks so T3P can skip its warm-up: 51.8 ± 1.7
simulations to converge cold against 6.2 ± 2.0 warm, an 8.3× speedup. Band specialisation is
real (diagonal R² 0.98 vs off-diagonal 0.05) but a single pooled network matches the
specialists, so severity mattering is not the same as needing one network per severity.

**Change detection** finds the moment severity changes mid-simulation, from beta-band power,
while a controller is running. 46/50 brains across stimulation amplitudes from 0 to 3500 µA,
roughly two seconds to notice, with a measured false-positive rate of 1/20. A changepoint test
beats k-means clustering decisively under stimulation — clustering fires on 7 of 10 runs where
nothing happened.

Both results documents record **retractions** as prominently as findings. Read them.

---

## Layout

```
dbs_t3p.py          T3P bandit — the live selection mechanism, and the published reward
helpers.py          biomarker and reward primitives. DO NOT EDIT — shared by everything

dbs_selector.py     minimum-effective-dose rule, severity inference, safety constraint
run_selector.py     driver for the selector's MATLAB stages

neural_t3p.py       per-band reward networks, data collection, training
neural_run.py       driver for the neural experiments
neural_gates.py     validation gates, no MATLAB, runs in seconds
neural_report.py    builds NEURAL_RESULTS.md from whatever is cached
neural_extra.py     pooled-network and boundary follow-ups

pd_switch.py        mid-simulation severity switch, sliding beta power, two detectors
experiments.py      detection under stimulation, severity ladders, adaptive bandit
recheck_stim.py     pair-averaged re-analysis of cached detection runs

dbs_core.py         Config, MatlabEnv, MockEnv, Hjorth features — plus the retired TD3 agent
dbs_checks.py       pre-flight checks: offline, live, calibration, fixed-policy baseline
dbs_reward_model.py reward-model approach, superseded; its datasets are still used
dbs_plots.py        TD3 training plots, superseded with it

matlab/             bgn_init.m, bgn_step.m, set_pd.m
gating/             28 gating-variable functions called by bgn_step
datasets/           400 sampled rows per severity, ~35 min of MATLAB each
runs_selector/      selector caches — small JSON, hours to regenerate, committed
runs_neural/        neural caches — same
runs/               TD3 checkpoints from the retired approach
```

Every module carries a header saying its status and what it holds. Start there.

---

## Requirements

- MATLAB with the Python engine (`matlab.engine`)
- Python 3.12, PyTorch, NumPy, SciPy, matplotlib
- Use `/opt/anaconda3/bin/python` — `matlab.engine` is installed there and nowhere else

```bash
/opt/anaconda3/bin/python -c "import matlab.engine; print('ok')"
```

---

## Running it

**Run from the repo root.** `bgn_vars.mat` is written to the working directory by MATLAB and
read back by Python. Launch Jupyter from the root, or pass an explicit path.

**Simulations must run serially.** `bgn_vars.mat` is a single shared file rewritten every
step; concurrent MATLAB engines corrupt each other.

**Restart the kernel after editing any `.py` file.** Do not rely on `%autoreload` — it leaves
objects bound to stale classes and produces `AttributeError`s on methods that visibly exist.

```bash
# selector, each stage cached independently
python -u run_selector.py --stage probe --n-probe 15      #  ~6 min
python -u run_selector.py --stage arms  --n 10            # ~49 min
python -u run_selector.py --report --tau -0.20            # no simulation

# neural, gates first
python neural_gates.py
python -u neural_run.py --stage collect

# detection — detection_results.ipynb runs top to bottom
```

---

## Units and conventions

These caused real bugs. Do not change without re-reading.

- `dt = 0.01` ms. The simulation index advances 100 units per millisecond.
- **`bgn_step` takes index units, not milliseconds.** The third argument is `tmax * 100`.
- `sim_time_per_step = 100` means 100 ms; `cfg.window_samples()` converts it to the 10,000
  index units that slicing needs.
- **Cast MATLAB arrays to float64 before arithmetic.** `dbs ** 2` in the native integer dtype
  overflows, wraps negative, and silently corrupts RMS.
- `bgn_init(pd, tmax, seed)` **reseeds on every call**, so every simulation is a different
  brain. A single measurement means nothing.
- `calculate_pb` sums FFT **bin indices** 13:35, not frequencies. Those bins only correspond
  to 13–35 Hz at a 1000 ms window. At 500 ms the same function measures 26–70 Hz.
- Use `calculate_pb` / `normalize_pb`. The `*_absolute` variants separate healthy from PD by
  only 1.24× against 2.83× and make severities look nearly identical.
- Hjorth features use **per-neuron** variances, and the bounds in `Config` were measured with
  that formula. Changing the convention invalidates those bounds.

---

## Known traps

**Grid-edge optima.** This project has produced an apparent optimum sitting on the boundary of
the search grid at least five times — 1000 µA, then 130 Hz, then 80 Hz, then 180 Hz, and the
selector's frequency result, which was retracted after extending the grid moved the answer
with it. After any selection, check whether the chosen arm sits at a grid edge in either
dimension, and flag it if so. A quality metric that improves when you widen the search is
measuring the search, not the controller.

**The default current grid in `dbs_t3p` starts at 1000 µA**, skipping the 300–550 µA region
where the useful settings actually are. The notebooks override it. Check before trusting a sweep.

**Two different rewards are in the tree.** `dbs_t3p` uses the published three-term reward on
beta power; `dbs_core` uses a two-term reward on `sgis`. Numbers from the two are not comparable.

**The reward surface saturates above ~350 µA**, so several arms become statistically
indistinguishable and "reached the same arm" is an unstable target. Report tied sets, not
single arms.

**Per-pull noise is not uniform.** The project assumed ±0.02 throughout; at the 155 Hz probe
arm it is 0.0068, three times tighter. That is what makes a short probe sufficient.

---

## Statistics

n ≥ 10 behind any reported number, with a standard error. Never report a difference smaller
than its error bar. Three findings in this project were retracted after failing at higher n,
and two more were retracted in `SELECTOR_RESULTS.md` after the measurement was corrected.
State null results plainly rather than hunting for a framing that makes them look positive.

---

## Open questions

- **Where the floor τ sits** for the selector. It decides how much biomarker control is traded
  for power, and nothing measured here fixes it.
- **Frequency cannot be chosen by the current objective.** Minimum power always walks to the
  lowest frequency in the grid. It needs either clinical reasoning or a different constraint.
- **Channel combination.** Hjorth features and beta power are computed per neuron and then
  averaged; a real lead records one combined field. The two are not interchangeable — averaging
  per-neuron statistics is blind to synchrony, which is the hallmark of the PD state.
- **Does the model's amplitude parameter correspond to real current?** It is labelled in µA but
  whether the stimulation term is calibrated to clinical currents is unverified.
- **Mode B is internally consistent, not cross-validated.** Probe calibration and arm
  measurements come from the same simulator.
