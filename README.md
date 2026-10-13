# DBS Controller Selection

Patient-specific stimulation control for deep brain stimulation, on a Rubin–Terman style
basal-ganglia–thalamic model in MATLAB. Severity is a parameter `pd` from 0 (healthy) to 1
(severe).

The question across all of it: a patient's symptom severity is not fixed, so can we choose the
right stimulation for the severity they are in right now — and notice when it changes?

---

## Start here

1. **`CLAUDE.md`** — operating rules: how to run things, known traps, statistical standards.
   Read before running anything, whether you are a person or an agent.
2. **This file's layout section**, below, for where everything lives.
3. **`docs/SELECTOR_RESULTS.md`**, **`docs/NEURAL_RESULTS.md`** and
   **`docs/DETECTION_RESULTS.md`** — the measured results, one per line of work. All three
   record retractions as prominently as findings; that is deliberate, and the retractions are
   often more informative than the findings.
4. **Notebooks `01`–`04`**, in order, for the detection work and the bandit's behaviour.

Every module in `src/` opens with a header giving its status, what it contains, and the traps
specific to it. If you are wondering whether a file is live, that header says so.

---

## Where the work stands

Four lines, three of them live.

| line | status | code | results |
|---|---|---|---|
| **Controller selection** | working | `src/dbs_selector.py`, `src/run_selector.py` | `docs/SELECTOR_RESULTS.md` |
| **Neural warm-start** | working | `src/neural_*.py` | `docs/NEURAL_RESULTS.md` |
| **Change detection** | working | `src/pd_switch.py`, `src/experiments.py`, `src/recheck_stim.py` | `docs/DETECTION_RESULTS.md` |
| **TD3 / reward models** | superseded | `src/dbs_core.py`, `src/dbs_reward_model.py` | module headers, and below |

**Controller selection** picks stimulation by minimising power subject to a floor on symptom
control, with a hard safety constraint, and infers severity from a 155 Hz probe rather than
assuming it is known. 2,353 simulations. Amplitude increases with severity; the harmful
sub-threshold band is real at 33σ; 8 probes are enough to prescribe correctly 4/4 without ever
reading the severity label.

**Neural warm-start** trains reward-predicting networks so T3P can skip its warm-up: 51.8 ± 1.7
simulations to converge cold against 6.2 ± 2.0 warm, an 8.3× speedup. Band specialisation is
real (diagonal R² 0.98 against off-diagonal 0.05) but a single pooled network matches the
specialists — so severity mattering is not the same as needing one network per severity.

**Change detection** finds the moment severity changes mid-simulation, from beta-band power,
while a controller is running. 46/50 brains across stimulation amplitudes from 0 to 3500 µA,
about two seconds to notice, with a measured false-positive rate of 1/20. A changepoint test
beats k-means clustering decisively under stimulation — clustering fires on 7 of 10 runs where
nothing happened at all.

**TD3** was dropped because its untrained actor only proposed amplitudes between roughly 1600
and 3500 µA, so it never sampled the low-amplitude region where the good settings turned out to
be. It trained on half the action space and lost to fixed open-loop stimulation. The
**reward-model** approach that replaced it covered the whole space, but its models almost all
selected the same action regardless of training severity — real specialisation in reward
prediction that never reached the control decision.

---

## Layout

```
README.md                  this file
CLAUDE.md                  operating rules — read before running anything
requirements.txt

01_t3p_sweeps.ipynb        T3P arm sweeps and runs across severities
02_pd_switch.ipynb         the mid-simulation severity switch, detected unstimulated
03_experiments.ipynb       detection under stimulation, severity ladders, adaptive bandit
04_detection_results.ipynb the conclusive detection run — both detectors, frozen thresholds

src/                       all Python. Nothing here runs from inside this directory
  paths.py                 where everything lives on disk. Import this, never hardcode
  helpers.py               biomarker and reward primitives. DO NOT EDIT — shared by all
  dbs_t3p.py               T3P bandit — the live mechanism, and the published reward

  dbs_selector.py          minimum-effective-dose rule, severity inference, safety band
  run_selector.py          driver for the selector's MATLAB stages

  neural_t3p.py            per-band reward networks, data collection, training
  neural_run.py            driver for the neural experiments
  neural_gates.py          validation gates — no MATLAB, runs in seconds
  neural_report.py         builds docs/NEURAL_RESULTS.md from whatever is cached
  neural_extra.py          pooled-network and boundary follow-ups

  pd_switch.py             the severity switch, sliding beta power, both detectors
  experiments.py           detection under stimulation, ladders, adaptive bandit
  recheck_stim.py          pair-averaged re-analysis of cached detection runs

  dbs_core.py              Config, MatlabEnv, MockEnv, Hjorth features — plus retired TD3
  dbs_checks.py            pre-flight checks: offline, live, calibration, baseline
  dbs_reward_model.py      reward-model approach, superseded; its datasets are still used
  dbs_plots.py             TD3 training plots, superseded with it

docs/                      results and the task documents they were carried out against
  SELECTOR_RESULTS.md      the selector: 2,353 simulations, two retractions
  NEURAL_RESULTS.md        the warm-start: 3,482 simulations, three experiments
  DETECTION_RESULTS.md     the change detection: ~280 simulations, four retractions
  NEURAL_T3P_TASK.md       the specification the neural work was done against (completed)
  DECISIONS.md             choices those task documents left open, and why
  CLAUDE_CODE_BRIEFING.md  project background and the measured findings behind it

cache/                     every simulation cache. Committed — small, and expensive to redo
  runs_selector/           the selector's arm and probe measurements (~2.5 h of MATLAB)
  runs_neural/             the neural work's samples and trained networks
  runs_switch/             30 unstimulated severity-switch runs, the detection baseline
  runs_stim_*/             detection under stimulation, one directory per amplitude
  runs_ladder_*/           severity-ladder runs, stimulated and not
  runs_null_*/             no-switch control runs — these give the false-positive rate
  runs_dir_*/              upward against downward switches
  runs_t3p/                bandit run histories
  runs/                    TD3 checkpoints from the retired approach
  figures/                 generated plots and summary JSON

matlab/                    bgn_init.m, bgn_step.m, set_pd.m
gating/                    28 gating-variable functions called by bgn_step
datasets/                  400 sampled rows per severity, ~35 min of MATLAB each
archive/                   superseded approaches, kept for the record. Nothing imports it
```

**Notebooks are committed without stored outputs.** Run them to see results; the files hold
code and explanation only, so diffs stay readable and the repo stays small.

**Never hardcode a cache path.** Code names caches bare — `"runs_switch"` — and `src/paths.py`
resolves them under `cache/`. A wrong cache path does not raise: it misses, and the code
silently re-simulates. For the detection caches that is hours of MATLAB, and for the selector
about two and a half.

---

## Requirements

- MATLAB with the Python engine (`matlab.engine`)
- Python 3.12 and the packages in `requirements.txt`
- Use `/opt/anaconda3/bin/python` — `matlab.engine` is installed there and nowhere else

```bash
/opt/anaconda3/bin/python -c "import matlab.engine; print('ok')"
```

---

## Running it

**Always from the repo root.** MATLAB writes `bgn_vars.mat` to the working directory and reads
it back, and `start_engine` resolves `matlab/` and `gating/` against the cwd. Launch Jupyter
from the root; run scripts by path, which puts `src/` on `sys.path` without a cwd change.

**Simulations must run serially.** `bgn_vars.mat` is one shared file rewritten every step, so
concurrent MATLAB engines corrupt each other.

**Restart the kernel after editing any `.py` file.** `%autoreload` leaves objects bound to
stale classes and raises `AttributeError` on methods you can see in the file.

```bash
python src/neural_gates.py                                 # no MATLAB, seconds
python -u src/run_selector.py --stage probe --n-probe 15    #  ~6 min
python -u src/run_selector.py --stage arms  --n 10          # ~49 min
python -u src/run_selector.py --report --tau -0.20          # cached, no simulation
python -u src/neural_run.py  --stage collect
```

Notebooks run top to bottom. Each opens with a cell putting `src/` on the import path.

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
- `calculate_pb` sums FFT **bin indices** 13:35, not frequencies. Those correspond to 13–35 Hz
  only at a 1000 ms window; at 500 ms the same function measures 26–70 Hz.
- Use `calculate_pb` / `normalize_pb`. The `*_absolute` variants separate healthy from PD by
  only 1.24× against 2.83× and make severities look nearly identical.
- Hjorth features use **per-neuron** variances, and the bounds in `Config` were measured with
  that formula. Changing the convention invalidates those bounds.

---

## Known traps

**Grid-edge optima.** This project has produced an apparent optimum sitting on the boundary of
the search grid at least five times — 1000 µA, then 130 Hz, then 80 Hz, then 180 Hz, and the
selector's frequency result, which was retracted after extending the grid moved the answer with
it. After any selection, check whether the chosen arm sits at a grid edge in either dimension
and flag it. A quality metric that improves when you widen the search is measuring the search,
not the controller.

**The default current grid in `dbs_t3p` starts at 1000 µA**, skipping the 300–550 µA region
where the useful settings are. The notebooks override it. Check before trusting a sweep.

**Two different rewards are in the tree.** `dbs_t3p` uses the published three-term reward on
beta power, `−0.7·r1 + 0.1·r3 − 0.2·r2`. `dbs_core` uses a two-term reward on `sgis`. Numbers
from the two paths are not comparable.

**The reward surface saturates above ~350 µA**, so several arms become statistically
indistinguishable and "reached the same arm" is an unstable target. Report tied sets.

**Per-pull noise is not uniform.** The project assumed ±0.02 throughout; at the 155 Hz probe arm
it is 0.0068, three times tighter. That is what makes a short probe sufficient, and assuming the
global figure where it does not hold wastes simulations.

---

## Statistics

n ≥ 10 behind any reported number, with a standard error. Never report a difference smaller than
its error bar. Three findings here were retracted after failing at higher n, and two more in
`docs/SELECTOR_RESULTS.md` after a measurement was corrected. State null results plainly rather
than hunting for a framing that makes them look positive.

---

## Open questions

- **Where the floor τ sits** for the selector. It decides how much biomarker control is traded
  for power, and nothing measured here fixes it.
- **Frequency cannot be chosen by the current objective.** Minimum power always walks to the
  lowest frequency in the grid. It needs clinical reasoning or a different constraint.
- **Channel combination.** Hjorth features and beta power are computed per neuron and then
  averaged; a real lead records one combined field. The two are not interchangeable — averaging
  per-neuron statistics is blind to synchrony, which is the hallmark of the PD state.
- **Does the model's amplitude parameter correspond to real current?** It is labelled in µA, but
  whether the stimulation term is calibrated to clinical currents is unverified. Until it is,
  no claim about clinical amplitude ranges is safe.
- **Mode B is internally consistent, not cross-validated.** The probe calibration and the arm
  measurements come from the same simulator.
- **Why the high-amplitude measurement artifact exists.** Above 500 µA consecutive beta-power
  windows alternate high–low, which inflates the within-segment spread without moving either
  segment mean. Pair-averaging cancels it and restores detection, and the correction is
  validated, but the mechanism is undiagnosed. An aliasing explanation was tested and ruled out.
- **Whether a cheaper feature than beta power can detect a severity change.** Detection
  currently needs an FFT per neuron per window. Hjorth parameters need only variances and
  differences. Untested, and cheap to test.
