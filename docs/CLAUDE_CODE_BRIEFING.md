# Context and Task Briefing

Read this in full before writing code. It covers the codebase, hard-won environment
gotchas, measured findings, and the task.

---

## Project

Deep Brain Stimulation for Parkinson's, simulated on a Basal Ganglia-Thalamic (BGT)
computational model in MATLAB, driven from Python.

Prior published work (ICCPS): **T3P MAB**, a Time & Threshold-Triggered Pruned
multi-armed bandit that selects stimulation frequency and amplitude, beating deep-RL
approaches at lower power. T3P = play all arms once (warm-up), prune to top K by mean
reward, then ε-greedy with linear decay, re-triggered on a timer or a biomarker deviation.

**This semester's goal:** patient-specific controller selection. Different PD severities
need different stimulation. Build a selector that lands on the right one per patient.

`pd` is the severity parameter: `pd = 0` is a healthy brain, scaling up to `pd = 1.0` for
full PD. It enters the model as a dopamine-depletion proxy through three applied currents
in `bgn_step.m`:

```matlab
Iappstn = 33 - pd*10
Iappgpe = 21 - 13*pd + r
Iappgpi = 22 - pd*6
```

---

## Codebase

```
workspace/
├── dbs_t3p.py            T3P bandit + MATLAB backend + arm sweep   <- the active module
├── helpers.py            biomarker and reward primitives (do not edit)
├── matlab/               bgn_init.m, bgn_step.m
├── gating/               28 gating functions (bgn_init.m adds this itself)
├── runs_t3p/             T3P run results as JSON
├── bgn_vars.mat          MATLAB state, rewritten every simulation step
│
├── dbs_core.py           TD3 agent + env. LEGACY — TD3 was abandoned, see below
├── dbs_checks.py         verification harness, written for the TD3 env
├── dbs_plots.py          plotting
└── dbs_reward_model.py   supervised reward model. Superseded by the T3P work
```

`dbs_t3p.py` is the live code. The other `dbs_*.py` modules are from earlier approaches
that were abandoned; don't build on them without asking.

### Running it

```python
import helpers, dbs_t3p as t3p
backend = t3p.MatlabBackend(code_dir=".", parent_dir="matlab")
cfg = t3p.T3PConfig(pd_value=1.0, tmax=1000)
rows = t3p.sweep_all_arms(backend, cfg, helpers, repeats=5)
res, bandit = t3p.run_t3p(backend, cfg, helpers)
```

One simulation ≈ **3.9 seconds**. Budget accordingly: a 31-arm sweep at 5 repeats is about
10 minutes; a full T3P run (31 warm-up + 75 rounds) is about 4 minutes.

`MatlabBackend` used to print a warning about `matlab/gating` not existing. That warning was
harmless, for the reason given here originally: `bgn_init.m` line 58 runs `addpath("gating")`
itself, relative to MATLAB's working directory, which is the repo root. It no longer appears —
`MatlabBackend` now adds `matlab/` and `gating/` explicitly through `paths.py`, so the model
does not depend on the cwd or on `bgn_init` having run first.

What was *not* harmless, and is now fixed: two notebooks constructed the backend as
`MatlabBackend(code_dir=".", parent_dir=".")`, which added the repo root to the MATLAB path
instead of `matlab/`. After the restructure moved the `.m` files into `matlab/`, that left
`bgn_init` itself undefined. Both arguments are now ignored and the paths come from `paths.py`,
so the call is just `t3p.MatlabBackend()`.

---

## Environment gotchas — every one of these cost real debugging time

**Use `calculate_pb` / `normalize_pb`, never `calculate_pb_absolute` /
`normalize_pb_absolute`.** `helpers.py` contains two P_β implementations. The correct one
separates healthy from PD by **2.83×** (369,808 vs 1,047,640, matching the paper's 348,000
/ 896,000). The "absolute" variant separates them by only **1.24×** and makes severities
look identical. `T3PConfig.pb_fn` / `pb_norm_fn` control this; the defaults are correct.

**Reward weights follow the paper**, not the older MAB scripts:
`R = -0.7·r1(P_β) + 0.1·r3(no-stim) - 0.2·r2(RMS)`. Some older code has the last two
swapped.

**`bgn_step` takes index units, not milliseconds.** `dt = 0.01` ms, so `i` advances 100
units per millisecond. The third argument is `tmax * 100`.

**Cast MATLAB arrays to float64 before arithmetic.** `dbs ** 2` in the native integer dtype
overflows and wraps negative, silently corrupting RMS.

**`bgn_init(pd, tmax, seed)` gets a fresh random seed per pull**, so every simulation is a
different brain realization. Single-sample measurements are unreliable — per-pull reward
noise is roughly ±0.02. Use n ≥ 10 with standard errors for anything that matters. Several
"findings" were retracted after they failed at higher n.

**Restart the kernel after editing a `.py` file.** Don't rely on `%autoreload` — it leaves
objects bound to stale classes and produces confusing `AttributeError`s on methods that
visibly exist in the file.

**The arm grid is configurable:**

```python
t3p.CURR_VALUES = [0, 100, 250, 500, 750, 1000, 2000, 3500, 5000]
t3p.ARMS = t3p.build_arms()
cfg = t3p.T3PConfig(n_arms=len(t3p.ARMS), top_k=39)
```

Amplitude 0 collapses across frequencies into a single "no DBS" arm, so the default grid is
31 arms, not 36.

---

## Measured findings — all confirmed, n ≥ 10, error bars throughout

### 1. Severity monotonically increases the required current

Reward at 180 Hz, n = 20 per point:

| amplitude | pd = 0.1 | pd = 0.4 | pd = 0.7 | pd = 1.0 |
|---|---|---|---|---|
| 250 µA | −0.3153 ± 0.0047 | −0.5047 ± 0.0059 | −0.6028 ± 0.0063 | −0.6231 ± 0.0068 |
| 300 µA | −0.1045 ± 0.0038 | −0.1775 ± 0.0031 | −0.2281 ± 0.0041 | −0.4240 ± 0.0099 |
| 350 µA | −0.0187 ± 0.0012 | −0.0275 ± 0.0016 | −0.0434 ± 0.0023 | −0.0431 ± 0.0025 |
| 400 µA | −0.0129 ± 0.0017 | −0.0218 ± 0.0016 | −0.0376 ± 0.0023 | −0.0376 ± 0.0026 |
| 450 µA | −0.0059 ± 0.0018 | −0.0124 ± 0.0022 | −0.0285 ± 0.0018 | −0.0292 ± 0.0020 |
| 500 µA | −0.0102 ± 0.0021 | −0.0142 ± 0.0017 | −0.0273 ± 0.0020 | −0.0205 ± 0.0024 |

Minimum current to clear a performance floor:

| floor | pd = 0.1 | pd = 0.4 | pd = 0.7 | pd = 1.0 | spread |
|---|---|---|---|---|---|
| −0.30 | 254 µA | 281 µA | 290 µA | 316 µA | +25% |
| −0.20 | 277 µA | 297 µA | 308 µA | 329 µA | +19% |
| −0.10 | 303 µA | 326 µA | 335 µA | 343 µA | +13% |
| −0.05 | 332 µA | 342 µA | 348 µA | 349 µA | +5% |

Monotonic at every floor. **Trust the −0.30 to −0.10 rows** — there every curve is crossed
in its steep region. Looser floors compress the difference; tighter ones (−0.03) exaggerate
it because mild brains cross in the steep region while severe ones cross in the flat region.

### 2. Reward maximization hides the effect; minimum-effective-dose reveals it

Ask "which arm maximizes reward" and every severity answers 450–500 µA — nearly identical.
Above ~350 µA the biomarker saturates, the reward collapses to the energy term, and the
argmax lands on the smallest sufficient current, which is similar everywhere.

**The selector must minimize power subject to a performance floor, not maximize reward.**
This is the single most important design consequence of the measurements.

### 3. Saturation constrains where the floor can sit

pd = 0.7 vs pd = 1.0 separation by amplitude: 2.2σ at 250 µA, **18.3σ at 300 µA**, then
0.1σ / 0.0σ / 0.3σ at 350 / 400 / 450 µA.

Above 350 µA the severe brains are statistically indistinguishable. **The floor must sit in
the steep region** (roughly −0.30 to −0.10) or every patient gets the same prescription.

### 4. Under-dosing harms mild brains, at 11σ

Measured on the biomarker directly, n = 15:

| amplitude | pd = 0.1 vs no-DBS | pd = 1.0 vs no-DBS |
|---|---|---|
| 150 µA | −0.2σ (inert) | −2.2σ (slight help) |
| 200 µA | +0.2σ (inert) | −3.4σ (slight help) |
| **250 µA** | **+11.1σ HARMFUL** | −2.5σ |
| **275 µA** | **+7.9σ HARMFUL** | −1.9σ |
| 300 µA | −23.8σ (helps) | −20.7σ (helps) |

Activation threshold is between 275 and 300 µA at both severities.

**This is a safety constraint, not an efficiency one.** A selector that errs low on a mild
patient does active harm; the same error on a severe patient merely wastes an opportunity.

### 5. 155 Hz is the best severity *probe*

Separation between pd = 0.1 and pd = 1.0:

| frequency | @ 350 µA | @ 450 µA |
|---|---|---|
| 130 Hz | 4.6σ | 3.9σ |
| **155 Hz** | **17.9σ** | **23.6σ** |
| 180 Hz | 5.0σ | 5.6σ |

180 Hz saturates both severities and compresses the difference; 155 Hz saturates the mild
brain but not the severe one. So **180 Hz is the better treatment frequency, 155 Hz the
better diagnostic frequency.**

Reproducible: the 155 Hz / 450 µA gap measures 0.0569 ± 0.0022 across three independent
seeds.

The severity effect itself holds at all three frequencies — it is a property of amplitude
generally, not of one frequency.

---

## What was tried and abandoned

**TD3 (deep RL).** Lost to fixed open-loop DBS in 6 of 6 runs across three severities, and
to no stimulation at all in 5 of 6. Root cause: standard TD3 seeds its replay buffer with
uniformly random actions, and this implementation did not — the untrained actor only ever
proposed 1600–3500 µA, so the critic never saw the region where the optimum lives. Code is
in `dbs_core.py`. Do not revive without discussion.

**Supervised reward model.** A network predicting reward from (state, action), with
uniformly sampled data. It located the optimum in its own data but could not resolve it —
validation R² topped out near 0.50 and the argmax landed on zero amplitude (optimizer's
curse). Useful secondary result: network capacity is irrelevant here, with 401 and 17,537
parameters performing identically. Code in `dbs_reward_model.py`.

---

## Task: build the selector

Create `dbs_selector.py`. The design follows from the findings above.

### Core rule — lexicographic, not reward-maximizing

1. Find the minimum stimulation amplitude whose measured reward clears a floor τ.
2. Among candidates that clear it, pick the lowest power.
3. Never select an amplitude in the harmful band (below ~300 µA) — this is a hard safety
   constraint, not a soft preference.

Finding 2 is why this is not an argmax over reward.

### Two modes

**Mode A — severity known.** Select the minimum effective dose directly. Straightforward
given the measured curves; useful as a reference upper bound on achievable performance.

**Mode B — severity inferred.** The interesting version, and newly possible given finding
5. Probe at 155 Hz / 450 µA, observe the reward, infer severity from the known mapping,
then set the dose. Needs a decision on how many probes are enough — the gap is 0.057 with
per-observation noise around 0.02, so a handful should suffice, but quantify it rather than
guessing.

Mode B retires an assumption the project has carried from the start ("assume we can
identify a patient's pd level"). Worth treating as the headline contribution.

### Connecting it to T3P

T3P is the group's own algorithm and should be the selection mechanism where it fits.
Two readings, worth implementing whichever is cleaner:

- T3P over stimulation arms, with pruning modified to enforce the floor and the harmful-band
  constraint rather than ranking by mean reward.
- T3P over per-severity controllers, treating each as an arm.

The first is closer to what the measurements support.

### Requirements

- Validate against the measured curves above before any long run. If the selector
  prescribes something contradicting finding 1, the selector is wrong, not the data.
- Report per-severity: chosen amplitude, achieved reward, whether the floor was cleared,
  and total power.
- Compare against two baselines: fixed 180 Hz / 450 µA for every patient (one-size-fits-all)
  and the measured optimum per severity (oracle).
- Make the floor τ a parameter. Its correct value is an open question for the advisor;
  the code should make the sensitivity visible rather than hardcoding a choice.
- Every reported number needs n ≥ 10 and a standard error. This project has already
  retracted three findings that failed at higher n.

### Start by

Writing the selector logic against the *measured tables in this document* as a mock, with
no MATLAB in the loop. That validates the rule in seconds rather than minutes. Only then
wire it to `MatlabBackend`.