# Patient-Specific DBS Controller Selection — Results

Minimum-effective-dose selector on the BGT model. Selects stimulation by minimising power
subject to a floor on **symptom control**, with a hard safety constraint, and infers severity
from a 155 Hz probe rather than assuming it is known.

- **2,353 simulations**, all on the real MATLAB model
- **n ≥ 10** behind every reported number, with standard errors
- Code: `dbs_selector.py`, `run_selector.py` · Data: `runs_selector/*.json`

> **Revision note.** Two corrections since the first version of this document, both of which
> removed headline claims rather than adding any. The floor was testing composite reward,
> which gave the no-DBS arm a structural bonus; and the frequency selection turned out to be
> an artifact of where the frequency grid stopped. §2 and §4 record what changed.

---

## Summary

| | |
|---|---|
| Validation against Finding 1 | **24/24 points within 2σ**, mean z = −0.07 |
| Amplitude increases with severity | **holds** — the core result |
| Harmful band | **confirmed**, 33σ at pd = 0.1 |
| Probe reproduces Finding 5 | **yes**, gap 0.0596 ± 0.0028 vs 0.0569 ± 0.0022 |
| Mode B (severity inferred) | **4/4** prescriptions match Mode A |
| Probes needed | **n = 8** |
| ~~"No stimulation for mild PD"~~ | **retracted** — artifact of the r3 bonus except at τ = −0.30 |
| ~~"T3P finds a cheaper 130 Hz controller"~~ | **retracted** — artifact of grid extent, confirmed 4/4 on live re-run |
| Open question | **where τ sits** |

---

## 1. Validation — the setup reproduces the published curves

Binding check before anything else: measured dose–response vs the Finding 1 table, compared
at 2σ on the **combined** error of both measurements.

**All 24 points agree.** Largest deviation 1.9σ (450 µA, pd = 0.1). Mean z = −0.07, so there
is no systematic offset.

Composite reward, 180 Hz, n = 10 per cell (mean ± SE):

| amplitude | pd = 0.1 | pd = 0.4 | pd = 0.7 | pd = 1.0 |
|---|---|---|---|---|
| 250 µA | −0.3253 ± 0.0046 | −0.4968 ± 0.0060 | −0.6084 ± 0.0089 | −0.6449 ± 0.0102 |
| 300 µA | −0.0984 ± 0.0050 | −0.1759 ± 0.0034 | −0.2193 ± 0.0053 | −0.4000 ± 0.0145 |
| 350 µA | −0.0220 ± 0.0018 | −0.0292 ± 0.0020 | −0.0418 ± 0.0039 | −0.0371 ± 0.0036 |
| 400 µA | −0.0168 ± 0.0029 | −0.0252 ± 0.0026 | −0.0407 ± 0.0025 | −0.0330 ± 0.0061 |
| 450 µA | −0.0005 ± 0.0021 | −0.0174 ± 0.0020 | −0.0304 ± 0.0016 | −0.0269 ± 0.0038 |
| 500 µA | −0.0074 ± 0.0025 | −0.0170 ± 0.0028 | −0.0287 ± 0.0015 | −0.0223 ± 0.0026 |
| 550 µA | −0.0092 ± 0.0033 | −0.0179 ± 0.0025 | −0.0306 ± 0.0028 | −0.0213 ± 0.0039 |
| **no DBS** | −0.1256 ± 0.0084 | −0.3628 ± 0.0095 | −0.4769 ± 0.0117 | −0.5799 ± 0.0146 |

**The curves fan out below 350 µA and collapse above it.** Above 350 µA the severities are
statistically indistinguishable, so a floor drawn in the flat region gives every patient the
same prescription. This is why the rule is not an argmax over reward.

**The harmful band is real at this scale.** At pd = 0.1, 250 µA scores −0.3253 ± 0.0046
against 300 µA at −0.0984 ± 0.0050 — a **33σ** gap. Sub-threshold stimulation is far worse
than the correct dose, so amplitudes in `(0, 300)` are excluded as a hard constraint rather
than penalised softly.

---

## 2. The floor now tests symptom control, not composite reward

### The confound

The floor was originally tested against the full composite

```
R = -0.7·r1(P_β) + 0.1·r3(no-stim) - 0.2·r2(RMS)
```

`r3` is ~1.0 for the no-DBS arm and ~0.001 for every stimulating arm, so no-DBS carried a
structural advantage **in the very quantity being tested for feasibility**. The `(r3, r2)`
contribution is deterministic given (freq, curr):

| arm | composite | 0.1·r3 − 0.2·r2 | biomarker −0.7·r1 |
|---|---|---|---|
| no DBS, pd = 0.1 | −0.1256 | **+0.1000** | **−0.2256 ± 0.0084** |
| 180 Hz / 350 µA, pd = 0.1 | −0.0220 | −0.0140 | −0.0080 ± 0.0018 |

A **+0.114 swing** between the two, before any biology. At pd = 0.1 the no-DBS biomarker term
is −0.2256, which does *not* clear τ = −0.20 — the composite value of −0.1256 did, purely on
the bonus.

### The fix

`SelectorConfig.floor_metric` now defaults to `"biomarker"`: feasibility tests
`mean(−0.7·r1) − z·SE ≥ τ`. The minimand is unchanged (total power). `"composite"` reproduces
the old behaviour for comparison.

No new simulation was needed. `r2` and `r3` are deterministic in (freq, curr) — `r2` because
`compute_rms` squares the waveform, so the biphasic pulse in `bgn_step.m` has the same RMS as
a monophasic one — which makes `−0.7·r1` exactly recoverable from an already-measured
composite. **Verified rather than assumed:** the 48 newly measured arms store `r1` directly,
and inverting their composites reproduces it to 3.0e-6, which is 0.30% of the tightest arm SE
in the dataset. (The residual is an edge effect — MATLAB's `dbs` array is preallocated to
`tmax*100` and grows when the final pulse runs past the end.)

### What changed

Prescribed amplitude at 180 Hz, before and after:

| τ | metric | pd 0.1 | pd 0.4 | pd 0.7 | pd 1.0 |
|---|---|---|---|---|---|
| −0.30 | composite | 0 | 300 | 300 | 350 |
| | **biomarker** | **0** | 300 | 300 | 350 |
| −0.20 | composite | 0 | 300 | 350 | 350 |
| | **biomarker** | **300** | 300 | 350 | 350 |
| −0.15 | composite | 0 | 350 | 350 | 350 |
| | **biomarker** | **300** | 350 | 350 | 350 |
| −0.10 | composite | 350 | 350 | 350 | 350 |
| | **biomarker** | **300** | 350 | 350 | 350 |
| −0.05 | composite | 350 | 350 | 350 | 350 |
| | **biomarker** | 350 | 350 | 350 | 350 |

### Retraction

**"At τ ≤ −0.15 the selector prescribes no stimulation for mild patients" was wrong.** It
survives only at **τ = −0.30**, where the biomarker term (−0.2256 ± 0.0084) genuinely clears
the floor. At τ = −0.20 and −0.15 the no-DBS arm was being carried by the no-stim bonus, not
by symptom control. Since τ = −0.20 was the recommended operating point, this removes the
most striking claim in the previous version.

---

## 3. τ sensitivity — the open question

Mode A at 180 Hz, biomarker floor, showing the biomarker term at the chosen arm:

| τ | pd 0.1 | pd 0.4 | pd 0.7 | pd 1.0 | distinct doses |
|---|---|---|---|---|---|
| −0.30 | no DBS (−0.2256) | 300 (−0.1639) | 300 (−0.2073) | 350 (−0.0231) | 3 |
| −0.20 | 300 (−0.0864) | 300 (−0.1639) | 350 (−0.0278) | 350 (−0.0231) | 2 |
| −0.15 | 300 (−0.0864) | 350 (−0.0152) | 350 (−0.0278) | 350 (−0.0231) | 2 |
| −0.10 | 300 (−0.0864) | 350 (−0.0152) | 350 (−0.0278) | 350 (−0.0231) | 2 |
| −0.05 | 350 | 350 | 350 | 350 | 1 |

τ still matters, but less dramatically than the first version claimed. Between −0.20 and −0.10
the prescription is a stable two-level split (300 µA for mild, 350 µA for the rest). τ = −0.30
is where the mild patient drops to no stimulation; τ = −0.05 is where personalisation
disappears entirely.

**Amplitude increases monotonically with severity at every floor.** This is the core result
and it survives both corrections.

---

## 4. Frequency selection is an artifact of grid extent

### The test

Power is `I·√(duty)` and duty scales with frequency, so if rewards are comparable across
frequency the min-power rule will always select the lowest frequency in the grid. With the
grid stopping at 130 Hz, it selected 130 Hz — the same shape as the earlier 1000 µA boundary
problem. 80 Hz and 105 Hz were added and measured (480 simulations, n = 10, four severities,
300–550 µA) to test whether the selection follows the grid edge.

**It does.** Unrestricted selection, biomarker floor:

| τ | pd 0.1 | pd 0.4 | pd 0.7 | pd 1.0 |
|---|---|---|---|---|
| −0.30 | no DBS | **80** Hz/300 | **80** Hz/350 | **80** Hz/400 |
| −0.20 | **105** Hz/300 | **80** Hz/350 | **80** Hz/350 | **105** Hz/350 |
| −0.15 | **80** Hz/350 | **80** Hz/350 | **105** Hz/350 | 130 Hz/350 |
| −0.10 | **80** Hz/350 | **80** Hz/350 | **105** Hz/350 | 130 Hz/350 |
| −0.05 | **105** Hz/350 | 130 Hz/350 | 130 Hz/350 | 130 Hz/350 |

### T3P confirms it on live runs

T3P was re-run on the extended grid (780 simulations, 31 safe arms × 5 warm-up + 40 rounds per
severity, τ = −0.20). It reproduces the cache-based selection exactly:

| pd | T3P (live) | cache prediction | power | old-grid T3P |
|---|---|---|---|---|
| 0.1 | 105 Hz / 300 | 105 Hz / 300 | 53.5 | 130 Hz / 350 (69.4) |
| 0.4 | 80 Hz / 350 | 80 Hz / 350 | 54.2 | 130 Hz / 350 (69.4) |
| 0.7 | 80 Hz / 350 | 80 Hz / 350 | 54.2 | 130 Hz / 350 (69.4) |
| 1.0 | 105 Hz / 350 | 105 Hz / 350 | 62.4 | 130 Hz / 350 (69.4) |

**4/4 agree.** The bandit's pruned sets are entirely low-frequency arms ranked by power — at
pd = 0.7 the survivors were `80Hz/350, 80Hz/400, 105Hz/350, 130Hz/350, 80Hz/450, …`.

### Retraction

**The 130 Hz result was not a finding.** Extending the grid downward moved the selection down
with it. Any claim that T3P "found a cheaper 130 Hz controller" — including the reported 14.7%
power saving over the 180 Hz rule — was T3P walking to the grid edge, not discovering a better
controller. The previous version's T3P comparison is withdrawn.

The clearest way to see it: **the headline metric improved by widening the grid.** Power saved
against the fixed 180 Hz / 450 µA baseline was 22–33% when the grid stopped at 130 Hz, and is
**40–49%** now that it stops at 80 Hz — same selector, same floor, same data, two extra
frequency columns. A quality metric that moves with the search bounds is measuring the bounds.

### But the floor does eventually bind

The selection is not purely grid-following. Biomarker term at 350 µA by frequency:

| freq | pd = 0.1 | pd = 0.4 | pd = 0.7 | pd = 1.0 | power |
|---|---|---|---|---|---|
| 80 Hz | −0.0826 ± 0.0018 | −0.0948 ± 0.0021 | −0.1689 ± 0.0056 | −0.3198 ± 0.0065 | 54.2 |
| 105 Hz | −0.0432 ± 0.0010 | −0.0537 ± 0.0033 | −0.0885 ± 0.0055 | −0.1617 ± 0.0070 | 62.4 |
| 130 Hz | −0.0174 ± 0.0018 | −0.0246 ± 0.0012 | −0.0285 ± 0.0017 | −0.0319 ± 0.0020 | 69.4 |
| 155 Hz | −0.0047 ± 0.0010 | −0.0198 ± 0.0017 | −0.0421 ± 0.0034 | −0.0590 ± 0.0040 | 75.7 |
| 180 Hz | −0.0080 ± 0.0018 | −0.0152 ± 0.0020 | −0.0278 ± 0.0039 | −0.0231 ± 0.0036 | 81.3 |

There is a real **efficacy cliff between 105 and 130 Hz**, steepest at high severity: at
pd = 1.0 the biomarker term degrades 5× from 130 Hz (−0.0319) to 105 Hz (−0.1617) and 10× to
80 Hz (−0.3198). At tight floors (τ = −0.10, −0.05) that cliff keeps the severe cases at
130 Hz. At the looser floors of interest it does not bind, and the grid edge decides.

**Implication:** frequency must either be fixed by clinical reasoning (180 Hz per Finding 5)
or governed by a constraint other than power. The current objective cannot select it
meaningfully, and any frequency result should be reported with the grid bounds stated.

All Mode A / Mode B results in this document hold frequency fixed at 180 Hz for this reason.

---

## 5. Mode B — severity inferred, not assumed

Retires the project's standing assumption that pd is known. Probe at 155 Hz / 450 µA, map the
reward back through the measured calibration, prescribe from the inferred severity.

### Probe calibration (n = 15 per severity)

| pd | probe reward | step vs previous |
|---|---|---|
| 0.10 | −0.0047 ± 0.0020 | — |
| 0.25 | −0.0104 ± 0.0019 | 2.1σ |
| 0.40 | −0.0245 ± 0.0018 | 5.3σ |
| 0.55 | −0.0354 ± 0.0014 | 4.8σ |
| 0.70 | −0.0490 ± 0.0015 | 6.7σ |
| 0.85 | −0.0597 ± 0.0016 | 4.8σ |
| 1.00 | −0.0643 ± 0.0020 | 1.8σ |

Monotone, with every interior step clearing 4σ.

- **pd 0.1 → 1.0 gap: 0.0596 ± 0.0028** vs the briefing's measured 0.0569 ± 0.0022 (z = +0.76).
  Finding 5 reproduces on independent data.
- **Per-pull σ at this arm is 0.0068**, not the ±0.02 assumed across the project — **3× tighter**.
  This is what makes a short probe sufficient.

Unaffected by both corrections: the probe arm is a single fixed arm, so the r3 term is
constant across the calibration and cancels out of the mapping entirely.

### Live results (8 probes per patient, τ = −0.20, 180 Hz)

| true pd | probe reward | pd̂ (raw mle) | prescribed | Mode A | result |
|---|---|---|---|---|---|
| 0.1 | −0.0082 ± 0.0030 | 0.258 (0.193) | 300 µA | 300 | match |
| 0.4 | −0.0198 ± 0.0026 | 0.378 (0.350) | 300 µA | 300 | match |
| 0.7 | −0.0480 ± 0.0014 | 0.706 (0.690) | 350 µA | 350 | match |
| 1.0 | −0.0604 ± 0.0030 | 0.971 (0.873) | 350 µA | 350 | match |

**4/4 match**, with the severity label never read.

**Read this carefully: 4/4 is not an improvement over the previous 3/4.** The probe inference
is byte-identical — pd̂ = 0.258 for a true pd of 0.1 is still a poor estimate. What changed is
that Mode A's answer for pd = 0.1 moved from 0 to 300 µA, which is where Mode B already was.
The reference moved onto Mode B, not the reverse.

The corollary is that the pd = 0.1 case no longer tests anything: under the corrected floor,
severities 0.1 and 0.4 share a prescription, so a wrong pd̂ between them is free. The genuine
test is the 300/350 boundary between pd 0.4 and 0.7, which Mode B gets right.

### The severity margin

Mode B shades pd̂ downward by one SE, prescribing for the most severe patient the probe
supports, because under-dosing is the asymmetric harm (Finding 4). Under the composite floor
this margin caused the single pd = 0.1 error; under the biomarker floor it is no longer
distinguishable at these four severities. Its Monte Carlo value (under-dosing 7.1% → 2.5%)
was computed under the old floor and **should be re-derived** before being relied on. Set
`severity_margin = 0` to disable.

### How many probes

| probes | pd MAE | dose agreement | under-dosed |
|---|---|---|---|
| 1 | 0.0986 | 73.4% | 2.3% |
| 3 | 0.0619 | 81.3% | 2.0% |
| **8** | **0.0414** | **84.8%** | 2.4% |
| 12 | 0.0347 | 84.9% | 2.5% |
| 20 | 0.0271 | 86.2% | 2.0% |

Dose agreement plateaus by **n = 8** (~31 s of stimulation). The pd MAE column is unaffected
by the floor correction; the dose columns were computed under the composite floor and are
indicative only.

---

## 6. Baselines

Fixed 180 Hz / 450 µA for every patient, n = 15, power 104.6:

| pd | baseline reward | selector (τ = −0.20, 180 Hz) | selector power | power saved |
|---|---|---|---|---|
| 0.1 | −0.0037 ± 0.0014 | 300 µA | 69.7 | 33.3% |
| 0.4 | −0.0173 ± 0.0024 | 300 µA | 69.7 | 33.3% |
| 0.7 | −0.0301 ± 0.0018 | 350 µA | 81.3 | 22.2% |
| 1.0 | −0.0239 ± 0.0038 | 350 µA | 81.3 | 22.2% |

The power saving is real, but the trade has to be stated: the baseline achieves **better
reward** at every severity because it sits in the saturated region. The selector gives up
biomarker control down to the floor in exchange for power. How much control is acceptable to
give up is exactly the τ question.

---

## 7. What is left standing

After both corrections:

- **Amplitude increases with severity.** Monotone at every floor, on validated measurements.
- **The harmful band is real and large** (33σ), and is enforced as a hard constraint.
- **The 155 Hz probe works.** Finding 5 reproduced at z = +0.76; σ is 3× tighter than assumed;
  8 probes suffice.
- **Severity can be inferred rather than assumed** — the project's oldest standing assumption
  is retired, with the caveat in §5 about what the 4/4 does and does not demonstrate.
- **The selector saves power** against one-size-fits-all, at a cost in biomarker control.
  The magnitude is **not** a stable quantity — it moves with the frequency grid bounds (§4),
  so quote it only with the grid stated, and prefer the 180 Hz-fixed figures (22–33%).

Withdrawn: the no-stimulation-for-mild-PD result (except at τ = −0.30), and every frequency
claim including the T3P power comparison.

---

## 8. Limits and open items

- **Mode B is internally consistent, not cross-validated.** The probe calibration and the arm
  measurements come from the same simulator. Held-out patients — or calibrating at one seed
  and testing at another — is the honest next test.
- **The dose-agreement and margin Monte Carlos predate the floor fix** and need re-deriving.
- **Four severities is a coarse dose mapping**, and under the corrected floor two of them share
  a prescription, which weakens Mode B's test.
- **80 Hz is now the grid floor.** Nothing here shows the selection would not follow it lower.
  The frequency question needs a non-power constraint, not a wider grid.
- **The floor is one scalar for all patients.** A severity-dependent floor is plausible and
  untested.
- **Timing fields in `t3p_selector.json` are wall-clock** and span a machine sleep. Not compute
  time.

---

## 9. Reproducing

```bash
# logic gates against the published tables — 15 checks, no MATLAB, seconds
/opt/anaconda3/bin/python dbs_selector.py --self-test

# MATLAB stages, each cached independently in runs_selector/
/opt/anaconda3/bin/python -u run_selector.py --stage probe    --n-probe 15   #  ~6 min
/opt/anaconda3/bin/python -u run_selector.py --stage arms     --n 10         # ~49 min
/opt/anaconda3/bin/python -u run_selector.py --stage extend   --n 10         # ~27 min
/opt/anaconda3/bin/python -u run_selector.py --stage baseline                #  ~4 min
/opt/anaconda3/bin/python -u run_selector.py --stage modeb    --tau -0.20 --n-probe 8
/opt/anaconda3/bin/python -u run_selector.py --stage t3p      --tau -0.20

# report from caches, no simulation — vary --tau freely here
/opt/anaconda3/bin/python -u run_selector.py --report --tau -0.20
```

`--stage extend` is additive: it measures only missing arms and merges them into the existing
caches, so the arm sweep and probe calibration are never re-run.

Simulations must run **serially** — `bgn_vars.mat` is a single file in the working directory,
rewritten every step, so concurrent engines would corrupt each other.

### Configuration

| setting | value | where |
|---|---|---|
| arm grid | 80/105/130/155/180 Hz × 0,250,…,550 µA | `dbs_selector.FREQ_VALUES`, `CURR_VALUES` |
| safety band | forbid `0 < I < 300` µA | `dbs_selector.AMP_MIN_SAFE` |
| floor metric | `biomarker` (−0.7·r1) | `SelectorConfig.floor_metric` |
| floor | τ, default −0.10 | `SelectorConfig.tau` / `--tau` |
| feasibility test | `mean(−0.7·r1) − 1.0·SE ≥ τ` | `SelectorConfig.z_feasible` |
| severity margin | 1.0 SE toward severe | `SelectorConfig.severity_margin` |
| reward | `−0.7·r1(P_β) + 0.1·r3(no-stim) − 0.2·r2(RMS)` | `dbs_t3p.T3PConfig` |
| biomarker | `calculate_pb` / `normalize_pb` | `T3PConfig.pb_fn` |

250 µA stays in the grid so validation can reproduce Findings 1 and 4, but is removed by the
safety filter before any selection. 80/105 Hz were measured at 300–550 µA only.

Pre-extend caches are preserved in `runs_selector.bak/`.

---

## Note on the briefing

`CLAUDE.md` points at `CLAUDE_CODE_BRIEFING.md`; the file is actually `claude-briefing.md`.
