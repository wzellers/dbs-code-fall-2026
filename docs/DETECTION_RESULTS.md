# Detecting a Severity Change — Results

Can we notice the moment a patient's severity changes, from beta-band power, **while a
controller is stimulating**? Yes — at every amplitude tested, in both directions, with a
measured false-positive rate.

- Roughly **280 simulations** on the real MATLAB model, n ≥ 10 behind every number
- Code: `src/pd_switch.py`, `src/experiments.py`, `src/recheck_stim.py` ·
  Notebooks: `02`–`04` · Data: `cache/runs_switch/`, `cache/runs_stim_*/`,
  `cache/runs_ladder_*/`, `cache/runs_null_*/`, `cache/runs_dir_*/`

> **Revision note.** Four claims were withdrawn while this work was being done, three of them
> mine and all three removed rather than weakened. They are recorded in §7 because the reasons
> are more useful than the claims were.

---

## Summary

| | |
|---|---|
| Detection survives stimulation | **yes** — 0 to 3500 µA |
| At the agents' operating point (180 Hz / 500 µA) | **10/10**, 1.8 s |
| At the worst amplitude (3500 µA) | **10/10**, 2.6 s |
| False-positive rate, changepoint | **1 in 20** null runs |
| Downward switches | **10/10** once the detector is two-sided |
| Changepoint against k-means under stimulation | changepoint wins decisively |
| k-means false-positive rate under stimulation | **7 in 10** — it carries no information |
| ~~"Detection fails above 500 µA"~~ | **withdrawn** — a windowing artifact, not signal loss |
| ~~"The ladder measures resolution"~~ | **withdrawn** — it measured the curve's shape |
| Open question | whether a cheaper feature than beta power works |

---

## 1. The setup

The severity parameter `pd` is changed **mid-simulation** by `set_pd.m`, which writes the new
value into `bgn_vars.mat` and leaves every state variable alone — so the second segment
continues the same brain rather than starting a new one. Raw GPi voltage either side of the
switch instant is continuous; there is no discontinuity to detect, only a change in character.

Beta power is computed on a 1000 ms window stepped 250 ms, after discarding a 2 s warm-up.
The warm-up matters: including it inflates the apparent change by about six percentage points,
because the model is still settling.

Two detectors, both **causal** — at each step they see only the data up to that point, as a
live controller would, and neither is told the switch time.

- **Changepoint.** Split the history at every allowed position, take the largest Welch
  *t*-statistic between the two halves, fire when it crosses a threshold.
- **k-means.** Cluster the history into two groups by value, fire when the high group is large
  enough, separated enough, and contiguous enough.

Thresholds were tuned on brains 1–10 and **frozen** before any test brain was touched:
changepoint 7.0 raw, 5.75 pair-averaged, k-means `min_gap` 0.13. The k-means value is a
fallback — no setting passed calibration cleanly, which is the first sign of what §4 confirms.

**Separation** throughout is the change in beta power divided by the variation within each
segment. Above 3 is reliably detectable; below 2 is not.

---

## 2. The unstimulated baseline

`pd` 0.3 → 0.7, no stimulation, 30 brains (`cache/runs_switch/`):

| | |
|---|---|
| beta power change | **+23.4% ± 1.6** |
| separation | **5.10 ± 0.36** |
| weakest brain | 1.67 |

Held out on brains 11–30, never seen during calibration:

| detector | caught | latency |
|---|---|---|
| changepoint, raw | 19/20 | 1.25 s ± 0.09 |
| changepoint, pair-averaged | 20/20 | 1.68 s ± 0.11 |
| k-means, pair-averaged | 17/20 | 1.57 s ± 0.16 |

**Reading the latency.** A detector cannot fire before `switch + 0.5 s`, because the first
window containing only post-switch data is centred half a window after the switch. So 1.25 s is
0.75 s of real delay on top of that floor.

Windows overlap — 1000 ms stepped 250 ms — so consecutive values share data and are correlated
at ρ = 0.75, 0.50, 0.25 for lags 1, 2, 3. That inflates any *t*-statistic by roughly a factor
of two, which is why a threshold of 7 is the right order where independent samples would call
for 3.6. The threshold was tuned empirically; this explains where it landed rather than
correcting it.

---

## 3. Detection under stimulation

The concern was structural: a controller's job is to suppress beta power, and beta power is also
what we watch. A thermometer held at a fixed reading.

`pd` 0.3 → 0.7, 10 brains per row (`cache/runs_stim_*/`):

| stimulation | sep raw | sep paired | gain | changepoint | k-means | latency |
|---|---|---|---|---|---|---|
| none | 5.04 | 5.93 | 1.18× | 10/10 | 8/10 | 1.98 s |
| 130 Hz / 250 µA | 3.93 | 4.51 | 1.15× | 9/10 | 4/10 | 2.40 s |
| **180 Hz / 500 µA** | 5.50 | 7.25 | 1.32× | 10/10 | 3/10 | 1.82 s |
| 130 Hz / 1000 µA | 1.46 | 3.17 | 2.17× | 8/10 | 4/10 | 3.19 s |
| 130 Hz / 3500 µA | 1.43 | 2.87 | 2.00× | 9/10 | 4/10 | 3.29 s |
| 130 Hz / 3500 µA, 10 s record | 2.01 | 3.16 | 1.57× | 10/10 | 0/10 | 2.62 s |

**This is not a clean amplitude sweep, and must not be read as one.** The 500 µA row ran at
180 Hz; every other stimulated row ran at 130 Hz. So that row differs from its neighbours in two
variables, and the apparent recovery of separation at 500 µA cannot be attributed to amplitude.
A frequency-matched sweep is one of the cheapest things left to run.

What the table does support, row by row, is the claim the section is about: **detection works at
every setting tested.** Each row is an independent measurement of whether a severity change
stays visible under that stimulus, and none of them needs a neighbour to be interpreted.

**The 180 Hz / 500 µA row is the one that matters** — it is where all three trained agents
converged, and detection there is perfect at 1.8 s. It is also the operating point used for the
null runs in §4 and the directionality test in §5, so those three results are mutually
comparable.

### The windowing artifact, and why the raw column misleads

Detection first appeared to fail completely above 500 µA: **1/10 and 0/10**. That was a flaw in
how beta power is measured, not a property of the brain.

At high amplitude, consecutive windows began alternating high–low. Separation divides the
change by the within-segment spread, so a regular alternation inflates the denominator while
leaving both segment means untouched — the signal vanished under a yardstick that had grown.

Averaging each window with its neighbour cancels an alternation exactly, while random noise only
falls by √2. Detection at the two failing amplitudes went to **8/10 and 9/10**, which is
p = 0.016 and p = 0.004 on the discordant pairs.

The `gain` column is the diagnostic, and it splits cleanly: **1.15–1.32× at the three settings
that always worked, 2.00–2.17× at the two that failed.** It is a within-row measure — how much
one run's own separation improves when its own alternation is cancelled — so unlike the
separation columns it is not touched by the frequency confound above. The cost is half the time
resolution, worth 0.3–1.0 s of extra latency.

**The cause of the alternation is not known.** The correction is validated and its effect on the
statistic is understood; why it appears above 500 µA is not.

---

## 4. Changepoint against k-means

Unstimulated the two are close — 20/20 against 17/20 held out, p = 0.22 by exact McNemar, which
is suggestive and not significant. Under stimulation they separate completely.

Null runs hold `pd` at 0.5 for the whole simulation, so **every firing is a false positive**
(`cache/runs_null_*/`, 10 brains each):

| condition | changepoint | k-means |
|---|---|---|
| no stimulation | 1/10 | 0/10 |
| 180 Hz / 500 µA | **0/10** | **7/10** |

**k-means fires on 7 of 10 runs where nothing happens, and on 3 to 4 of 10 where something
does.** Its firing rate on real switches is at or below its rate on null data, which means it
carries no information under stimulation. The 500 µA row is the clearest case: separation 7.25,
the signal obvious, clustering catches 3.

This is visible in the algorithm, not inferred from the data. The acceptance rule tests whether
two cluster centres differ by more than a fixed fraction — `(c.max() − c.min()) / c.min() >
min_gap` — and contains no term for how spread out the points within each cluster are. Measured
on pure noise, k-means with k = 2 produces centres 13% apart once the data wobbles by about 8%
of its own level, which is roughly where beta power sits under stimulation. The changepoint
statistic divides by the within-segment spread, so noise raises its own bar.

**The general point, which is worth more than our numbers:** clustering asks *are there two
groups here*, and in any real recording there always are, because the signal wanders. Detection
asks *did something change*. That is a different question and it needs a noise reference.
Calibration said as much before any test run — no `min_gap` passed cleanly on the ten
calibration brains.

**The false-positive rate is what makes everything else here mean something.** The scoring rule
counts a detection as correct if it lands after the switch, and with the switch halfway through
the record, a detector firing at an arbitrary time scores correct about half the time. Without
null runs, any count below roughly 5/10 was uninterpretable.

---

## 5. Both directions

Every result above is an increase in severity. Reading the detector's code turned up that it is
**one-sided by construction**: it scores a split by how much higher the later half is, so a
decrease gives a negative score that can never cross the threshold. k-means has the same problem
for the same reason — it anchors on where the high cluster starts, which on a downward switch is
the beginning.

A downward test with the original code would have returned zero catches, and that would have
been a fact about our code rather than the brain. `detect_cp2` is `detect_cp` with `abs()` around
the difference and nothing else changed.

Both directions at 180 Hz / 500 µA, 10 brains each (`cache/runs_dir_*/`):

| direction | change | one-sided | two-sided | latency |
|---|---|---|---|---|
| up, 0.3 → 0.7 | +79% | 10/10 | 10/10 | 1.82 s |
| down, 0.7 → 0.3 | −43% | **0/10** | 10/10 | 2.12 s |

**The signal is symmetric; only the detector was not.** Separation is 7.25 up against 7.59 down.
The percentages differ only because each is measured from its own starting point — a rise of 79%
followed by a return to the start is a fall of about 44%.

Two-sidedness costs a little: false alarms on null runs go from 1 in 20 to 2 in 20, since the
test can now be tripped either way. Worth it.

---

## 6. How small a change can be seen

A ladder from `pd` 0.5, stepping up by 0.1 to 0.4, 10 brains each (`cache/runs_ladder_*/`).
Separation, raw and pair-averaged, as in §3:

| step | unstim raw | unstim paired | 180/500 raw | 180/500 paired |
|---|---|---|---|---|
| 0.1 | 0.64 ± 0.16 | 0.81 ± 0.19 | 1.56 ± 0.27 | 2.24 ± 0.38 |
| 0.2 | 0.81 ± 0.26 | 1.03 ± 0.30 | 3.68 ± 0.36 | **5.27 ± 0.53** |
| 0.3 | 1.01 ± 0.27 | 1.31 ± 0.32 | 3.88 ± 0.59 | **5.79 ± 0.83** |
| 0.4 | 1.44 ± 0.41 | 1.95 ± 0.57 | 1.76 ± 0.92 | 2.47 ± 1.24 |

The gap percentages quoted below are raw, so that the 0.5 → 0.9 row is measured the same way as
the 30-brain 0.3 → 0.7 baseline it is being compared against.

**This did not measure what it was designed to measure.** The design held the starting point
fixed so that step size was the only variable, which assumed beta power rises roughly
monotonically with severity. It does not, so the ladder measured the shape of the curve instead.

The clearest evidence is two steps of identical width, both unstimulated:

| switch | width | change |
|---|---|---|
| 0.3 → 0.7 | 0.4 | **+23.4% ± 1.6** (n = 30) |
| 0.5 → 0.9 | 0.4 | **+5.5% ± 1.3** (n = 10) |

Four times the signal for the same step width — 8.7 standard errors apart. **Step size is not
what governs detectability; where the step starts and ends is.**

Two things follow.

**Unstimulated, there is almost no severity signal above pd 0.5.** No step on the ladder clears
a separation of 2 — the widest reaches 1.95 ± 0.57, an error bar that straddles it. Nearly all
of the 0.3 → 0.7 result comes from the stretch between 0.3 and 0.5.

**Under stimulation there is a strong signal in exactly that range.** This is the opposite of the
concern that an effective controller erases the biomarker. A fixed stimulus suppresses a mild
brain differently from a severe one, so the *response* encodes severity where the resting level
does not. The controller does not destroy observability — it moves where observability lives.

That reconciles with the selector work, which probes at 155 Hz / 450 µA and finds beta power
monotone across the whole severity range with roughly five times the spread seen at 180 Hz /
500 µA. **Whether beta power carries severity information depends on where you are stimulating,
and the agents' operating point is one of the bad ones.** It also fits the efficacy cliff the
selector measured between 105 and 130 Hz.

The 0.4 step falling back to 2.47 ± 1.24 fits the same picture: pd 0.9 is past the peak, and an
error bar half the size of the estimate says brains disagree about exactly where that peak sits.

---

## 7. Withdrawn

**"Detection fails above 500 µA."** A windowing artifact, not signal loss. §3.

**"The ladder measures the method's resolution."** It measured the shape of the beta-power curve.
The design assumed monotonicity, which a severity sweep had already shown to be false — that
sweep should have been checked before the ladder was built on it. §6.

**An aliasing explanation for the alternating artifact.** Proposed from arithmetic — 250 ms steps
against 130 Hz stimulation — and tested. Every frequency gave identical results. The hypothesis
was wrong and the mechanism remains unknown.

**A baseline-length explanation for a latency difference.** Detection at 3500 µA measured 3.29 s
on a 5 s record and 2.62 s on a 10 s record, and the longer baseline was offered as the cause.
Truncating the baseline from 10 s to 3 s changed latency not at all. Worse, the difference was
never significant — 3.29 ± 0.38 against 2.62 ± 0.11 is 1.7σ, and the two runs used different
`tmax`, so they were not even the same brains. A mechanism was proposed for noise.

**"3500 µA is about 7× therapeutic."** It is 7× *the agents' chosen setting*. Whether the model's
amplitude parameter maps onto clinical current is unverified, so no claim about therapeutic
range is supported.

One useful by-product of the failed baseline test: **latency is insensitive to baseline length
from 3 to 10 s**, so the detector needs no warm-up to reach full performance.

---

## 8. Limits and open items

- **n = 10 per condition**, 30 for the baseline. Differences smaller than about two combined
  standard errors are not differences.
- **One brain model, one biomarker.** Everything rests on `calculate_pb` over GPi voltage.
- **The amplitude sweep in §3 is not frequency-matched.** One of its six rows ran at 180 Hz and
  the rest at 130 Hz, so the rows cannot be compared to each other as amplitude steps. The
  per-row conclusions stand; a sweep of amplitude at fixed frequency has not been run.
- **Beta power is computed per neuron and then averaged**, which is not what a single lead
  measures and is blind to synchrony — the hallmark of the Parkinsonian state. Whether the
  combined-signal convention changes any of these numbers is untested and cheap to test.
- **The ladder has one starting point.** Given the curve is not monotone, a resolution measured
  from 0.5 says nothing about one measured from 0.3.
- **The alternation mechanism is undiagnosed.**
- **Detection is half the loop.** Knowing that severity changed is not knowing what it changed
  to. The selector work addresses the other half, and §6 explains why these two lines appeared to
  contradict each other and do not.
- **Detection assumes beta power can be computed online**, which costs an FFT per neuron per
  window. If that is too expensive for the device, the open question is whether a Hjorth
  feature — variances and differences, no FFT — works in its place. Untested.

---

## 9. Reproducing

Every number above is re-derived from the committed caches by one script, with no MATLAB and in
about thirty seconds:

```bash
python src/verify_detection.py     # 77 checks against the figures quoted in this document
```

It compares each quoted value against a fresh computation and prints `FAIL` on any disagreement,
so this document is checkable by anyone who clones the repo rather than being a record of
numbers someone once saw. **Run it after editing `pd_switch.py` or moving a cache.** It currently
passes 77/77.

Two things it reads from each run's stored settings rather than assuming, because both have
already caused a wrong number: `runs_stim_a500` is at 180 Hz where the rest of the sweep is at
130 Hz, and `runs_stim_long_a3500` switches at 10 s where everything else switches at 5 s.
Scoring that run with the default `switch_s=5.0` finds a changepoint in the middle of segment 1
and reports a plausible-looking wrong latency.

```bash
# notebooks run top to bottom, from the repo root
# 02_pd_switch.ipynb           the switch and the unstimulated baseline
# 03_experiments.ipynb         detection under stimulation, ladders, diagnostics
# 04_detection_results.ipynb   the conclusive run - both detectors, frozen thresholds
```

`04` is self-contained: it calibrates both detectors on the unstimulated brains, freezes them,
then scores every cached run. Sections 1–3 read caches and take about a minute; the ladder
sections simulate and take roughly 37 minutes each.

Every cache is under `cache/`, committed, and found by name through `src/paths.py`. Check for a
cache before running anything — a cache miss does not raise, it silently re-simulates.
