# Task: Per-Band Neural Warm-Start for T3P

> **Status: completed.** Results are in `NEURAL_RESULTS.md`, decisions in
> `DECISIONS.md`. Kept as the specification this work was carried out against.

Background and measured findings are in `CLAUDE_CODE_BRIEFING.md`. Operating rules are in
`CLAUDE.md`. This document is the current task. Work autonomously; log decisions to
`DECISIONS.md`.

---

## 1. The idea in one paragraph

T3P must play all arms once before it can prune, and that warm-up dominates its convergence
time. If a neural network can predict what each arm will score for a given patient, T3P can
prune immediately and spend its rounds refining instead of surveying. Train one network per
severity band; at run time, look up the band from the patient's `pd`, predict every arm, warm-
start T3P from those predictions, then let T3P run normally against the real brain.

**The claim to test: warm-started T3P reaches the same arm as cold T3P in substantially fewer
simulations.** Faster to the same answer is the result. Faster to a different answer is a
failure.

---

## 2. Objective

**Reward maximization, using the paper's reward function.** T3P's normal behaviour.

```
R = -0.7*r1(P_beta) + 0.1*r3(no-stim) - 0.2*r2(RMS)
```

Power is already inside the reward as `r2`. Do **not** implement a minimum-power rule, a
performance floor, or a lexicographic selection rule. A previous task did, and it caused the
optimizer to walk to the cheapest edge of whatever grid it was given. That approach is
withdrawn.

The one exception: **amplitudes in `(0, 300)` uA are excluded from selection** as a hard
safety constraint. Sub-threshold stimulation measurably worsens the biomarker in mild brains
at 11 sigma, and the reward function alone will not reliably prevent choosing it. Keep those
arms in the training data so the network learns the region exists; block them at selection.

---

## 3. Bands

| band | pd range | training pd values |
|---|---|---|
| A | 0.0 - 0.3 | 0.0, 0.1, 0.2, 0.3 |
| B | 0.4 - 0.6 | 0.4, 0.5, 0.6 |
| C | 0.7 - 1.0 | 0.7, 0.8, 0.9, 1.0 |

Lookup: `pd <= 0.3 -> A`, `pd <= 0.6 -> B`, else `C`.

`pd = 0.0` is a healthy brain. Band A must learn that minimal stimulation is correct there.
Do not exclude it.

---

## 4. Arm grid

Start from the paper's grid and add the refined low-amplitude region, because the paper's
amplitude set cannot express the optimum (it jumps 0 -> 1000 uA, and the optimum is near
350-450 uA).

```python
FREQ_VALUES = [55, 80, 105, 130, 155, 180]
CURR_VALUES = [0, 250, 300, 350, 400, 450, 500, 550, 1000, 2000, 3000, 4000, 5000]
```

Amplitude 0 collapses across frequencies into one no-DBS arm -> **73 arms**.

If the budget will not cover 73 arms across 11 pd values, reduce the grid rather than the pd
coverage, and log the reduction. Amplitudes above 1000 uA are the first candidates to drop -
they were measured as strictly worse at every severity tested.

---

## 5. Network

```
input  : [pb_baseline_norm, freq_norm, amp_norm]
hidden : Linear(3,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,1)
output : predicted reward
```

- `pb_baseline_norm` = `normalize_pb(calculate_pb(vgi))` measured with stimulation **off** for
  that patient. It is the only per-patient measurement, and it is what an implant can actually
  sense.
- `freq_norm = freq / 180.0`, `amp_norm = amp / 5000.0`
- **`pd` is never an input.** The severity label does not enter the network, only its
  measurable consequence. Do not add it.

Training is ordinary supervised regression: Adam, lr 1e-3, MSE, 20% validation split, early
stopping. No reinforcement learning.

32 hidden units is enough - a previous experiment found validation R-squared flat from 401 to
17,537 parameters on this problem. Do not spend time on architecture search.

---

## 6. Neural T3P

Subclass `T3PBandit` from `dbs_t3p.py`. Two changes:

1. **`warm_start(network, pb_baseline)`** - set every arm's initial `Q` to the network's
   predicted reward with a pseudo-count of 1, then prune immediately by the existing top-K
   rule. No simulation spent on warm-up.
2. **Online updates unchanged.** Epsilon-greedy with the existing decay; real observed rewards
   overwrite the predictions by running average.

The only simulation before warm-start is the no-DBS pull that measures `pb_baseline`.

---

## 7. Experiments

### Experiment 1 - does band specialization exist?

Score each trained network against every band's held-out data. Report a 3x3 validation
R-squared matrix.

**Success:** diagonal exceeds off-diagonal. **If flat, report that one network suffices.**
That is a legitimate finding.

Cheap - uses cached data, no MATLAB. Run it before Experiments 2 and 3 and let the result
inform how much to invest in them.

### Experiment 2 - does the warm-start speed up T3P?

For each pd value, compare:

- **Cold T3P** - full warm-up over all arms, then prune, then epsilon-greedy
- **Warm T3P** - one no-DBS pull, network warm-start, immediate prune, then epsilon-greedy

Metrics: simulations used before the selected arm stops changing for 10 consecutive rounds,
and the final arm each reaches.

**Success:** warm converges in meaningfully fewer simulations *and* reaches the same arm.

3 seeds per pd. Report mean and spread.

### Experiment 3 - does the wrong band hurt?

Warm-start T3P with each of the three networks for each pd. **Success:** the correct band
converges fastest or reaches a better arm.

Note in advance: under reward maximization the severities may converge on similar arms,
because the reward surface saturates above ~350 uA even though the curves differ sharply
below it. So Experiment 3 may come back weak even when Experiment 1 is strongly positive.
That combination is an informative result - severity strongly shapes the response surface,
but the argmax is similar across severities - and should be reported as such, not buried.

---

## 8. Validation gates - before any MATLAB

Mock against `runs_selector/arms_pd*.json`, which is real measured data.

1. Band lookup correct at boundaries 0.3, 0.4, 0.6, 0.7
2. A network trained on cached pd=0.1 data predicts its own training rewards within 2 SE
3. Warm-started T3P never selects an amplitude in `(0, 300)`
4. Warm-started T3P with a perfect oracle network converges in <= 5 simulations
5. `pd` appears nowhere in any network input tensor
6. Reward recomputed from cached raw `pb`/`rms` reproduces cached rewards to within 1% of the
   tightest arm's SE

Gate 4 matters most: if a perfect network does not produce near-instant convergence, the
warm-start mechanism is broken and Experiment 2 will be uninterpretable.

---

## 9. Reporting

`NEURAL_RESULTS.md`, written incrementally as stages complete:

- The three experiment tables
- Per pd: selected arm, achieved reward +/- SE, simulations to converge, cold vs warm
- Every number with n >= 10 and a standard error
- Boundary flags: for each selection, whether it sits at a grid edge in frequency or amplitude
- An explicit statement on whether Experiments 1 and 3 support the three-network design
- Total simulations used against the 4-hour budget

---

## 10. Order of work

1. Validation gates (seconds, no MATLAB)
2. Data collection - reuse `runs_selector/arms_pd*.json` for pd = 0.1, 0.4, 0.7, 1.0
3. Train three networks (seconds)
4. Experiment 1 (no MATLAB)
5. Experiment 2
6. Experiment 3
7. Final `NEURAL_RESULTS.md` and `DECISIONS.md`

Write results to disk as each step finishes. An interrupted session should still leave a
readable report.