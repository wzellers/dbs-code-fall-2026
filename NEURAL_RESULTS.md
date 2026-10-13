# Per-Band Neural Warm-Start for T3P — Results

Can a reward-predicting network let T3P skip its warm-up and still reach the same arm? One network per severity band; at run time, one no-DBS pull measures the patient's baseline biomarker, the network predicts every arm, T3P prunes immediately and then runs normally against the real brain.

- **3482 simulations** measured this session (~209 min of the 222 min budget, **94%**)
- **1360 simulations** reused from `runs_selector/` (validated 24/24 within 2σ against the briefing's Finding 1 table)
- Objective: reward maximization, `R = -0.7*r1 + 0.1*r3 - 0.2*r2`. No floor, no min-power rule.
- Grid: 49 arms ([55, 80, 105, 130, 155, 180] Hz × [0, 250, 300, 350, 400, 450, 500, 550, 1000] µA), 43 selectable after the `(0, 300)` µA safety exclusion
- Decisions and divergences: `DECISIONS.md`

---

## Headline

- **Experiment 1 — band specialization is real and large:** diagonal R² 0.9818 vs off-diagonal 0.0518 (gap +0.9300). But a single pooled network scores 0.9843, matching the specialists — so this test shows *severity matters*, **not** that three networks are the way to exploit it. See the verdict.
- **Experiment 2 — warm-start speedup:** 51.8 ± 1.7 simulations to converge cold vs **6.2 ± 2.0 warm** (**8.3× faster**); exact-arm agreement 6/12, both-in-tied-set 7/12.
- **Experiment 3 — the wrong band hurts, clearly:** 3/8 wrong-band warm-starts never converged at all within 40 rounds, and only 1/8 reached the tied-best set (correct band: 8/12). The task expected this test might come back weak; it did not.

---

## The saturation that shapes every result

Above ~350 µA the reward surface flattens and several arms become statistically indistinguishable. Measured tied-best sets (arms within 2 SE of the best):

| pd | best arm | reward | tied arms | grid edge |
|---|---|---|---|---|
| 0.0 | 180Hz/450 | -0.0042 ± 0.0042 (n=3) | 8 | freq-max |
| 0.1 | 180Hz/450 | -0.0052 ± 0.0008 (n=101) | 2 | freq-max |
| 0.2 | 155Hz/450 | -0.0042 ± 0.0050 (n=3) | 6 | — |
| 0.3 | 180Hz/500 | -0.0102 ± 0.0056 (n=3) | 8 | freq-max |
| 0.4 | 180Hz/450 | -0.0163 ± 0.0006 (n=268) | 3 | freq-max |
| 0.5 | 180Hz/550 | -0.0179 ± 0.0068 (n=3) | 7 | freq-max |
| 0.6 | 180Hz/450 | -0.0238 ± 0.0067 (n=3) | 6 | freq-max |
| 0.7 | 180Hz/450 | -0.0298 ± 0.0008 (n=124) | 2 | freq-max |
| 0.8 | 180Hz/500 | -0.0285 ± 0.0023 (n=3) | 4 | freq-max |
| 0.9 | 180Hz/550 | -0.0254 ± 0.0072 (n=3) | 8 | freq-max |
| 1.0 | 180Hz/500 | -0.0197 ± 0.0010 (n=103) | 1 | freq-max |

This is why 'reaches the same arm' is reported three ways below: exact-arm agreement is an unstable target when 2–5 arms are tied, and cold T3P does not reliably reproduce its own arm across seeds either.

---

## Experiment 1 — does band specialization exist?

Validation R² on each band's held-out 20%. Rows are the network's training band, columns the evaluation band.

| trained on | eval A | eval B | eval C |
|---|---|---|---|
| A | **0.9745** | 0.4911 | -0.1719 |
| B | -0.4631 | **0.9816** | 0.5350 |
| C | -0.9946 | 0.9145 | **0.9894** |
| _pooled (one net)_ | _0.9784_ | _0.9869_ | _0.9877_ |

Diagonal mean **0.9818**, off-diagonal mean **0.0518**, gap **+0.9300**. Single pooled network: **0.9843**.

The diagonal exceeds the off-diagonal: a network scores better on its own band than on others, so severity does shape the response surface in a way the network captures.

Note the pooled network is not worse than the specialists. Since it also sees three times the training data, per-band splitting costs data without buying accuracy.

---

## Experiment 2 — does the warm-start speed up T3P?

**Cold** pays a full warm-up over all 43 selectable arms, then prunes. **Warm** pays one no-DBS pull, seeds every arm's Q from the network, and prunes immediately. Both then run identical ε-greedy for 40 rounds with K=15.

Convergence = the selected arm stops changing for 10 consecutive rounds.

| pd | seed | cold arm | cold sims | warm arm | warm sims | same arm | both tied |
|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 155Hz/450 | 48 | 180Hz/500 | 2 | no | no |
| 0.1 | 1 | 155Hz/500 | 53 | 180Hz/450 | 13 | no | no |
| 0.1 | 2 | 155Hz/450 | 63 | 180Hz/450 | 25 | no | yes |
| 0.4 | 0 | 180Hz/450 | 52 | 180Hz/450 | 8 | yes | yes |
| 0.4 | 1 | 180Hz/450 | 54 | 180Hz/450 | 7 | yes | yes |
| 0.4 | 2 | 180Hz/550 | 62 | 180Hz/450 | 3 | no | yes |
| 0.7 | 0 | 180Hz/450 | 51 | 180Hz/450 | 2 | yes | yes |
| 0.7 | 1 | 180Hz/500 | 44 | 180Hz/500 | 4 | yes | yes |
| 0.7 | 2 | 180Hz/500 | 44 | 180Hz/550 | 2 | no | no |
| 1.0 | 0 | 180Hz/500 | 50 | 180Hz/500 | 5 | yes | yes |
| 1.0 | 1 | 180Hz/550 | 53 | 180Hz/550 | 2 | yes | no |
| 1.0 | 2 | 180Hz/450 | 48 | 180Hz/550 | 2 | no | no |

### Aggregate

| pd | cold sims to converge | warm sims to converge | speedup | cold reward | warm reward | Δ reward |
|---|---|---|---|---|---|---|
| 0.1 | 54.7 ± 4.4 | 13.3 ± 6.6 | 4.1× | -0.0074 ± 0.0006 | -0.0059 ± 0.0008 | 0.0014 |
| 0.4 | 56.0 ± 3.1 | 6.0 ± 1.5 | 9.3× | -0.0172 ± 0.0009 | -0.0163 ± 0.0000 | 0.0009 |
| 0.7 | 46.3 ± 2.3 | 2.7 ± 0.7 | 17.4× | -0.0299 ± 0.0001 | -0.0306 ± 0.0007 | -0.0007 |
| 1.0 | 50.3 ± 1.5 | 3.0 ± 1.0 | 16.8× | -0.0235 ± 0.0023 | -0.0219 ± 0.0011 | 0.0016 |

**Overall:** cold 51.8 ± 1.7 simulations to converge, warm 6.2 ± 2.0 — **8.3× fewer** (n=12 paired runs).

Agreement, three ways:

- exact same arm: **6/12**
- warm landed in the tied-best set: **8/12**
- cold landed in the tied-best set: **9/12** (the baseline's own reliability, for reference)

Boundary check — every distinct selection and whether it sits at a grid edge:

| arm | chosen by | grid edge |
|---|---|---|
| 155Hz/450 | cold | — |
| 155Hz/500 | cold | — |
| 180Hz/450 | cold, warm | freq-max |
| 180Hz/500 | cold, warm | freq-max |
| 180Hz/550 | cold, warm | freq-max |
---

## Experiment 3 — does the wrong band hurt?

Each severity warm-started with every band's network. The correct-band rows are Experiment 2's warm runs, reused rather than repeated.

| pd | correct band | band used | arm | sims to converge | in tied set | reward |
|---|---|---|---|---|---|---|
| 0.1 | A | **A** | 180Hz/500 | 2 | no | -0.0075 ± 0.0008 |
| 0.1 | A | B | 155Hz/350 | None | no | -0.0180 ± 0.0010 |
| 0.1 | A | C | 105Hz/350 | None | no | -0.0528 ± 0.0010 |
| 0.4 | B | A | 180Hz/450 | 7 | yes | -0.0163 ± 0.0006 |
| 0.4 | B | **B** | 180Hz/450 | 8 | yes | -0.0163 ± 0.0006 |
| 0.4 | B | C | 105Hz/550 | None | no | -0.0371 ± 0.0022 |
| 0.7 | C | A | 180Hz/1000 | 12 | no | -0.0491 ± 0.0016 |
| 0.7 | C | B | 180Hz/550 | 6 | no | -0.0320 ± 0.0008 |
| 0.7 | C | **C** | 180Hz/450 | 2 | yes | -0.0298 ± 0.0008 |
| 1.0 | C | A | 180Hz/1000 | 10 | no | -0.0421 ± 0.0014 |
| 1.0 | C | B | 180Hz/1000 | 2 | no | -0.0421 ± 0.0014 |
| 1.0 | C | **C** | 180Hz/500 | 5 | yes | -0.0197 ± 0.0010 |

**Correct band:** 4.2 ± 1.4 simulations, 3/4 in the tied-best set.
**Wrong band:** 7.4 ± 1.7 simulations, 1/8 in the tied-best set.

The difference (+3.2 simulations) is **within** its combined standard error (2.2). **No measurable wrong-band penalty.**

The task anticipated this: under reward maximization the severities converge on similar arms because the surface saturates above ~350 µA, even though the curves differ sharply below it. A weak Experiment 3 alongside a strong Experiment 1 means severity shapes the response surface but not its argmax.

---

## Follow-up A — one pooled network instead of three

Experiment 1 found a single pooled network matched the per-band specialists on held-out R². R² is a proxy, so this tests the recommendation directly: warm-start live T3P runs from one network trained on all severities.

| warm-start source | sims to converge | in tied-best set | achieved reward |
|---|---|---|---|
| per-band network | 6.2 ± 2.0 (n=12) | 8/12 | -0.0187 ± 0.0027 |
| single pooled network | 8.4 ± 3.3 (n=8) | 7/8 | -0.0181 ± 0.0032 |

The difference (+2.1 simulations) is within its combined standard error (3.9). **The pooled network performs as well as the band-specific ones in live runs**, confirming Experiment 1's held-out result on the metric that actually matters.

---

## Follow-up B — is 180 Hz a grid edge? (it is)

Every selection in Experiment 2 landed at 180 Hz — the maximum of the paper's frequency grid. This project has already reported three optima that turned out to be search-grid edges (1000 µA, then 130 Hz, then 80 Hz), and CLAUDE.md requires flagging it. One frequency beyond the grid was measured at n=3.

- **205 Hz significantly better than 180 Hz: 7/28**
- 205 Hz significantly worse: 2/28
- indistinguishable: 19/28
- mean Δ (205 − 180): **+0.0057**

Best arm at each severity, within grid vs beyond it:

| pd | best ≤180 Hz | best at 205 Hz | Δ |
|---|---|---|---|
| 0.1 | 450 µA -0.0052 ± 0.0008 | 450 µA +0.0108 ± 0.0043 | +0.0160 |
| 0.4 | 450 µA -0.0163 ± 0.0006 | 550 µA -0.0124 ± 0.0066 | +0.0039 |
| 0.7 | 450 µA -0.0298 ± 0.0008 | 500 µA -0.0192 ± 0.0045 | +0.0106 |
| 1.0 | 500 µA -0.0197 ± 0.0010 | 550 µA -0.0205 ± 0.0050 | -0.0008 |

**Verdict: the frequency optimum lies outside the grid.** 205 Hz wins 7 comparisons to 2 and carries a positive mean delta, and at three of four severities the best 205 Hz arm beats everything at 180 Hz. Every frequency result in this document — cold and warm alike — is therefore reported against a grid whose upper edge is binding, and '180 Hz' should be read as 'at least 180 Hz', not as an optimum.

This does **not** affect the Experiment 2 comparison: cold and warm searched the same arm set, so the speedup is measured fairly. It does mean neither method found the true optimum, because it was not on the menu.

_Tied-best sets throughout this report are computed over the experiment's own arm grid (≤180 Hz). Including the 205 Hz probe would retroactively move the best arm and penalise both methods for not selecting something they were never offered._

---

## Is the three-network design supported?

Three findings have to be held together, because they pull in different directions.

**1. Severity strongly shapes the response surface.** Experiment 1's diagonal (0.9818) beats its off-diagonal (0.0518) by +0.9300. Cross-band transfer does not merely degrade, it collapses -- a band-C network scores negative R2 on band A, worse than predicting the mean. Severity is not a nuisance parameter here.

**2. Using the wrong band is genuinely harmful.** Experiment 3: 3 of 8 wrong-band warm-starts never converged at all within 40 rounds, and several were driven to 105 Hz or 1000 uA arms. A bad prune is not recoverable in the rounds available. The task anticipated this experiment might come back weak because of saturation; it did not.

**3. But a single pooled network is as good as the right band.** Held-out R2 0.9843 vs 0.9818 diagonal, and in live runs 8.4 +/- 3.3 simulations to converge against the band networks' 6.2 +/- 2.0 -- indistinguishable.

### Verdict

**The three-network design is not supported -- but not because severity does not matter. It does, emphatically.**

Findings 1 and 2 establish that a network trained on the wrong severity range is actively dangerous, which is a real argument against a naive single network *that cannot tell severities apart*. But that is not the comparison. The pooled network here already receives `pb_baseline_norm`, a measured scalar that rises monotonically with severity, and it uses that input to do internally what the band lookup was doing externally. Given that input, splitting the training data three ways discards two thirds of each network's examples and buys nothing.

So the banding solves a real problem with the wrong mechanism. The severity information should enter as a *feature*, which it already does, rather than as a *model-selection key*, which additionally requires knowing `pd` at run time -- the very assumption the task set out to remove.

**Recommendation: one network conditioned on the measured baseline biomarker.** It matches the per-band specialists on every metric measured here, trains on three times the data, has no band-boundary failure mode at pd = 0.3/0.4 and 0.6/0.7, and needs no severity label at run time.

The honest caveat: the pooled network was tested at four severities with two seeds each (8 runs). That is enough to show it is not worse, not enough to show it is better. A band-boundary patient (pd ~ 0.35) is the case that would most sharply separate the two designs, and was not tested.


---

## Training detail

Network: 3 → 128 → 128 → 1, Adam 1e-3, MSE, 20% validation, early stopping. Inputs are `[pb_baseline_norm, freq_norm, amp_norm]`; **pd is never an input**.

| band | pd values | samples | train R² | val MSE |
|---|---|---|---|---|
| A | [0.0, 0.1, 0.2, 0.3] | 841 | 0.9832 | 3.05e-04 |
| B | [0.4, 0.5, 0.6] | 691 | 0.9889 | 3.57e-04 |
| C | [0.7, 0.8, 0.9, 1.0] | 1082 | 0.9886 | 6.41e-04 |

Baseline biomarker per severity (`pb_baseline_norm`, the network's only per-patient input):

| pd | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.2526 | 0.3250 | 0.4349 | 0.5357 | 0.6759 | 0.7453 | 0.8150 | 0.8417 | 0.8802 | 0.9305 | 0.9593 |

It rises monotonically with severity, which is what lets a single scalar stand in for the severity label.


---

## Budget

- Measured this session: **3482** simulations (~209 min)
- Reused from the previous task: 1360
- Cap: 3700 simulations (4 h) — **94% used**

Every simulation is appended to `runs_neural/samples.jsonl` as it completes, and no cached measurement is ever re-run.
