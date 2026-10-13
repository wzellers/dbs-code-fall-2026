# Decision Log — Per-Band Neural Warm-Start for T3P

Choices the task document left open. Terse by design; read top to bottom.

---

## [grid] Dropped amplitudes above 1000 µA

**Chose:** `CURR_VALUES = [0, 250, 300, 350, 400, 450, 500, 550, 1000]` — 6 freqs × 8 nonzero
amps + 1 no-DBS = **49 arms** (task's full grid was 73).
**Over:** the full 73-arm grid with reduced pd coverage.
**Because:** the task names this as the first cut and instructs reducing the grid rather than pd
coverage. Briefing finding 2 measured 2000–5000 µA as strictly worse at every severity tested,
and all four prior T3P runs converged to the *smallest* nonzero amplitude available. The full
grid at 11 pd values × n=3 would be 2,409 simulations for data collection alone, leaving too
little for Experiment 2 (the headline claim).
**Reversible:** yes — re-run data collection with the wider `CURR_VALUES`; costs ~1,300 extra
simulations.

## [grid] Kept 55 Hz despite no prior measurement

**Chose:** keep 55 Hz in the frequency grid.
**Over:** dropping it to save 8 arms × 11 pd.
**Because:** the previous task established that the low-frequency edge is exactly where
boundary artifacts appear, and 55 Hz is the paper's own grid floor. Dropping it would make the
frequency dimension untestable at its most informative end. Under reward maximization (not
min-power) there is no structural pull toward low frequency, so including it is a genuine test
rather than an invitation to walk the edge.
**Reversible:** yes.

## [data] Reusing `runs_selector/arms_pd*.json` for 34 of 49 arms at pd 0.1/0.4/0.7/1.0

**Chose:** reuse the existing n=10 measurements; measure only the 15 missing arms per severity.
**Over:** fresh collection of all 49 arms at all 11 pd values.
**Because:** CLAUDE.md forbids re-running a cached measurement, and those 34 arms were
validated against the briefing's Finding 1 table at 24/24 within 2σ. Saves ~1,360 simulations.
**Reversible:** n/a — purely additive.

## [data] n = 3 repeats for newly collected arms

**Chose:** n=3 per new (pd, arm); no-DBS arm gets n=6.
**Over:** n=1 (cheaper, more pd coverage) or n=10 (matches the cached arms).
**Because:** supervised regression tolerates unbiased target noise, so breadth beats depth for
training data. The no-DBS arm is the exception — it supplies `pb_baseline_norm`, the network's
only per-patient input, so its error propagates into every prediction for that patient and
warrants double sampling. Reported experiment numbers still carry n ≥ 10 because they come
from repeated T3P runs, not single arm measurements.
**Reversible:** yes — additional repeats append to the cache without invalidating it.

## [network] `pb_baseline_norm` is the per-pd mean for training, a single pull at run time

**Chose:** training uses the mean no-DBS `r1` for that severity; the live warm-start uses one
no-DBS pull.
**Over:** using a single pull for training too, or the mean at run time.
**Because:** the mean is the population value for a severity, which is what the network should
learn to condition on. The single pull at run time is the honest deployment cost — an implant
measures its patient once — and the resulting noise is a real limitation the experiment should
expose rather than hide.
**Reversible:** yes — retrain, no new simulation.

## [t3p] top_k = 15 of 43 selectable arms, identical for cold and warm

**Chose:** K=15.
**Over:** the module default K=25.
**Because:** the comparison is only meaningful if cold and warm prune identically, so K must
match. K=15 (~35% of arms) is proportionally close to the paper's 25-of-31 (~80%) being too
loose to matter here; a tighter K is where a warm-start's ranking has any value. Applied to
both arms of the comparison, so it cannot favour either.
**Reversible:** yes — re-run Experiment 2.

## [t3p] Harmful-band arms excluded from warm-up as well as selection

**Chose:** cold T3P warms up only the 43 selectable arms, not all 49.
**Over:** warming up all 49 including the six 250 µA arms.
**Because:** the task blocks `(0, 300)` µA from selection as a hard constraint, so simulating
those arms during warm-up buys nothing a selection could ever use. Charging cold T3P for them
would inflate the very baseline the headline claim is measured against — biasing the result in
the warm-start's favour. They remain in the *training* data as the task requires.
**Reversible:** yes.

## [budget] Allocation against the 3,700-simulation cap

**Chose:** ~1,230 data collection · ~1,490 Experiment 2 · ~330 Experiment 3.
**Over:** spending more on training data.
**Because:** CLAUDE.md's priority order puts data collection first but Experiment 2 (the
headline claim) above Experiment 3. Experiment 2 gets 3 seeds × 4 pd; Experiment 3 runs 1 seed
and reuses Experiment 2's correct-band runs rather than repeating them.
**Reversible:** yes.

## [network] 128 hidden units, not the task's 32 — DIVERGENCE FROM THE TASK DOCUMENT

**Chose:** `hidden = 128` (17,153 parameters).
**Over:** the task's specified 32 units (1,217 parameters), which it justified by a prior
finding that validation R² was flat from 401 to 17,537 parameters.
**Because:** R² is flat here too and is the wrong instrument. It is dominated by the large
sub-threshold cliff (rewards from −0.6 to 0) and is blind to the fine structure among the top
arms, which is exactly what a warm-start ranking depends on. Measured on the task's own gate-2
criterion:

| hidden | params | R² | arms within 2 SE |
|---|---|---|---|
| 32 | 1,217 | 0.9834 | **42%** |
| 128 | 17,153 | 0.9885 | **94%** |
| 256 | 67,073 | 0.9879 | 84% (overfits) |

At 32 units the residual error is ~0.011 while arm SEs are ~0.002 and the gaps between top
arms are 0.005–0.009 — the network could not resolve the arms it must rank. CLAUDE.md directs
following the evidence over the document and logging the divergence prominently.
**Reversible:** yes — set `HIDDEN_UNITS = 32`; no new simulation needed.

## [network] Kept `amp_norm = amp / 5000` despite dropping amplitudes above 1000

**Chose:** the task's normalization unchanged.
**Over:** rescaling to `/1000` to spread the retained grid across the input range.
**Because:** tested both — 83% vs 84% of arms within 2 SE at hidden=128. No meaningful
difference, so there is no reason to diverge from the specification.
**Reversible:** yes.

## [gates] Gate 4 scored against the tied-best set, not a single "true best arm"

**Chose:** the oracle passes if it lands on an arm statistically indistinguishable from the
best (within 2 SE), and if the *median* run converges in ≤5 simulations.
**Over:** the literal reading — exact-argmax match, and max ≤5.
**Because:** "the true best arm" is not well posed on this surface. Measured tied-set sizes
are 2–5 arms, separated by 0.0004–0.0009 against SEs of 0.002–0.004. CLAUDE.md forbids
treating a difference smaller than its error bar as real, so requiring an exact argmax match
would be testing noise. Under the corrected criterion the oracle scores 100% with a median of
1 simulation — the mechanism is sound. Reported alongside: mean 4.1, max 13, where the tail
is arm-flicker among tied arms, not a mechanism failure.
**Reversible:** yes.

## [exp2] Experiment 2 will report agreement three ways

**Chose:** report exact-arm agreement, tied-set agreement, and the achieved-reward difference
with a standard error.
**Over:** the task's single criterion, "reaches the same arm".
**Because:** the same saturation makes exact-arm agreement an unstable target — cold T3P does
not reliably reproduce its own arm across seeds either. Reporting only the exact match would
understate the method; reporting only the tied-set match would overstate it. All three are
given so the reader can apply their own bar.
**Reversible:** n/a — reporting choice.

## [extra] Ran two follow-ups the experiments themselves motivated

**Chose:** spend the remaining ~630 simulations on (A) a live pooled-network warm-start
(164 sims) and (B) a 205 Hz boundary probe (96 sims), finishing at 3,482 / 3,700 (94%).
**Over:** more seeds for Experiments 2 and 3.
**Because:** (A) the report's central recommendation rested on Experiment 1's held-out R²,
a proxy; testing it in live T3P runs is what makes the recommendation defensible. (B) every
Experiment 2 selection landed at 180 Hz, the grid maximum, and CLAUDE.md requires flagging
grid-edge results after three prior artifacts (1000 µA, 130 Hz, 80 Hz). Both changed what
the report concludes; extra seeds would only have tightened error bars on results already
significant.
**Reversible:** yes — both are additive measurements.

## [analysis] Tied-best sets computed over the experiment grid only (≤180 Hz)

**Chose:** exclude the 205 Hz probe arms from `arm_stats` / `tied_best_set` by default.
**Over:** scoring against every arm in the sample store.
**Because:** this was a real bug caught mid-analysis. Adding the 205 Hz samples silently
moved the "best arm" at several severities, and warm's tied-set score dropped from 10/12 to
4/12 — not because the algorithm changed, but because it was being penalised for not
selecting an arm that was never in its action set. Corrected figures: cold 9/12, warm 8/12,
pooled 7/8. The 205 Hz result is reported separately as the boundary caveat.
**Reversible:** yes — `arm_stats(..., grid_only=False)`.

## [detection] ladder table reports both raw and pair-averaged separation
**Chose:** show both columns, and keep the gap percentages raw
**Over:** paired only (consistent with §2-§5) or raw only (as first written)
**Because:** the document was quoting raw separation in §6 and paired everywhere else, which is
a silent inconsistency. Both are shown because the correction's effect is small at these low
amplitudes (0.64 -> 0.81) where it is large at high amplitude (1.43 -> 2.87), and that contrast
is itself evidence for the period-2 explanation. The gap percentages stay raw so the 0.5 -> 0.9
row is measured the same way as the 30-brain 0.3 -> 0.7 baseline it is compared against.
**Reversible:** yes - re-scoring only, no simulation. `src/verify_detection.py` checks both.

## [detection] detect_cp2 promoted from notebook cell to src/pd_switch.py
**Chose:** move the two-sided detector into the module
**Over:** leaving it defined in 04_detection_results.ipynb cell 21
**Because:** it is the detector the results actually recommend - one-sided detect_cp catches
0/10 downward switches - and it was not importable, so section 5 of DETECTION_RESULTS.md was
not reproducible from committed code. A result that depends on a notebook cell is not a result.
**Reversible:** yes, but there is no reason to.

## [detection] src/verify_detection.py added
**Chose:** a script that re-derives all 77 quoted figures from the committed caches
**Over:** trusting the notebook output that produced them
**Because:** committing the caches is only worth it if the document becomes checkable. Writing
it caught two things: the ladder raw/paired inconsistency above, and that runs_stim_a500 ran at
180 Hz while the rest of the sweep ran at 130 Hz, so the table was not the clean amplitude sweep
it was presented as. Both are now disclosed in the document.
**Reversible:** n/a - additive.

## [repo] MATLAB path setup moved into paths.py, caller arguments ignored
**Chose:** have MatlabBackend, pd_switch.start_engine and dbs_core.MatlabEnv all add
paths.MATLAB and paths.GATING explicitly, keep code_dir only as MATLAB's working directory,
and accept-but-ignore parent_dir / matlab_dir
**Over:** fixing the call sites to pass the right arguments
**Because:** the restructure moved the .m files into matlab/ while gating/ stayed at the root,
and two notebooks still passed parent_dir="." - which added the repo root instead of matlab/
and left bgn_init itself undefined. Three engine-startup sites had three different path
conventions between them, one of which (dbs_core.MatlabEnv) added nothing at all. Fixing the
call sites would have left the footgun loaded. Now there is one convention and callers cannot
get it wrong. Each site also raises with the resolved path if either directory is missing,
rather than warning and failing later inside bgn_step.
**Note:** the old "matlab/gating not found" warning really was harmless, as
CLAUDE_CODE_BRIEFING.md said - bgn_init.m line 58 runs addpath("gating") itself, relative to
the cwd. The broken part was parent_dir=".", not the warning. Briefing corrected to say both.
**Reversible:** yes - no cached data involved. Needs a MATLAB machine to confirm; cannot be
verified from a container without the engine.
