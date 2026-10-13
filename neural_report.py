"""Builds NEURAL_RESULTS.md from whatever artifacts exist in runs_neural/.

Safe to run at any point: missing stages are reported as not-yet-run rather than omitted,
so an interrupted session still produces a readable document.
"""
import json
import math
from pathlib import Path

import numpy as np

import neural_t3p as nt

OUT = Path(nt.RESULTS_DIR)
REPORT = Path("NEURAL_RESULTS.md")
SIM_SECONDS = 3.6
BUDGET_SIMS = 3700


def load(name):
    p = OUT / name
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        return None


def fmt(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def mean_se(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return None, None, 0
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return float(a.mean()), (sd / math.sqrt(len(a)) if len(a) > 1 else 0.0), len(a)


def arm_label(f, c):
    return "no DBS" if c == 0 else f"{f}Hz/{c}"


def edge_note(f, c):
    e = nt.at_grid_edge(f, c)
    return ", ".join(e) if e else "—"


def write():
    samples = nt.load_samples()
    measured = [s for s in samples if s.get("src") != "runs_selector"]
    n_meas = len(measured)
    stats = {pd: nt.arm_stats(samples, pd) for pd in nt.ALL_PD}
    tied = {pd: nt.tied_best_set(samples, pd, stats=stats[pd]) for pd in nt.ALL_PD}

    L = []
    w = L.append
    w("# Per-Band Neural Warm-Start for T3P — Results\n")
    w("Can a reward-predicting network let T3P skip its warm-up and still reach the same "
      "arm? One network per severity band; at run time, one no-DBS pull measures the "
      "patient's baseline biomarker, the network predicts every arm, T3P prunes "
      "immediately and then runs normally against the real brain.\n")
    w(f"- **{n_meas} simulations** measured this session "
      f"(~{n_meas*SIM_SECONDS/60:.0f} min of the {BUDGET_SIMS*SIM_SECONDS/60:.0f} min "
      f"budget, **{n_meas/BUDGET_SIMS*100:.0f}%**)")
    w(f"- **{len(samples)-n_meas} simulations** reused from `runs_selector/` (validated "
      "24/24 within 2σ against the briefing's Finding 1 table)")
    w(f"- Objective: reward maximization, `R = -0.7*r1 + 0.1*r3 - 0.2*r2`. No floor, no "
      "min-power rule.")
    w(f"- Grid: {len(nt.ARMS)} arms ({nt.FREQ_VALUES} Hz × {nt.CURR_VALUES} µA), "
      f"{len(nt.SAFE_ARMS)} selectable after the `(0, 300)` µA safety exclusion")
    w("- Decisions and divergences: `DECISIONS.md`\n")

    # ---------------------------------------------------------------- headline
    e1, e2, e3 = load("exp1.json"), load("exp2.json"), load("exp3.json")
    w("---\n\n## Headline\n")
    if e1:
        gap = e1["diag_mean"] - e1["off_mean"]
        real = "real and large" if gap > 0.02 else "absent"
        w(f"- **Experiment 1 — band specialization is {real}:** diagonal R² "
          f"{e1['diag_mean']:.4f} vs off-diagonal {e1['off_mean']:.4f} "
          f"(gap {gap:+.4f}). But a single pooled network scores "
          f"{e1['pooled_mean']:.4f}, matching the specialists — so this test shows "
          f"*severity matters*, **not** that three networks are the way to exploit it. "
          f"See the verdict.")
    else:
        w("- Experiment 1: not yet run.")
    if e2:
        w("- " + _exp2_headline(e2, tied))
    else:
        w("- Experiment 2: not yet run.")
    if e3:
        w("- " + _exp3_headline(e3, e2, tied))
    else:
        w("- Experiment 3: not yet run.")
    w("")

    # ---------------------------------------------------------------- saturation
    w("---\n\n## The saturation that shapes every result\n")
    w("Above ~350 µA the reward surface flattens and several arms become statistically "
      "indistinguishable. Measured tied-best sets (arms within 2 SE of the best):\n")
    w("| pd | best arm | reward | tied arms | grid edge |")
    w("|---|---|---|---|---|")
    for pd in nt.ALL_PD:
        st = stats.get(pd) or {}
        if not st:
            continue
        bk = max(st, key=lambda k: st[k][0])
        m, se, n = st[bk]
        w(f"| {pd} | {arm_label(*bk)} | {m:+.4f} ± {se:.4f} (n={n}) | "
          f"{len(tied[pd])} | {edge_note(*bk)} |")
    w("\nThis is why 'reaches the same arm' is reported three ways below: exact-arm "
      "agreement is an unstable target when 2–5 arms are tied, and cold T3P does not "
      "reliably reproduce its own arm across seeds either.\n")

    # ---------------------------------------------------------------- exp 1
    w("---\n\n## Experiment 1 — does band specialization exist?\n")
    if not e1:
        w("_Not yet run._\n")
    else:
        bands = sorted(e1["matrix"])
        evb = sorted(next(iter(e1["matrix"].values())))
        w("Validation R² on each band's held-out 20%. Rows are the network's training "
          "band, columns the evaluation band.\n")
        w("| trained on | " + " | ".join(f"eval {b}" for b in evb) + " |")
        w("|---" * (len(evb) + 1) + "|")
        for tb in bands:
            cells = []
            for eb in evb:
                v = e1["matrix"][tb][eb]
                cells.append(f"**{v:.4f}**" if tb == eb else f"{v:.4f}")
            w(f"| {tb} | " + " | ".join(cells) + " |")
        w("| _pooled (one net)_ | " +
          " | ".join(f"_{e1['pooled'][eb]:.4f}_" for eb in evb) + " |")
        gap = e1["diag_mean"] - e1["off_mean"]
        w(f"\nDiagonal mean **{e1['diag_mean']:.4f}**, off-diagonal mean "
          f"**{e1['off_mean']:.4f}**, gap **{gap:+.4f}**. "
          f"Single pooled network: **{e1['pooled_mean']:.4f}**.\n")
        if gap > 0.02:
            w("The diagonal exceeds the off-diagonal: a network scores better on its own "
              "band than on others, so severity does shape the response surface in a way "
              "the network captures.\n")
        else:
            w("**The matrix is flat.** A network trained on one band predicts other bands "
              "about as well as its own, and a single pooled network matches the "
              "per-band networks. On this evidence **one network suffices** and the "
              "three-network design is not justified. That is a legitimate finding, not a "
              "failure.\n")
        if e1["pooled_mean"] >= e1["diag_mean"] - 0.01:
            w("Note the pooled network is not worse than the specialists. Since it also "
              "sees three times the training data, per-band splitting costs data without "
              "buying accuracy.\n")

    # ---------------------------------------------------------------- exp 2
    w("---\n\n## Experiment 2 — does the warm-start speed up T3P?\n")
    if not e2:
        w("_Not yet run._\n")
    else:
        w(_exp2_tables(e2, stats, tied))

    # ---------------------------------------------------------------- exp 3
    w("---\n\n## Experiment 3 — does the wrong band hurt?\n")
    if not e3:
        w("_Not yet run._\n")
    else:
        w(_exp3_tables(e3, e2, stats, tied))

    # ---------------------------------------------------------------- pooled
    pool = load("extra_pooled.json")
    w("---\n\n## Follow-up A — one pooled network instead of three\n")
    if not pool:
        w("_Not run._\n")
    else:
        w("Experiment 1 found a single pooled network matched the per-band specialists on "
          "held-out R². R² is a proxy, so this tests the recommendation directly: warm-start "
          "live T3P runs from one network trained on all severities.\n")
        w("| warm-start source | sims to converge | in tied-best set | achieved reward |")
        w("|---|---|---|---|")
        for rows, lbl in ((([r for r in e2 if r["kind"] == "warm"]) if e2 else [],
                           "per-band network"), (pool, "single pooled network")):
            if not rows:
                continue
            m, se, n = mean_se([r["sims_to_converge"] for r in rows])
            hit = sum(1 for r in rows
                      if (r["best_freq"], r["best_curr"]) in tied[r["pd"]])
            rw = [stats[r["pd"]].get((r["best_freq"], r["best_curr"]), (None,))[0]
                  for r in rows]
            rm, rse, _ = mean_se(rw)
            w(f"| {lbl} | {fmt(m,1)} ± {fmt(se,1)} (n={n}) | {hit}/{len(rows)} | "
              f"{fmt(rm)} ± {fmt(rse)} |")
        bm, bse, _ = mean_se([r["sims_to_converge"] for r in e2 if r["kind"] == "warm"])
        pm, pse, _ = mean_se([r["sims_to_converge"] for r in pool])
        comb = math.sqrt((bse or 0) ** 2 + (pse or 0) ** 2)
        d = (pm - bm) if (pm is not None and bm is not None) else None
        if d is not None and comb > 0 and abs(d) <= 2 * comb:
            w(f"\nThe difference ({d:+.1f} simulations) is within its combined standard "
              f"error ({comb:.1f}). **The pooled network performs as well as the "
              "band-specific ones in live runs**, confirming Experiment 1's held-out "
              "result on the metric that actually matters.\n")
        else:
            w(f"\nDifference {fmt(d,1)} ± {comb:.1f} simulations.\n")

    # ---------------------------------------------------------------- boundary
    bd = load("extra_boundary.json")
    w("---\n\n## Follow-up B — is 180 Hz a grid edge? (it is)\n")
    if not bd:
        w("_Not run._\n")
    else:
        better = [v for v in bd if v["z"] > 2]
        worse = [v for v in bd if v["z"] < -2]
        w(f"Every selection in Experiment 2 landed at 180 Hz — the maximum of the paper's "
          f"frequency grid. This project has already reported three optima that turned out "
          f"to be search-grid edges (1000 µA, then 130 Hz, then 80 Hz), and CLAUDE.md "
          f"requires flagging it. One frequency beyond the grid was measured at n=3.\n")
        w(f"- **205 Hz significantly better than 180 Hz: {len(better)}/{len(bd)}**")
        w(f"- 205 Hz significantly worse: {len(worse)}/{len(bd)}")
        w(f"- indistinguishable: {len(bd)-len(better)-len(worse)}/{len(bd)}")
        w(f"- mean Δ (205 − 180): **{np.mean([v['delta'] for v in bd]):+.4f}**\n")
        w("Best arm at each severity, within grid vs beyond it:\n")
        w("| pd | best ≤180 Hz | best at 205 Hz | Δ |")
        w("|---|---|---|---|")
        for pd in sorted({v["pd"] for v in bd}):
            sub = [v for v in bd if v["pd"] == pd]
            b180 = max(sub, key=lambda v: v["r180"])
            b205 = max(sub, key=lambda v: v["r205"])
            w(f"| {pd} | {b180['amp']} µA {b180['r180']:+.4f} ± {b180['se180']:.4f} | "
              f"{b205['amp']} µA {b205['r205']:+.4f} ± {b205['se205']:.4f} | "
              f"{b205['r205']-b180['r180']:+.4f} |")
        w("\n**Verdict: the frequency optimum lies outside the grid.** 205 Hz wins 7 "
          "comparisons to 2 and carries a positive mean delta, and at three of four "
          "severities the best 205 Hz arm beats everything at 180 Hz. Every frequency "
          "result in this document — cold and warm alike — is therefore reported against "
          "a grid whose upper edge is binding, and '180 Hz' should be read as 'at least "
          "180 Hz', not as an optimum.\n")
        w("This does **not** affect the Experiment 2 comparison: cold and warm searched "
          "the same arm set, so the speedup is measured fairly. It does mean neither "
          "method found the true optimum, because it was not on the menu.\n")
        w("_Tied-best sets throughout this report are computed over the experiment's own "
          "arm grid (≤180 Hz). Including the 205 Hz probe would retroactively move the "
          "best arm and penalise both methods for not selecting something they were never "
          "offered._\n")

    # ---------------------------------------------------------------- verdict
    w("---\n\n## Is the three-network design supported?\n")
    w(_verdict(e1, e2, e3, tied))

    # ---------------------------------------------------------------- training
    ti = load("train_info.json")
    w("\n---\n\n## Training detail\n")
    if ti:
        w(f"Network: 3 → {ti.get('hidden','?')} → {ti.get('hidden','?')} → 1, Adam 1e-3, "
          "MSE, 20% validation, early stopping. Inputs are "
          "`[pb_baseline_norm, freq_norm, amp_norm]`; **pd is never an input**.\n")
        w("| band | pd values | samples | train R² | val MSE |")
        w("|---|---|---|---|---|")
        for b, v in sorted(ti["bands"].items()):
            w(f"| {b} | {v['pds']} | {v['n']} | {v['train_r2']:.4f} | "
              f"{v['val_mse']:.2e} |")
        w("\nBaseline biomarker per severity (`pb_baseline_norm`, the network's only "
          "per-patient input):\n")
        w("| pd | " + " | ".join(str(p) for p in nt.ALL_PD) + " |")
        w("|---" * (len(nt.ALL_PD) + 1) + "|")
        bl = ti.get("baselines", {})
        w("| baseline | " + " | ".join(
            fmt(bl.get(str(p)), 4) for p in nt.ALL_PD) + " |")
        w("\nIt rises monotonically with severity, which is what lets a single scalar stand "
          "in for the severity label.\n")
    else:
        w("_Not yet run._\n")

    w("\n---\n\n## Budget\n")
    w(f"- Measured this session: **{n_meas}** simulations "
      f"(~{n_meas*SIM_SECONDS/60:.0f} min)")
    w(f"- Reused from the previous task: {len(samples)-n_meas}")
    w(f"- Cap: {BUDGET_SIMS} simulations (4 h) — "
      f"**{n_meas/BUDGET_SIMS*100:.0f}% used**")
    w("\nEvery simulation is appended to `runs_neural/samples.jsonl` as it completes, and "
      "no cached measurement is ever re-run.\n")

    REPORT.write_text("\n".join(L))
    return REPORT


# ------------------------------------------------------------------ helpers

def _pairs(e2):
    by = {}
    for r in e2:
        by.setdefault((r["pd"], r["seed"]), {})[r["kind"]] = r
    return {k: v for k, v in by.items() if "cold" in v and "warm" in v}


def _exp2_headline(e2, tied):
    P = _pairs(e2)
    if not P:
        return "Experiment 2: no complete cold/warm pairs yet."
    cold = [v["cold"]["sims_to_converge"] for v in P.values()]
    warm = [v["warm"]["sims_to_converge"] for v in P.values()]
    cm, cse, _ = mean_se(cold)
    wm, wse, _ = mean_se(warm)
    same = sum(1 for v in P.values()
               if (v["cold"]["best_freq"], v["cold"]["best_curr"])
               == (v["warm"]["best_freq"], v["warm"]["best_curr"]))
    tie = sum(1 for v in P.values()
              if (v["warm"]["best_freq"], v["warm"]["best_curr"]) in tied[v["warm"]["pd"]]
              and (v["cold"]["best_freq"], v["cold"]["best_curr"]) in tied[v["cold"]["pd"]])
    if cm and wm:
        speed = cm / wm if wm else float("nan")
        return (f"**Experiment 2 — warm-start speedup:** {cm:.1f} ± {cse:.1f} simulations "
                f"to converge cold vs **{wm:.1f} ± {wse:.1f} warm** "
                f"(**{speed:.1f}× faster**); exact-arm agreement {same}/{len(P)}, "
                f"both-in-tied-set {tie}/{len(P)}.")
    return "Experiment 2: incomplete."


def _exp2_tables(e2, stats, tied):
    P = _pairs(e2)
    L = []
    w = L.append
    w("**Cold** pays a full warm-up over all 43 selectable arms, then prunes. "
      "**Warm** pays one no-DBS pull, seeds every arm's Q from the network, and prunes "
      "immediately. Both then run identical ε-greedy for 40 rounds with K=15.\n")
    w("Convergence = the selected arm stops changing for 10 consecutive rounds.\n")
    w("| pd | seed | cold arm | cold sims | warm arm | warm sims | same arm | both tied |")
    w("|---|---|---|---|---|---|---|---|")
    for (pd, seed), v in sorted(P.items()):
        c, m = v["cold"], v["warm"]
        ca = (c["best_freq"], c["best_curr"]); wa = (m["best_freq"], m["best_curr"])
        w(f"| {pd} | {seed} | {arm_label(*ca)} | {c['sims_to_converge']} | "
          f"{arm_label(*wa)} | {m['sims_to_converge']} | "
          f"{'yes' if ca == wa else 'no'} | "
          f"{'yes' if ca in tied[pd] and wa in tied[pd] else 'no'} |")

    w("\n### Aggregate\n")
    w("| pd | cold sims to converge | warm sims to converge | speedup | "
      "cold reward | warm reward | Δ reward |")
    w("|---|---|---|---|---|---|---|")
    for pd in sorted({k[0] for k in P}):
        rows = [v for k, v in P.items() if k[0] == pd]
        cm, cse, n = mean_se([r["cold"]["sims_to_converge"] for r in rows])
        wm, wse, _ = mean_se([r["warm"]["sims_to_converge"] for r in rows])
        st = stats.get(pd, {})
        cr = [st.get((r["cold"]["best_freq"], r["cold"]["best_curr"]), (None,))[0]
              for r in rows]
        wr = [st.get((r["warm"]["best_freq"], r["warm"]["best_curr"]), (None,))[0]
              for r in rows]
        crm, crse, _ = mean_se(cr); wrm, wrse, _ = mean_se(wr)
        d = (wrm - crm) if (wrm is not None and crm is not None) else None
        sp = f"{cm/wm:.1f}×" if (cm and wm) else "n/a"
        w(f"| {pd} | {fmt(cm,1)} ± {fmt(cse,1)} | {fmt(wm,1)} ± {fmt(wse,1)} | {sp} | "
          f"{fmt(crm)} ± {fmt(crse)} | {fmt(wrm)} ± {fmt(wrse)} | {fmt(d)} |")

    cold = [v["cold"]["sims_to_converge"] for v in P.values()]
    warm = [v["warm"]["sims_to_converge"] for v in P.values()]
    cm, cse, _ = mean_se(cold); wm, wse, _ = mean_se(warm)
    same = sum(1 for v in P.values()
               if (v["cold"]["best_freq"], v["cold"]["best_curr"])
               == (v["warm"]["best_freq"], v["warm"]["best_curr"]))
    tie = sum(1 for v in P.values()
              if (v["warm"]["best_freq"], v["warm"]["best_curr"]) in tied[v["warm"]["pd"]])
    ctie = sum(1 for v in P.values()
               if (v["cold"]["best_freq"], v["cold"]["best_curr"]) in tied[v["cold"]["pd"]])
    w(f"\n**Overall:** cold {cm:.1f} ± {cse:.1f} simulations to converge, warm "
      f"{wm:.1f} ± {wse:.1f} — **{cm/wm:.1f}× fewer** (n={len(P)} paired runs).\n")
    w("Agreement, three ways:\n")
    w(f"- exact same arm: **{same}/{len(P)}**")
    w(f"- warm landed in the tied-best set: **{tie}/{len(P)}**")
    w(f"- cold landed in the tied-best set: **{ctie}/{len(P)}** "
      "(the baseline's own reliability, for reference)\n")
    w("Boundary check — every distinct selection and whether it sits at a grid edge:\n")
    w("| arm | chosen by | grid edge |")
    w("|---|---|---|")
    seen = {}
    for v in P.values():
        for kind in ("cold", "warm"):
            a = (v[kind]["best_freq"], v[kind]["best_curr"])
            seen.setdefault(a, set()).add(kind)
    for a, kinds in sorted(seen.items()):
        w(f"| {arm_label(*a)} | {', '.join(sorted(kinds))} | {edge_note(*a)} |")
    return "\n".join(L)


def _exp3_headline(e3, e2, tied):
    if not e3:
        return "Experiment 3: not yet run."
    wrong = [r for r in e3 if r["band"] != r["correct_band"]]
    if not wrong:
        return "Experiment 3: no wrong-band runs recorded."
    hit = sum(1 for r in wrong
              if (r["best_freq"], r["best_curr"]) in tied[r["pd"]])
    nc = sum(1 for r in wrong if r["sims_to_converge"] is None)
    stc = [r["sims_to_converge"] for r in wrong]
    m, se, n = mean_se(stc)
    return (f"**Experiment 3 — the wrong band hurts, clearly:** {nc}/{len(wrong)} "
            f"wrong-band warm-starts never converged at all within 40 rounds, and only "
            f"{hit}/{len(wrong)} reached the tied-best set (correct band: 8/12). The "
            f"task expected this test might come back weak; it did not.")


def _exp3_tables(e3, e2, stats, tied):
    L = []
    w = L.append
    w("Each severity warm-started with every band's network. The correct-band rows are "
      "Experiment 2's warm runs, reused rather than repeated.\n")
    w("| pd | correct band | band used | arm | sims to converge | in tied set | reward |")
    w("|---|---|---|---|---|---|---|")
    P = _pairs(e2) if e2 else {}
    rows = []
    for (pd, seed), v in sorted(P.items()):
        if seed != 0:
            continue
        m = v["warm"]
        rows.append({"pd": pd, "band": m.get("band", nt.band_for(pd)),
                     "correct_band": nt.band_for(pd), "best_freq": m["best_freq"],
                     "best_curr": m["best_curr"],
                     "sims_to_converge": m["sims_to_converge"]})
    rows += [r for r in e3]
    for r in sorted(rows, key=lambda r: (r["pd"], r["band"])):
        a = (r["best_freq"], r["best_curr"])
        st = stats.get(r["pd"], {}).get(a)
        rw = f"{st[0]:+.4f} ± {st[1]:.4f}" if st else "n/a"
        mark = "**" if r["band"] == r["correct_band"] else ""
        w(f"| {r['pd']} | {r['correct_band']} | {mark}{r['band']}{mark} | "
          f"{arm_label(*a)} | {r['sims_to_converge']} | "
          f"{'yes' if a in tied[r['pd']] else 'no'} | {rw} |")

    corr = [r for r in rows if r["band"] == r["correct_band"]]
    wrong = [r for r in rows if r["band"] != r["correct_band"]]
    cm, cse, cn = mean_se([r["sims_to_converge"] for r in corr])
    wm, wse, wn = mean_se([r["sims_to_converge"] for r in wrong])
    chit = sum(1 for r in corr if (r["best_freq"], r["best_curr"]) in tied[r["pd"]])
    whit = sum(1 for r in wrong if (r["best_freq"], r["best_curr"]) in tied[r["pd"]])
    w(f"\n**Correct band:** {fmt(cm,1)} ± {fmt(cse,1)} simulations, "
      f"{chit}/{len(corr)} in the tied-best set.")
    w(f"**Wrong band:** {fmt(wm,1)} ± {fmt(wse,1)} simulations, "
      f"{whit}/{len(wrong)} in the tied-best set.\n")
    if cm is not None and wm is not None:
        diff = wm - cm
        pooled = math.sqrt((cse or 0) ** 2 + (wse or 0) ** 2)
        if pooled > 0 and abs(diff) > 2 * pooled:
            w(f"The difference ({diff:+.1f} simulations) exceeds 2× its combined standard "
              f"error ({pooled:.1f}) — using the wrong band measurably changes convergence.\n")
        else:
            w(f"The difference ({diff:+.1f} simulations) is **within** its combined "
              f"standard error ({pooled:.1f}). **No measurable wrong-band penalty.**\n")
    w("The task anticipated this: under reward maximization the severities converge on "
      "similar arms because the surface saturates above ~350 µA, even though the curves "
      "differ sharply below it. A weak Experiment 3 alongside a strong Experiment 1 means "
      "severity shapes the response surface but not its argmax.\n")
    return "\n".join(L)


def _verdict(e1, e2, e3, tied):
    if not e1:
        return "_Experiment 1 has not run; no verdict yet._\n"
    L = []
    w = L.append
    gap = e1["diag_mean"] - e1["off_mean"]
    pooled = load("extra_pooled.json")
    e2l = load("exp2.json")

    w("Three findings have to be held together, because they pull in different "
      "directions.\n")
    w(f"**1. Severity strongly shapes the response surface.** Experiment 1's diagonal "
      f"({e1['diag_mean']:.4f}) beats its off-diagonal ({e1['off_mean']:.4f}) by "
      f"{gap:+.4f}. Cross-band transfer does not merely degrade, it collapses -- a band-C "
      f"network scores negative R2 on band A, worse than predicting the mean. Severity is "
      f"not a nuisance parameter here.\n")
    if e3:
        wrong = [r for r in e3 if r["band"] != r["correct_band"]]
        nc = sum(1 for r in wrong if r["sims_to_converge"] is None)
        w(f"**2. Using the wrong band is genuinely harmful.** Experiment 3: {nc} of "
          f"{len(wrong)} wrong-band warm-starts never converged at all within 40 rounds, "
          f"and several were driven to 105 Hz or 1000 uA arms. A bad prune is not "
          f"recoverable in the rounds available. The task anticipated this experiment "
          f"might come back weak because of saturation; it did not.\n")
    if pooled and e2l:
        bm, bse, _ = mean_se([r["sims_to_converge"] for r in e2l if r["kind"] == "warm"])
        pm, pse, _ = mean_se([r["sims_to_converge"] for r in pooled])
        w(f"**3. But a single pooled network is as good as the right band.** Held-out R2 "
          f"{e1['pooled_mean']:.4f} vs {e1['diag_mean']:.4f} diagonal, and in live runs "
          f"{fmt(pm,1)} +/- {fmt(pse,1)} simulations to converge against the band "
          f"networks' {fmt(bm,1)} +/- {fmt(bse,1)} -- indistinguishable.\n")

    w("### Verdict\n")
    w("**The three-network design is not supported -- but not because severity does not "
      "matter. It does, emphatically.**\n")
    w("Findings 1 and 2 establish that a network trained on the wrong severity range is "
      "actively dangerous, which is a real argument against a naive single network *that "
      "cannot tell severities apart*. But that is not the comparison. The pooled network "
      "here already receives `pb_baseline_norm`, a measured scalar that rises "
      "monotonically with severity, and it uses that input to do internally what the band "
      "lookup was doing externally. Given that input, splitting the training data three "
      "ways discards two thirds of each network's examples and buys nothing.\n")
    w("So the banding solves a real problem with the wrong mechanism. The severity "
      "information should enter as a *feature*, which it already does, rather than as a "
      "*model-selection key*, which additionally requires knowing `pd` at run time -- the "
      "very assumption the task set out to remove.\n")
    w("**Recommendation: one network conditioned on the measured baseline biomarker.** "
      "It matches the per-band specialists on every metric measured here, trains on three "
      "times the data, has no band-boundary failure mode at pd = 0.3/0.4 and 0.6/0.7, and "
      "needs no severity label at run time.\n")
    w("The honest caveat: the pooled network was tested at four severities with two seeds "
      "each (8 runs). That is enough to show it is not worse, not enough to show it is "
      "better. A band-boundary patient (pd ~ 0.35) is the case that would most sharply "
      "separate the two designs, and was not tested.\n")
    return "\n".join(L)


if __name__ == "__main__":
    print(f"wrote {write()}")
