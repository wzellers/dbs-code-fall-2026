"""Validation gates for the neural warm-start. No MATLAB; runs in seconds.

Gate 4 matters most: if a perfect network does not produce near-instant convergence, the
warm-start mechanism is broken and Experiment 2 is uninterpretable.
"""
import math
import sys

import numpy as np

import dbs_t3p as t3p
import neural_t3p as nt


def main():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("\nGATE 1 -- band lookup at the boundaries")
    cases = [(0.0, "A"), (0.3, "A"), (0.4, "B"), (0.6, "B"), (0.7, "C"), (1.0, "C"),
             (0.301, "B"), (0.601, "C")]
    bad = [(p, nt.band_for(p), e) for p, e in cases if nt.band_for(p) != e]
    check("boundaries 0.3/0.4/0.6/0.7 map correctly", not bad, str(bad) if bad else
          "0.0-0.3->A, 0.4-0.6->B, 0.7-1.0->C")
    check("pd=0.0 (healthy) is in band A and not excluded",
          nt.band_for(0.0) == "A" and 0.0 in nt.BANDS["A"])

    print("\nGATE 6 -- cached rewards reconstruct from their components")
    # Run before gate 2 because the training data depends on it.
    samples = nt.load_samples()
    if not samples:
        check("samples present", False, "run --import first")
        return 1
    import dbs_selector as ds
    worst, n = 0.0, 0
    for s in samples:
        if "r1" not in s:
            continue
        det = ds.deterministic_terms(int(s["freq"]), int(s["curr"]))[2]
        recon = -0.7 * float(s["r1"]) + det
        worst = max(worst, abs(recon - float(s["reward"]))); n += 1
    # Compare against the tightest arm SE in the dataset, not an arbitrary constant.
    by = {}
    for s in samples:
        by.setdefault((round(float(s["pd"]), 3), int(s["freq"]), int(s["curr"])), []
                      ).append(float(s["reward"]))
    ses = [float(np.std(v, ddof=1)) / math.sqrt(len(v)) for v in by.values() if len(v) > 1]
    tightest = min(ses) if ses else float("nan")
    check("reward = -0.7*r1 + (0.1*r3 - 0.2*r2) to within 1% of the tightest arm SE",
          worst < 0.01 * tightest,
          f"worst {worst:.2e} vs tightest SE {tightest:.2e} "
          f"({worst/tightest*100:.3f}%) over {n} samples")

    print("\nGATE 5 -- pd never enters the network input")
    base, _ = nt.baseline_by_pd(samples)
    X, y, meta = nt.make_xy(samples, base)
    check("feature matrix has exactly 3 columns", X.shape[1] == 3, str(nt.FEATURE_NAMES))
    # A pd value appearing verbatim as a feature would be the obvious leak; check every
    # column against every pd, allowing that a legitimate feature may coincide numerically.
    leaks = []
    for j, nm in enumerate(nt.FEATURE_NAMES):
        col = X[:, j]
        for pd in nt.ALL_PD:
            rows = [i for i, m in enumerate(meta) if m[0] == pd]
            if rows and np.allclose(col[rows], pd, atol=1e-9):
                leaks.append((nm, pd))
    check("no input column equals pd for any severity", not leaks, str(leaks))
    pds_present = sorted({m[0] for m in meta})
    check("feature rows carry >1 distinct baseline (network can distinguish patients)",
          len({round(base[p], 6) for p in pds_present}) > 1,
          f"{len(pds_present)} pd values, baselines "
          f"{[round(base[p],4) for p in pds_present]}")

    print("\nGATE 2 -- a network fits its own training data")
    pd_fit = 0.1 if 0.1 in pds_present else pds_present[0]
    Xa, ya, ma = nt.make_xy(samples, base, pds={pd_fit})
    net = nt.RewardNet(seed=0).fit(Xa, ya, seed=0)
    pred = net.predict(Xa)
    # Per-arm: does the prediction land within 2 SE of that arm's measured mean?
    arms = {}
    for (pd, f, c), yy, pp in zip(ma, ya, pred):
        arms.setdefault((f, c), {"y": [], "p": []})
        arms[(f, c)]["y"].append(float(yy)); arms[(f, c)]["p"].append(float(pp))
    within, tot, worst_arm = 0, 0, None
    worst_z = 0.0
    for k, v in arms.items():
        if len(v["y"]) < 2 or not nt.is_safe(k[1]):
            continue   # harmful-band arms are never selected; their fit cannot matter
        mu = float(np.mean(v["y"]))
        se = float(np.std(v["y"], ddof=1)) / math.sqrt(len(v["y"]))
        z = abs(float(np.mean(v["p"])) - mu) / se if se > 0 else 0.0
        tot += 1; within += (z <= 2.0)
        if z > worst_z:
            worst_z, worst_arm = z, k
    frac = within / tot if tot else 0.0
    check(f"network predicts its own pd={pd_fit} training rewards within 2 SE",
          frac >= 0.80, f"{within}/{tot} selectable arms ({frac*100:.0f}%), "
          f"worst {worst_arm} at {worst_z:.1f} SE, R2={nt.r2_score(ya, pred):.3f}")
    check(f"network is 3-{nt.HIDDEN_UNITS}-{nt.HIDDEN_UNITS}-1 as configured",
          net.n_params() == 3*nt.HIDDEN_UNITS + nt.HIDDEN_UNITS
          + nt.HIDDEN_UNITS**2 + nt.HIDDEN_UNITS + nt.HIDDEN_UNITS + 1,
          f"{net.n_params()} parameters (task specified 32 units; see DECISIONS.md)")

    print("\nGATE 3 -- warm-started T3P never selects the harmful band")
    cfg = t3p.T3PConfig(top_k=15, eps_start=0.2, eps_decay=0.025, seed=1)
    src = nt.CachedSource(samples, seed=3)
    viol = []
    for pd in pds_present:
        for seed in (0, 1, 2):
            r = nt.run_warm(src, pd, cfg, net, n_rounds=40, seed=seed,
                            pb_baseline=base[pd])
            if 0 < r["best_curr"] < nt.AMP_MIN_SAFE:
                viol.append((pd, seed, r["best_curr"]))
            for e in r["log"]:
                if 0 < e["curr"] < nt.AMP_MIN_SAFE:
                    viol.append((pd, seed, "pulled", e["curr"]))
    check("no selection or pull in (0, 300) uA", not viol, f"{len(viol)} violations")
    check("harmful-band arms exist in the grid but are excluded from the arm set",
          any(0 < c < nt.AMP_MIN_SAFE for _, c in nt.ARMS)
          and not any(0 < c < nt.AMP_MIN_SAFE for _, c in nt.SAFE_ARMS),
          f"{len(nt.ARMS)} arms, {len(nt.SAFE_ARMS)} selectable")

    print("\nGATE 4 -- a perfect oracle network converges near-instantly")
    # "The true best arm" is not well posed here: in the saturated region several arms are
    # statistically tied (at pd=0.4, five arms sit within 2 SE of the best, separated by
    # 0.0004-0.0009 against SEs of 0.002-0.004). CLAUDE.md forbids treating a difference
    # smaller than its error bar as real, so the criterion is membership in the tied set.
    rows = []
    for pd in pds_present:
        oracle = nt.OracleNet(samples, pd)
        tied = nt.tied_best_set(samples, pd)
        for seed in (0, 1, 2):
            r = nt.run_warm(src, pd, cfg, oracle, n_rounds=40, seed=seed,
                            pb_baseline=base[pd])
            rows.append({"pd": pd, "seed": seed, "stc": r["sims_to_converge"],
                         "hit": (r["best_freq"], r["best_curr"]) in tied,
                         "ntied": len(tied)})
    stcs = sorted(r["stc"] for r in rows if r["stc"] is not None)
    med = float(np.median(stcs)) if stcs else float("nan")
    check("oracle warm-start converges in <= 5 simulations (median)",
          bool(stcs) and med <= 5,
          f"median {med:.0f}, mean {np.mean(stcs):.1f}, max {max(stcs)} "
          f"over {len(rows)} runs" if stcs else "never converged")
    hit = sum(r["hit"] for r in rows) / len(rows)
    check("oracle warm-start lands on an arm statistically tied with the best",
          hit >= 0.9, f"{hit*100:.0f}% of runs; tied-set sizes "
          f"{sorted({r['ntied'] for r in rows})}")

    print("\n" + "=" * 68)
    if fails:
        print(f"GATES FAILED -- {len(fails)}: {fails}")
    else:
        print("ALL GATES PASSED -- safe to spend MATLAB time")
    print("=" * 68 + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
