"""Re-derive every number in docs/DETECTION_RESULTS.md from the committed caches.

No MATLAB, no simulation, about 30 seconds. Run it after touching pd_switch.py, after
moving a cache, or before quoting the document to anyone:

    python src/verify_detection.py

Each run's stimulation settings and switch time are read from the `settings` field stored
inside its own .npz rather than assumed, because two of them are not what a reader would
guess: runs_stim_a500 is the only stimulated sweep row at 180 Hz (the rest are 130 Hz), and
runs_stim_long_a3500 switches at 10 s, not 5 s. Scoring the long run with the default
switch_s=5.0 silently measures a changepoint in the middle of segment 1 and reports a
plausible-looking wrong latency.
"""

import json
import sys

import numpy as np

import paths
import pd_switch as ps
import recheck_stim as rs

TH_RAW, TH_PAR, KM_GAP = 7.0, 5.75, 0.13


def meta(name):
    z = np.load(paths.cache(name) / "seed1.npz")
    return json.loads(str(z["settings"]))


def switch_of(name):
    return meta(name)["hold_ms"] / 1000.0


def sem(a):
    return a.std(ddof=1) / np.sqrt(len(a))


def fired(r):
    """Null runs have no switch to catch, so a 'correct' detection is still a false alarm."""
    return r["ok"] + len(r["false"])


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {label:<44} got {got:>8.2f}   doc {want:>8.2f}")
    return ok


def main():
    results = []

    print("\n2. UNSTIMULATED BASELINE  (cache/runs_switch, 30 brains)")
    base = rs.load_cached("runs_switch")
    gaps, seps = ps.summarize(base)
    results += [
        check("gap %", gaps.mean(), 23.4, 0.1),
        check("gap sem", sem(gaps), 1.6, 0.1),
        check("separation", seps.mean(), 5.10, 0.02),
        check("separation sem", sem(seps), 0.36, 0.02),
        check("weakest brain", seps.min(), 1.67, 0.01),
    ]

    held = {k: v for k, v in base.items() if k > 10}
    paired = rs.pair_average(held)
    for label, det, series, kw, n_doc, lat_doc in [
        ("held-out cp raw", ps.detect_cp, held, dict(thresh=TH_RAW, min_size=4), 19, 1.25),
        ("held-out cp paired", ps.detect_cp, paired, dict(thresh=TH_PAR, min_size=3), 20, 1.68),
        ("held-out kmeans", ps.detect_kmeans, paired, dict(min_gap=KM_GAP, min_size=3), 17, 1.57),
    ]:
        r = ps.evaluate(series, det, **kw)
        results += [check(f"{label} caught", r["ok"], n_doc, 0),
                    check(f"{label} latency", r["latency"], lat_doc, 0.01)]

    print("\n3. UNDER STIMULATION  (settings and switch time read from each cache)")
    stim_doc = {
        "runs_stim_a0":         (5.04, 5.93, 10, 8, 1.98),
        "runs_stim_a250":       (3.93, 4.51,  9, 4, 2.40),
        "runs_stim_a500":       (5.50, 7.25, 10, 3, 1.82),
        "runs_stim_a1000":      (1.46, 3.17,  8, 4, 3.19),
        "runs_stim_a3500":      (1.43, 2.87,  9, 4, 3.29),
        "runs_stim_long_a3500": (2.01, 3.16, 10, 0, 2.62),
    }
    for name, (sr_d, sp_d, cp_d, km_d, lat_d) in stim_doc.items():
        m, sw = meta(name), switch_of(name)
        raw = rs.load_cached(name)
        par = rs.pair_average(raw)
        _, sr = ps.summarize(raw, switch_s=sw)
        _, sp = ps.summarize(par, switch_s=sw)
        rc = ps.evaluate(par, ps.detect_cp, switch_s=sw, thresh=TH_PAR, min_size=3)
        rk = ps.evaluate(par, ps.detect_kmeans, switch_s=sw, min_gap=KM_GAP, min_size=3)
        print(f"  [{m['freq']:>3.0f} Hz / {m['amp']:>4.0f} uA, switch {sw:>4.1f}s]")
        results += [
            check(f"{name} sep raw", sr.mean(), sr_d, 0.02),
            check(f"{name} sep paired", sp.mean(), sp_d, 0.02),
            check(f"{name} changepoint", rc["ok"], cp_d, 0),
            check(f"{name} kmeans", rk["ok"], km_d, 0),
            check(f"{name} latency", rc["latency"], lat_d, 0.01),
        ]

    print("\n   pre-correction raw counts, the numbers that looked like failure")
    for name, want in [("runs_stim_a1000", 1), ("runs_stim_a3500", 0)]:
        r = ps.evaluate(rs.load_cached(name), ps.detect_cp,
                        switch_s=switch_of(name), thresh=TH_RAW, min_size=4)
        results.append(check(f"{name} raw caught", r["ok"], want, 0))

    print("\n4. NULL RUNS  (pd never changes, so every firing is a false positive)")
    null_doc = {"runs_null_nostim": (1, 0, 1), "runs_null_stim180_500": (0, 7, 1)}
    for name, (cp_d, km_d, two_d) in null_doc.items():
        par = rs.pair_average(rs.load_cached(name))
        results += [
            check(f"{name} cp fired",
                  fired(ps.evaluate(par, ps.detect_cp, thresh=TH_PAR, min_size=3)), cp_d, 0),
            check(f"{name} kmeans fired",
                  fired(ps.evaluate(par, ps.detect_kmeans, min_gap=KM_GAP, min_size=3)), km_d, 0),
            check(f"{name} cp two-sided fired",
                  fired(ps.evaluate(par, ps.detect_cp2, thresh=TH_PAR, min_size=3)), two_d, 0),
        ]

    print("\n5. DIRECTION  (180 Hz / 500 uA - detect_cp is one-sided by construction)")
    dir_doc = {
        "runs_dir_0.3to0.7_stim180_500": (+79.1, +7.25, 10, 10, 1.82),
        "runs_dir_0.7to0.3_stim180_500": (-43.1, -7.59,  0, 10, 2.12),
    }
    for name, (g_d, s_d, one_d, two_d, lat_d) in dir_doc.items():
        par = rs.pair_average(rs.load_cached(name))
        g, s = ps.summarize(par)
        one = ps.evaluate(par, ps.detect_cp, thresh=TH_PAR, min_size=3)
        two = ps.evaluate(par, ps.detect_cp2, thresh=TH_PAR, min_size=3)
        results += [
            check(f"{name} gap %", g.mean(), g_d, 0.1),
            check(f"{name} separation", s.mean(), s_d, 0.02),
            check(f"{name} one-sided", one["ok"], one_d, 0),
            check(f"{name} two-sided", two["ok"], two_d, 0),
            check(f"{name} latency", two["latency"], lat_d, 0.01),
        ]

    print("\n6. SEVERITY LADDER FROM pd 0.5  (raw and pair-averaged separation)")
    ladder_doc = {
        ("nostim", "0.6"): (0.64, 0.81), ("nostim", "0.7"): (0.81, 1.03),
        ("nostim", "0.8"): (1.01, 1.31), ("nostim", "0.9"): (1.44, 1.95),
        ("stim180_500", "0.6"): (1.56, 2.24), ("stim180_500", "0.7"): (3.68, 5.27),
        ("stim180_500", "0.8"): (3.88, 5.79), ("stim180_500", "0.9"): (1.76, 2.47),
    }
    for (tag, end), (raw_d, par_d) in ladder_doc.items():
        raw = rs.load_cached(f"runs_ladder_{tag}_0.5to{end}")
        _, sr = ps.summarize(raw)
        _, sp = ps.summarize(rs.pair_average(raw))
        results += [check(f"ladder {tag} 0.5->{end} raw", sr.mean(), raw_d, 0.02),
                    check(f"ladder {tag} 0.5->{end} paired", sp.mean(), par_d, 0.02)]

    print("\n   the finding the ladder actually supports: equal step width, unequal signal")
    g9, _ = ps.summarize(rs.load_cached("runs_ladder_nostim_0.5to0.9"))
    diff = gaps.mean() - g9.mean()
    nsig = diff / np.sqrt(sem(gaps) ** 2 + sem(g9) ** 2)
    results += [check("0.5->0.9 gap %", g9.mean(), 5.5, 0.1),
                check("0.3->0.7 against 0.5->0.9, in sigma", nsig, 8.7, 0.1)]

    bad = len(results) - sum(results)
    print(f"\n{'=' * 72}\n{len(results)} checks, {bad} failed")
    if bad:
        print("docs/DETECTION_RESULTS.md disagrees with the caches. Fix one or the other.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
