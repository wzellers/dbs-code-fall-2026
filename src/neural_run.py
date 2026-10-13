"""Driver for the per-band neural warm-start experiments.

Stages write to disk as they finish, so an interrupted session still leaves results.

    --stage collect   training data for all 11 pd values
    --stage train     fit the three band networks (no MATLAB)
    --stage exp1      3x3 cross-band R^2 matrix (no MATLAB)
    --stage exp2      cold vs warm T3P convergence
    --stage exp3      wrong-band penalty
    --stage report    regenerate NEURAL_RESULTS.md from cached artifacts
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import paths

import dbs_t3p as t3p
import neural_t3p as nt

OUT = paths.cache(nt.RESULTS_DIR)
REPORT = paths.DOCS / "NEURAL_RESULTS.md"
SIM_SECONDS = 3.6
BUDGET_SIMS = 3700

EXP_PD = [0.1, 0.4, 0.7, 1.0]     # severities used for Experiments 2 and 3
N_ROUNDS = 40
TOP_K = 15


def cfg_for(seed=1):
    return t3p.T3PConfig(top_k=TOP_K, eps_start=0.2, eps_decay=0.025, seed=seed)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    return p


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def sims_used():
    return len([s for s in nt.load_samples() if s.get("src") != "runs_selector"])


def budget_line():
    n = sims_used()
    return (f"{n} simulations measured this session "
            f"(~{n*SIM_SECONDS/60:.0f} min of {BUDGET_SIMS*SIM_SECONDS/60:.0f} min budget, "
            f"{n/BUDGET_SIMS*100:.0f}%)")


# --------------------------------------------------------------------------------------

def stage_collect(n_rep=3, n_rep_nodbs=6, verbose=True):
    targets = []
    for pd in nt.ALL_PD:
        for f, c in nt.ARMS:
            targets.append((pd, f, c, n_rep_nodbs if c == 0 else n_rep))
    print(f"\n=== STAGE collect: {len(nt.ALL_PD)} pd x {len(nt.ARMS)} arms ===")
    src = nt.MatlabSource()
    try:
        done = nt.collect(src, targets, verbose=verbose)
    finally:
        src.close()
    print(f"  {done} new simulations. {budget_line()}")
    return done


def train_bands(samples=None, seed=0, verbose=True):
    samples = nt.load_samples() if samples is None else samples
    base, base_se = nt.baseline_by_pd(samples)
    nets, info = {}, {}
    for band, pds in nt.BANDS.items():
        pds_have = [p for p in pds if p in base]
        X, y, meta = nt.make_xy(samples, base, pds=set(pds_have))
        if len(X) < 20:
            if verbose:
                print(f"  band {band}: only {len(X)} samples, skipping")
            continue
        net = nt.RewardNet(seed=seed).fit(X, y, seed=seed)
        nets[band] = net
        info[band] = {"pds": pds_have, "n": int(len(X)),
                      "val_mse": net.history["val_mse"],
                      "epochs": net.history["epochs"],
                      "params": net.n_params(),
                      "train_r2": nt.r2_score(y, net.predict(X))}
        if verbose:
            print(f"  band {band}: pd {pds_have}  n={len(X)}  "
                  f"train R2={info[band]['train_r2']:.4f}  "
                  f"val MSE={net.history['val_mse']:.2e}")
    return nets, info, base, base_se


def stage_train(verbose=True):
    print("\n=== STAGE train ===")
    samples = nt.load_samples()
    nets, info, base, base_se = train_bands(samples, verbose=verbose)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "nets.pkl", "wb") as fh:
        pickle.dump({b: n.net.state_dict() for b, n in nets.items()}, fh)
    save("train_info.json", {"bands": info,
                             "baselines": {str(k): v for k, v in base.items()},
                             "baseline_se": {str(k): v for k, v in base_se.items()},
                             "hidden": nt.HIDDEN_UNITS})
    return nets, info, base


# --------------------------------------------------------------------------------------

def stage_exp1(verbose=True):
    """Cross-band validation R^2: does band specialization exist?"""
    print("\n=== EXPERIMENT 1: cross-band prediction ===")
    samples = nt.load_samples()
    base, _ = nt.baseline_by_pd(samples)
    rng = np.random.default_rng(0)

    # Hold out 20% of each band's samples, train on the rest, score every band's holdout.
    held, trainset = {}, {}
    for band, pds in nt.BANDS.items():
        pds_have = [p for p in pds if p in base]
        X, y, meta = nt.make_xy(samples, base, pds=set(pds_have))
        if len(X) < 20:
            continue
        idx = rng.permutation(len(X))
        k = max(1, int(0.2 * len(X)))
        held[band] = (X[idx[:k]], y[idx[:k]])
        trainset[band] = (X[idx[k:]], y[idx[k:]])

    nets = {}
    for band, (X, y) in trainset.items():
        nets[band] = nt.RewardNet(seed=0).fit(X, y, seed=0)

    matrix = {}
    for tb, net in nets.items():
        matrix[tb] = {}
        for eb, (Xh, yh) in held.items():
            matrix[tb][eb] = float(nt.r2_score(yh, net.predict(Xh)))

    # A single pooled network, as the control the task asks about.
    Xall = np.vstack([trainset[b][0] for b in trainset])
    yall = np.concatenate([trainset[b][1] for b in trainset])
    pooled = nt.RewardNet(seed=0).fit(Xall, yall, seed=0)
    pooled_r2 = {eb: float(nt.r2_score(yh, pooled.predict(Xh)))
                 for eb, (Xh, yh) in held.items()}

    diag = [matrix[b][b] for b in matrix]
    off = [matrix[a][b] for a in matrix for b in matrix[a] if a != b]
    res = {"matrix": matrix, "pooled": pooled_r2,
           "diag_mean": float(np.mean(diag)), "off_mean": float(np.mean(off)),
           "pooled_mean": float(np.mean(list(pooled_r2.values()))),
           "n_held": {b: int(len(held[b][1])) for b in held}}
    save("exp1.json", res)

    if verbose:
        print(f"  {'trained':>8} " + " ".join(f"{('eval '+b):>10}" for b in held))
        for tb in matrix:
            print(f"  {tb:>8} " + " ".join(f"{matrix[tb][eb]:>10.4f}" for eb in held))
        print(f"  {'pooled':>8} " + " ".join(f"{pooled_r2[eb]:>10.4f}" for eb in held))
        print(f"\n  diagonal mean {res['diag_mean']:.4f} vs "
              f"off-diagonal mean {res['off_mean']:.4f} "
              f"(gap {res['diag_mean']-res['off_mean']:+.4f})")
        print(f"  single pooled network mean {res['pooled_mean']:.4f}")
    return res


# --------------------------------------------------------------------------------------

def stage_exp2(seeds=(0, 1, 2), pds=None, verbose=True):
    """Cold vs warm T3P. The headline claim."""
    pds = EXP_PD if pds is None else pds
    print(f"\n=== EXPERIMENT 2: cold vs warm, {len(pds)} pd x {len(seeds)} seeds ===")
    samples = nt.load_samples()
    nets, info, base, _ = train_bands(samples, verbose=False)
    src = nt.MatlabSource()
    rows = load("exp2.json") or []
    done = {(r["pd"], r["seed"], r["kind"]) for r in rows}
    try:
        for pd in pds:
            band = nt.band_for(pd)
            for seed in seeds:
                for kind in ("cold", "warm"):
                    if (pd, seed, kind) in done:
                        continue
                    t0 = time.time()
                    if kind == "cold":
                        r = run_cold_logged(src, pd, seed)
                    else:
                        r = run_warm_logged(src, pd, seed, nets[band], band)
                    r["minutes"] = round((time.time() - t0) / 60, 2)
                    rows.append(r)
                    save("exp2.json", rows)       # after every run, not at stage end
                    if verbose:
                        print(f"  pd={pd} seed={seed} {kind:<4} -> "
                              f"{r['best_freq']}Hz/{r['best_curr']}  "
                              f"sims={r['sims']}  to_converge={r['sims_to_converge']}  "
                              f"({r['minutes']:.1f} min)", flush=True)
    finally:
        src.close()
    print(f"  {budget_line()}")
    return rows


def _strip(r):
    r = dict(r)
    r.pop("log", None)
    return r


def run_cold_logged(src, pd, seed):
    r = nt.run_cold(src, pd, cfg_for(seed + 1), n_rounds=N_ROUNDS, seed=seed)
    return _strip(r)


def run_warm_logged(src, pd, seed, net, band):
    r = nt.run_warm(src, pd, cfg_for(seed + 1), net, n_rounds=N_ROUNDS, seed=seed)
    r = _strip(r)
    r["band"] = band
    return r


# --------------------------------------------------------------------------------------

def stage_exp3(seeds=(0,), pds=None, verbose=True):
    """Warm-start each pd with every band's network. Does the wrong band hurt?"""
    pds = EXP_PD if pds is None else pds
    print(f"\n=== EXPERIMENT 3: wrong-band penalty ===")
    samples = nt.load_samples()
    nets, info, base, _ = train_bands(samples, verbose=False)
    src = nt.MatlabSource()
    rows = load("exp3.json") or []
    done = {(r["pd"], r["seed"], r["band"]) for r in rows}
    try:
        for pd in pds:
            correct = nt.band_for(pd)
            for band in sorted(nets):
                if band == correct:
                    continue    # reuse Experiment 2's correct-band runs
                for seed in seeds:
                    if (pd, seed, band) in done:
                        continue
                    r = run_warm_logged(src, pd, seed, nets[band], band)
                    r["correct_band"] = correct
                    rows.append(r)
                    save("exp3.json", rows)
                    if verbose:
                        print(f"  pd={pd} band={band} (correct {correct}) -> "
                              f"{r['best_freq']}Hz/{r['best_curr']}  "
                              f"to_converge={r['sims_to_converge']}", flush=True)
    finally:
        src.close()
    print(f"  {budget_line()}")
    return rows


# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["collect", "train", "exp1", "exp2", "exp3", "report"])
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    if args.stage == "collect":
        stage_collect(n_rep=args.n)
    elif args.stage == "train":
        stage_train()
    elif args.stage == "exp1":
        stage_exp1()
    elif args.stage == "exp2":
        stage_exp2()
    elif args.stage == "exp3":
        stage_exp3()
    elif args.stage == "report":
        import neural_report
        neural_report.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
