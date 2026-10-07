"""
run_seeds.py  -  Week 5: repeat the whole pipeline with several random seeds and report mean +- std,
                 plus a sensitivity check of the quantile-tracking step size (eta).

For each seed s:
    1. manhole_sim.py   --seed s   -> new random scenarios
    2. forecast.py      --seed s   -> new split + new LSTM / GBR training
    3. week4_adaptive.py --seed s --eta e   for each eta in --etas

Then all week4_metrics.csv files are combined.

Which eta is reported: the PRE-REGISTERED default (--main-eta, 0.05), fixed before any results
were seen. The script also prints what tuning on the TEST split alone would pick, as a diagnostic:
in-distribution data cannot tell you how fast to adapt to an unseen shift, so that choice can be
unsafe under shift. Report both in the paper.

Outputs (runs/)
    runs/seed{s}/...              every intermediate result, so nothing has to be re-run
    runs/all_metrics.csv          every seed x eta x method x split x horizon
    runs/summary_mean_std.csv     mean and std over seeds
    runs/seeds_plot.png           trade-off plot with error bars (shift and test, 10 min)

Run (from the folder with the other scripts)
    python run_seeds.py                       # seeds 0 1 2, eta 0.02 0.05 0.1  (~10-15 min on a laptop)
    python run_seeds.py --seeds 0 1 2 3 4     # more seeds
    python run_seeds.py --skip-existing       # resume after an interruption
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ALPHA = 0.05
MAIN_METHODS = ["point", "split_cp", "aci", "aci_reset", "aci_rc", "qt", "norm_cp", "norm_aci", "norm_qt"]


def run(cmd):
    print("  $", " ".join(cmd))
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def pipeline(seeds, etas, root, skip):
    for s in seeds:
        base = os.path.join(root, f"seed{s}")
        data, w2 = os.path.join(base, "data"), os.path.join(base, "week2")
        t0 = time.time()
        print(f"\n=== seed {s} ===")
        if not (skip and os.path.exists(os.path.join(data, "timeseries.csv"))):
            run(["manhole_sim.py", "--seed", str(s), "--out", data])
        if not (skip and os.path.exists(os.path.join(w2, "predictions.csv"))):
            run(["forecast.py", "--data", data, "--out", w2, "--seed", str(s)])
        for e in etas:
            w4 = os.path.join(base, f"week4_eta{e}")
            if not (skip and os.path.exists(os.path.join(w4, "week4_metrics.csv"))):
                run(["week4_adaptive.py", "--data", data, "--pred", w2, "--out", w4,
                     "--seed", str(s), "--eta", str(e)])
        print(f"  seed {s} done in {time.time() - t0:.0f}s")


def collect(seeds, etas, root):
    frames = []
    for s in seeds:
        for e in etas:
            f = os.path.join(root, f"seed{s}", f"week4_eta{e}", "week4_metrics.csv")
            d = pd.read_csv(f)
            d["seed"], d["eta"] = s, e
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def pick_eta(allm, etas):
    """Choose eta on TEST only: norm_qt at 10 min, missed danger <= alpha, max usability."""
    t = (allm[(allm.split == "test") & (allm.method == "norm_qt") & (allm.H_min == 10)]
         .groupby("eta")[["missed_danger", "usable"]].mean())
    ok = t[t.missed_danger <= ALPHA]
    chosen = (ok if len(ok) else t).usable.idxmax()
    print("\nEta selection on TEST split (norm_qt, 10 min, mean over seeds):")
    print(t.round(3).to_string())
    print(f"-> test-only tuning picks eta = {chosen}")
    return chosen


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def report(allm, eta, root, n_seeds):
    summ = (allm.groupby(["eta", "split", "H_min", "method"])[["coverage", "missed_danger", "usable"]]
            .agg(["mean", "std"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(root, "summary_mean_std.csv"), index=False, float_format="%.4f")

    for split in ("test", "shift"):
        print(f"\n=== {split}, eta = {eta}, mean ± std over {n_seeds} seeds ===")
        for H in (5, 10, 20):
            d = summ[(summ.eta == eta) & (summ.split == split) & (summ.H_min == H)].set_index("method")
            d = d.reindex(MAIN_METHODS)
            tab = pd.DataFrame({
                "coverage": [fmt(a, b) for a, b in zip(d.coverage_mean, d.coverage_std)],
                "missed_danger": [fmt(a, b) for a, b in zip(d.missed_danger_mean, d.missed_danger_std)],
                "usable": [fmt(a, b) for a, b in zip(d.usable_mean, d.usable_std)],
            }, index=d.index)
            print(f"-- H = {H} min --")
            print(tab.to_string())

    print("\nSensitivity of norm_qt and qt to eta (10 min, mean over seeds):")
    sens = summ[(summ.H_min == 10) & summ.method.isin(["qt", "norm_qt"])].pivot_table(
        index=["method", "eta"], columns="split", values=["missed_danger_mean", "usable_mean"])
    print(sens.round(3).to_string())
    return summ


def plot(summ, eta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("tab10")
    d = summ[(summ.eta == eta) & (summ.H_min == 10)]
    for i, m in enumerate(MAIN_METHODS):
        for split, mk, face in (("test", "o", "none"), ("shift", "s", cmap(i))):
            r = d[(d.split == split) & (d.method == m)].iloc[0]
            ax.errorbar(r.missed_danger_mean, r.usable_mean, xerr=r.missed_danger_std,
                        yerr=r.usable_std, fmt=mk, mfc=face, mec=cmap(i), ecolor=cmap(i),
                        capsize=2, ms=7, label=m if split == "shift" else None)
        r = d[(d.split == "shift") & (d.method == m)].iloc[0]
        ax.annotate(m, (r.missed_danger_mean, r.usable_mean), fontsize=7, xytext=(5, 3),
                    textcoords="offset points")
    ax.axvline(ALPHA, color="r", ls=":", lw=1, label=f"target {ALPHA}")
    ax.set_xlabel("missed danger (said 'safe', gas went over the limit)")
    ax.set_ylabel("usable (said 'safe' when it really was)")
    ax.set_title(f"Safe-for-10-min decision, mean ± std over seeds (eta={eta})\n"
                 "filled = shift, hollow = test; goal: top-left")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nplot saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    ap.add_argument("--root", default="runs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--main-eta", type=float, default=0.05, help="pre-registered eta to report")
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)

    pipeline(a.seeds, a.etas, a.root, a.skip_existing)
    allm = collect(a.seeds, a.etas, a.root)
    allm.to_csv(os.path.join(a.root, "all_metrics.csv"), index=False, float_format="%.4f")
    picked = pick_eta(allm, a.etas)
    eta = a.main_eta
    print(f"(test-only tuning would pick eta = {picked}; reporting pre-registered eta = {eta})")
    summ = report(allm, eta, a.root, len(a.seeds))
    plot(summ, eta, os.path.join(a.root, "seeds_plot.png"))


if __name__ == "__main__":
    main()
