"""
conformal.py  -  Week 3: conformal upper bounds and the "safe for the next H minutes" decision.

Question the robot answers at each moment t:
    "Will H2S at the breathing zone stay below the limit for the whole next H minutes?"
    H in {5, 10, 20} minutes.

Approach
    1. From week 2's point forecasts (default: LSTM) take the predicted MAXIMUM over the next H minutes
       (max of the forecasts at horizons <= H).
    2. Conformalize that maximum directly: score s = log(1+true_max) - log(1+pred_max)
       (log scale, because errors grow with concentration - the heteroscedasticity issue).
    3. Upper bound UB = (1+pred_max) * exp(q) - 1, with q a conformal quantile of the calibration scores.
    4. Decision: declare SAFE for H minutes  <=>  UB <= exposure limit.
       If the bound holds with probability >= 1 - alpha, then
       P(declared safe AND limit actually exceeded) <= alpha.

Methods compared
    point       : no bound, trust the forecast (pred_max <= limit)
    margin      : common rule of thumb, forecast must stay under half the limit
    split_cp    : split conformal, one quantile for everything
    mondrian_cp : one quantile per predicted-level group (low / medium / high), valid within each group
    traj_cp     : conformal on each calibration SCENARIO's worst score -> the bound holds over a whole
                  scenario at once (the most rigorous version; also the most conservative)
    aci         : adaptive conformal inference (Gibbs & Candes 2021), alpha updated online as outcomes
                  arrive H minutes later; carried across consecutive manholes (scenarios)

Reported per split (test = same conditions as training, shift = bigger bursts and drift)
    coverage              P(true_max <= UB)                       target >= 1 - alpha
    false_safe            P(declared safe AND exceeded)           guaranteed <= alpha (marginal)
    false_safe|danger     among windows that DID exceed: share declared safe   (NOT guaranteed)
    usable                among windows that stayed safe: share declared safe  (how useful the robot is)

Run (needs data/ from manhole_sim.py and results_week2/ from forecast.py)
    python conformal.py
    python conformal.py --alpha 0.1 --model gbr
"""

import argparse
import os

import numpy as np
import pandas as pd

from manhole_sim import EXPOSURE_LIMIT_PPM, SAMPLE_EVERY

WINDOWS_MIN = (5, 10, 20)
FORECAST_H = (1, 2, 5, 10, 15, 20)
LIMIT = EXPOSURE_LIMIT_PPM
GROUP_EDGES = (0.3 * LIMIT, LIMIT)       # mondrian groups by predicted max: low / medium / high


# ---------------------------------------------------------------- data

def load(data_dir, pred_dir, model):
    ts = pd.read_csv(os.path.join(data_dir, "timeseries.csv"))
    pr = pd.read_csv(os.path.join(pred_dir, "predictions.csv"))
    wide = pr.pivot_table(index=["scenario_id", "split", "t_s"], columns="horizon_min",
                          values=f"pred_{model}").reset_index()
    wide.columns = [c if isinstance(c, str) else f"p{c}" for c in wide.columns]

    rows = []
    for sid, d in ts.groupby("scenario_id"):
        y = d.truth_bz.to_numpy()
        t = d.t_s.to_numpy()
        out = pd.DataFrame({"scenario_id": sid, "t_s": t})
        for H in WINDOWS_MIN:
            n = H * 60 // SAMPLE_EVERY
            mx = np.full(len(y), np.nan)
            if len(y) > n:
                mx[: len(y) - n] = np.lib.stride_tricks.sliding_window_view(y[1:], n).max(1)
            out[f"true_max{H}"] = mx
        rows.append(out)
    truth = pd.concat(rows, ignore_index=True)
    df = wide.merge(truth, on=["scenario_id", "t_s"], how="left")
    for H in WINDOWS_MIN:
        cols = [f"p{h}" for h in FORECAST_H if h <= H]
        df[f"pred_max{H}"] = df[cols].max(axis=1)
    return df.sort_values(["split", "scenario_id", "t_s"]).reset_index(drop=True)


# ---------------------------------------------------------------- conformal pieces

def score(true_max, pred_max):
    return np.log1p(true_max) - np.log1p(pred_max)


def upper(pred_max, q):
    return np.expm1(np.log1p(pred_max) + q)


def cp_quantile(s, alpha):
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(s))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if k > len(s) else s[k - 1]


def group_of(pred_max):
    return np.digitize(pred_max, GROUP_EDGES)


def run_aci(d, H, cal_scores, alpha, gamma):
    """Online ACI with delayed feedback: the outcome of a bound issued at t is known at t + H.
    Scenarios are processed one after another (a robot inspecting manholes in sequence)."""
    n = H * 60 // SAMPLE_EVERY
    cal_sorted = np.sort(cal_scores)
    a_t = alpha
    ub_all = np.empty(len(d))
    pos = 0
    for _, g in d.groupby("scenario_id", sort=False):
        pm = g[f"pred_max{H}"].to_numpy()
        tm = g[f"true_max{H}"].to_numpy()
        ub = np.empty(len(g))
        for k in range(len(g)):
            if k - n >= 0:                                  # outcome of bound issued n steps ago
                err = float(tm[k - n] > ub[k - n])
                a_t += gamma * (alpha - err)
            level = 1 - a_t
            # level >= 1 -> infinite bound ("not safe"); level <= 0 -> smallest calibration score,
            # never -inf (a collapsed bound would wrongly declare "safe")
            q = np.inf if level >= 1 else np.quantile(cal_sorted, max(level, 0.0))
            ub[k] = upper(pm[k], q)
        for k in range(max(len(g) - n, 0), len(g)):        # remaining outcomes at scenario end
            a_t += gamma * (alpha - float(tm[k] > ub[k]))
        ub_all[pos: pos + len(g)] = ub
        pos += len(g)
    return ub_all


# ---------------------------------------------------------------- evaluation

def evaluate(df, alpha, gamma):
    results, bounds = [], {}
    for H in WINDOWS_MIN:
        ok = df[f"true_max{H}"].notna()
        d = df[ok].copy()
        cal = d[d.split == "cal"]
        s_cal = score(cal[f"true_max{H}"], cal[f"pred_max{H}"]).to_numpy()

        q_split = cp_quantile(s_cal, alpha)
        g_cal = group_of(cal[f"pred_max{H}"].to_numpy())
        q_group = {g: cp_quantile(s_cal[g_cal == g], alpha) for g in range(len(GROUP_EDGES) + 1)}
        per_scen = pd.Series(s_cal).groupby(cal.scenario_id.to_numpy()).max().to_numpy()
        q_traj = cp_quantile(per_scen, alpha)

        for split in ("test", "shift"):
            e = d[d.split == split]
            pm = e[f"pred_max{H}"].to_numpy()
            tm = e[f"true_max{H}"].to_numpy()
            ubs = {
                "point": pm,
                "margin": pm * 2.0,                              # safe only if pm <= limit / 2
                "split_cp": upper(pm, q_split),
                "mondrian_cp": upper(pm, np.array([q_group[g] for g in group_of(pm)])),
                "traj_cp": upper(pm, q_traj),
                "aci": run_aci(e, H, s_cal, alpha, gamma),
            }
            danger = tm > LIMIT
            for m, ub in ubs.items():
                safe = ub <= LIMIT
                results.append(dict(
                    H_min=H, split=split, method=m,
                    coverage=float(np.mean(tm <= ub)),
                    false_safe=float(np.mean(safe & danger)),
                    false_safe_given_danger=float((safe & danger).sum() / max(danger.sum(), 1)),
                    usable=float((safe & ~danger).sum() / max((~danger).sum(), 1)),
                    n=len(e), n_danger=int(danger.sum())))
                bounds[(H, split, m)] = (e[["scenario_id", "t_s"]].to_numpy(), ub)
        print(f"H={H:2d} min  calibration windows={len(s_cal)}  "
              f"q_split={q_split:.3f}  q_traj={q_traj:.3f}  "
              f"q_groups={ {k: round(v, 3) for k, v in q_group.items()} }")
    return pd.DataFrame(results), bounds


def decisions(df, bounds, method):
    """The robot's message at each moment: longest H (of 5/10/20 min) declared safe, else 0."""
    out = df[df.split.isin(["test", "shift"])][["scenario_id", "split", "t_s"]].copy()
    out["safe_minutes"] = 0
    for H in sorted(WINDOWS_MIN):
        for split in ("test", "shift"):
            keys, ub = bounds[(H, split, method)]
            safe = pd.DataFrame({"scenario_id": keys[:, 0], "t_s": keys[:, 1], f"ub{H}": ub})
            out = out.merge(safe, on=["scenario_id", "t_s"], how="left", suffixes=("", "_dup"))
            if f"ub{H}_dup" in out:
                out[f"ub{H}"] = out[f"ub{H}"].fillna(out.pop(f"ub{H}_dup"))
        out.loc[out[f"ub{H}"] <= LIMIT, "safe_minutes"] = H
    return out


def plot(df, res, bounds, alpha, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["point", "margin", "split_cp", "mondrian_cp", "traj_cp", "aci"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    r = res[res.H_min == 10]
    x = np.arange(len(methods))
    for i, (split, c) in enumerate((("test", "tab:blue"), ("shift", "tab:red"))):
        v = [r[(r.split == split) & (r.method == m)].false_safe_given_danger.iloc[0] for m in methods]
        u = [r[(r.split == split) & (r.method == m)].usable.iloc[0] for m in methods]
        ax[0].bar(x + (i - 0.5) * 0.38, v, 0.38, color=c, alpha=0.8, label=f"missed danger ({split})")
        ax[0].plot(x + (i - 0.5) * 0.38, u, "o", color=c, mfc="white", label=f"usable ({split})")
    ax[0].set_xticks(x, methods, rotation=20)
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Safe-for-10-min decision: missed danger (bars) vs usefulness (dots)")
    ax[0].legend(fontsize=7)

    H, split = 10, "shift"
    keys, _ = bounds[(H, split, "point")]
    sids = pd.Series(keys[:, 0])
    e = df[(df.split == split) & df[f"true_max{H}"].notna()]
    pick = e.groupby("scenario_id")[f"true_max{H}"].max()
    sid = pick[(pick > 1.5 * LIMIT) & (pick < 6 * LIMIT)].index[0] if (
        (pick > 1.5 * LIMIT) & (pick < 6 * LIMIT)).any() else pick.idxmax()
    sel = (sids == sid).to_numpy()
    tm = keys[sel, 1] / 60
    ax[1].plot(tm, e[e.scenario_id == sid][f"true_max{H}"], "k", lw=2, label="true max, next 10 min")
    for m, c, ls in (("point", "tab:gray", "--"), ("split_cp", "tab:green", "-"), ("aci", "tab:purple", "-")):
        ax[1].plot(tm, np.minimum(bounds[(H, split, m)][1][sel], 8 * LIMIT), color=c, ls=ls, lw=1.3,
                   label={"point": "point forecast", "split_cp": "split CP bound",
                          "aci": "ACI bound"}[m])
    ax[1].axhline(LIMIT, color="r", ls=":", label="limit (placeholder)")
    ax[1].set_xlabel("time the decision is made [min since lid opened]")
    ax[1].set_ylabel("H2S [ppm]")
    ax[1].set_title(f"Shift scenario #{sid}: below the red line = 'safe for 10 min'")
    ax[1].legend(fontsize=7)
    fig.suptitle(f"target miss rate alpha = {alpha}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--pred", default="results_week2")
    ap.add_argument("--out", default="results_week3")
    ap.add_argument("--model", default="lstm", choices=["lstm", "gbr", "persist"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.01, help="ACI step size")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.data, a.pred, a.model)
    res, bounds = evaluate(df, a.alpha, a.gamma)
    res.to_csv(os.path.join(a.out, "conformal_metrics.csv"), index=False, float_format="%.4f")
    decisions(df, bounds, "split_cp").to_csv(os.path.join(a.out, "decisions_split_cp.csv"),
                                              index=False, float_format="%.3f")

    pd.set_option("display.width", 160)
    for split in ("test", "shift"):
        print(f"\n=== {split} (alpha = {a.alpha}, model = {a.model}) ===")
        t = res[res.split == split].pivot_table(
            index="method", columns="H_min",
            values=["coverage", "false_safe", "false_safe_given_danger", "usable"])
        print(t.reindex(["point", "margin", "split_cp", "mondrian_cp", "traj_cp", "aci"]).round(3).to_string())
    plot(df, res, bounds, a.alpha, os.path.join(a.out, "conformal_plot.png"))
    print(f"outputs in {a.out}/")


if __name__ == "__main__":
    main()
