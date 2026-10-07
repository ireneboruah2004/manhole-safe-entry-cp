"""
week4_adaptive.py  -  Week 4: keep the safety guarantee under gas bursts WITHOUT the robot
                      saying "unsafe" all the time.

Week 3 found:
    split CP (static)  -> usable, but misses danger under shift (guarantee breaks)
    ACI (adaptive)     -> keeps the guarantee under shift, but usable only ~20% of the time,
                          because after a few misses its bound jumps to infinity and stays there,
                          and that state is carried into the next manholes.

Fixes tested here (all built on the same week-2 LSTM forecasts and week-3 score):
    aci_reset    : ACI, but alpha_t is reset to alpha at every new manhole
    aci_rc       : reset + clip: alpha_t never drops below 0.001 -> bound always finite
    norm_cp      : static CP with a burst-aware NORMALISED score s / sigma(x).
                   sigma(x) = typical error size predicted from the sensors right now
                   (bottom-sensor rate of rise, bottom-middle gap, levels). The bound widens only
                   when a burst seems to be starting. sigma is learned on half of the calibration
                   scenarios; the conformal quantile uses the other half (keeps the guarantee).
    norm_aci     : normalised score + ACI with reset and clip
    qt           : conformal P-control / quantile tracking (Angelopoulos, Candes, Tibshirani 2023):
                   the bound's quantile q itself is nudged after each outcome, q += eta*(err - alpha),
                   so it moves gradually in score units instead of jumping to infinity
    norm_qt      : quantile tracking on the normalised score

Baselines repeated from week 3: point, split_cp, aci.

Settings (gamma, eta, clip) are fixed in advance and NOT tuned on the shift set, to avoid
leakage. Tune them only on the 'test' split if you change them.

Outputs (results_week4/)
    week4_metrics.csv     coverage / missed danger / usable per method, split, horizon
    week4_plot.png        left: missed danger vs usable (shift, 10 min) - the trade-off figure
                          right: the same shift scenario as week 3, old ACI vs the new bounds

Run
    python week4_adaptive.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from conformal import LIMIT, WINDOWS_MIN, cp_quantile, load, score, upper
from manhole_sim import SAMPLE_EVERY

ALPHA_FLOOR = 0.001          # aci clip: bound uses at most the 99.9% calibration quantile
SIGMA_FLOOR = 0.02           # smallest allowed sigma (log units)


# ---------------------------------------------------------------- features for sigma(x)

def add_sensor_features(df, data_dir):
    ts = pd.read_csv(os.path.join(data_dir, "timeseries.csv"))
    lag = 60 // SAMPLE_EVERY
    feats = []
    for sid, d in ts.groupby("scenario_id"):
        d = d.sort_values("t_s")
        f = pd.DataFrame({"scenario_id": sid, "t_s": d.t_s})
        for k in (1, 2, 3):
            s = d[f"sensor_z{k}"]
            f[f"lz{k}"] = np.log1p(s.clip(lower=0))
            f[f"rise{k}"] = (s - s.shift(lag)).fillna(0.0)
        f["gap12"] = d.sensor_z1 - d.sensor_z2
        feats.append(f)
    return df.merge(pd.concat(feats), on=["scenario_id", "t_s"], how="left")


FEATS = ["lz1", "lz2", "lz3", "rise1", "rise2", "rise3", "gap12"]


def fit_sigma(d_fit, H, seed):
    x = d_fit[FEATS + [f"pred_max{H}"]].to_numpy()
    y = np.abs(score(d_fit[f"true_max{H}"], d_fit[f"pred_max{H}"]).to_numpy())
    m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                      max_leaf_nodes=15, random_state=seed)
    m.fit(x, y)
    return m


def sigma(model, d, H):
    return np.maximum(model.predict(d[FEATS + [f"pred_max{H}"]].to_numpy()), SIGMA_FLOOR)


# ---------------------------------------------------------------- online methods

def online(d, H, cal_scores, sig, alpha, method, gamma, eta, reset, clip):
    """Delayed-feedback online calibration over consecutive manholes.
    method 'aci': adapt alpha_t, q = calibration quantile at 1 - alpha_t
    method 'qt' : adapt q directly, q += eta * (err - alpha)"""
    n = H * 60 // SAMPLE_EVERY
    cal_sorted = np.sort(cal_scores)
    q0 = cp_quantile(cal_scores, alpha)
    a_t, q_t = alpha, q0
    out = np.empty(len(d))
    pos = 0
    for _, g in d.groupby("scenario_id", sort=False):
        idx = np.arange(pos, pos + len(g))
        pm = g[f"pred_max{H}"].to_numpy()
        tm = g[f"true_max{H}"].to_numpy()
        sg = sig[idx]
        if reset:
            a_t, q_t = alpha, q0
        ub = np.empty(len(g))

        def feedback(k):
            nonlocal a_t, q_t
            err = float(tm[k] > ub[k])
            if method == "aci":
                a_t += gamma * (alpha - err)
                if clip:
                    a_t = min(max(a_t, ALPHA_FLOOR), 0.5)
            else:
                q_t += eta * (err - alpha)

        for k in range(len(g)):
            if k - n >= 0:
                feedback(k - n)
            if method == "aci":
                level = 1 - a_t
                q = np.inf if level >= 1 else np.quantile(cal_sorted, max(level, 0.0))
            else:
                q = q_t
            ub[k] = upper(pm[k], q * sg[k])
        for k in range(max(len(g) - n, 0), len(g)):
            feedback(k)
        out[idx] = ub
        pos += len(g)
    return out


# ---------------------------------------------------------------- evaluation

def evaluate(df, alpha, gamma, eta, seed):
    rows, keep = [], {}
    for H in WINDOWS_MIN:
        d = df[df[f"true_max{H}"].notna()]
        cal = d[d.split == "cal"]
        cal_ids = np.sort(cal.scenario_id.unique())
        fit_ids, q_ids = cal_ids[::2], cal_ids[1::2]          # half for sigma, half for quantile
        cal_fit, cal_q = cal[cal.scenario_id.isin(fit_ids)], cal[cal.scenario_id.isin(q_ids)]

        s_cal = score(cal[f"true_max{H}"], cal[f"pred_max{H}"]).to_numpy()
        sig_model = fit_sigma(cal_fit, H, seed)
        s_norm = (score(cal_q[f"true_max{H}"], cal_q[f"pred_max{H}"]).to_numpy()
                  / sigma(sig_model, cal_q, H))
        q_split, q_norm = cp_quantile(s_cal, alpha), cp_quantile(s_norm, alpha)

        for split in ("test", "shift"):
            e = d[d.split == split]
            pm, tm = e[f"pred_max{H}"].to_numpy(), e[f"true_max{H}"].to_numpy()
            ones, sg = np.ones(len(e)), sigma(sig_model, e, H)
            ubs = {
                "point": pm,
                "split_cp": upper(pm, q_split),
                "aci": online(e, H, s_cal, ones, alpha, "aci", gamma, eta, reset=False, clip=False),
                "aci_reset": online(e, H, s_cal, ones, alpha, "aci", gamma, eta, reset=True, clip=False),
                "aci_rc": online(e, H, s_cal, ones, alpha, "aci", gamma, eta, reset=True, clip=True),
                "qt": online(e, H, s_cal, ones, alpha, "qt", gamma, eta, reset=False, clip=False),
                "norm_cp": upper(pm, q_norm * sg),
                "norm_aci": online(e, H, s_norm, sg, alpha, "aci", gamma, eta, reset=True, clip=True),
                "norm_qt": online(e, H, s_norm, sg, alpha, "qt", gamma, eta, reset=False, clip=False),
            }
            danger = tm > LIMIT
            for m, ub in ubs.items():
                safe = ub <= LIMIT
                rows.append(dict(H_min=H, split=split, method=m,
                                 coverage=float(np.mean(tm <= ub)),
                                 missed_danger=float((safe & danger).sum() / max(danger.sum(), 1)),
                                 false_safe=float(np.mean(safe & danger)),
                                 usable=float((safe & ~danger).sum() / max((~danger).sum(), 1))))
                keep[(H, split, m)] = (e.scenario_id.to_numpy(), e.t_s.to_numpy(), ub, tm)
        print(f"H={H:2d}  q_split={q_split:.3f}  q_norm={q_norm:.3f}  "
              f"sigma fitted on {len(fit_ids)} scenarios, quantile on {len(q_ids)}")
    return pd.DataFrame(rows), keep


METHODS = ["point", "split_cp", "aci", "aci_reset", "aci_rc", "qt", "norm_cp", "norm_aci", "norm_qt"]


def plot(res, keep, alpha, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    r = res[res.H_min == 10]
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(METHODS):
        for split, mk, fill in (("test", "o", "none"), ("shift", "s", cmap(i))):
            row = r[(r.split == split) & (r.method == m)].iloc[0]
            ax[0].scatter(row.missed_danger, row.usable, marker=mk, s=60, edgecolors=cmap(i),
                          facecolors=fill, label=m if split == "shift" else None)
        a = r[(r.split == "shift") & (r.method == m)].iloc[0]
        ax[0].annotate(m, (a.missed_danger, a.usable), fontsize=7, xytext=(4, 3),
                       textcoords="offset points")
    ax[0].axvline(alpha, color="r", ls=":", lw=1)
    ax[0].set_xlabel("missed danger (said 'safe', gas went over the limit)")
    ax[0].set_ylabel("usable (said 'safe' when it really was)")
    ax[0].set_title("Safe-for-10-min decision. Filled = shift, hollow = test.\nGoal: top-left")
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(0, 1.02)

    sids, ts_, _, tm = keep[(10, "shift", "point")]
    sid = 300 if 300 in set(sids) else sids[0]
    sel = sids == sid
    t = ts_[sel] / 60
    ax[1].plot(t, tm[sel], "k", lw=2, label="true max, next 10 min")
    for m, c in (("aci", "tab:purple"), ("split_cp", "tab:green"), ("norm_qt", "tab:orange"),
                 ("aci_rc", "tab:blue")):
        ax[1].plot(t, np.minimum(keep[(10, "shift", m)][2][sel], 8 * LIMIT), color=c, lw=1.3, label=m)
    ax[1].axhline(LIMIT, color="r", ls=":", label="limit (placeholder)")
    ax[1].set_xlabel("time the decision is made [min since lid opened]")
    ax[1].set_ylabel("H2S [ppm]")
    ax[1].set_title(f"Shift scenario #{sid}: bound below red line = 'safe for 10 min'")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--pred", default="results_week2")
    ap.add_argument("--out", default="results_week4")
    ap.add_argument("--model", default="lstm", choices=["lstm", "gbr", "persist"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.01, help="ACI step size")
    ap.add_argument("--eta", type=float, default=0.05, help="quantile-tracking step size")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = add_sensor_features(load(a.data, a.pred, a.model), a.data)
    res, keep = evaluate(df, a.alpha, a.gamma, a.eta, a.seed)
    res.to_csv(os.path.join(a.out, "week4_metrics.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 160)
    for split in ("test", "shift"):
        print(f"\n=== {split}  (alpha = {a.alpha}) ===")
        t = res[res.split == split].pivot_table(index="method", columns="H_min",
                                                values=["coverage", "missed_danger", "usable"])
        print(t.reindex(METHODS).round(3).to_string())
    plot(res, keep, a.alpha, os.path.join(a.out, "week4_plot.png"))


if __name__ == "__main__":
    main()
