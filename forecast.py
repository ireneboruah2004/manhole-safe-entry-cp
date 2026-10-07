"""
forecast.py  -  Week 2: forecast H2S at the worker's breathing zone, 1-20 minutes ahead.

Input  : the last 5 minutes of the 3 sensor readings (from manhole_sim.py output)
Output : true breathing-zone concentration at horizons 1, 2, 5, 10, 15, 20 minutes ahead

Models compared
    persist : "it stays as it is now" (latest reading of the 1.5 m sensor) - the naive baseline
    gbr     : gradient boosting on summary features of the window (one model per horizon)
    lstm    : small LSTM on the raw 5-minute window, all horizons at once

Data split (by scenario, never by time step, so no leakage between splits)
    train-regime scenarios -> 200 train / 50 calibration / 50 test   (calibration is for week 3)
    shift-regime scenarios -> all 100 used only as a hard test set

Outputs (in --out folder)
    predictions.csv     scenario_id, split, t_s, horizon_min, y_true, pred_persist, pred_gbr, pred_lstm
                        (calibration + test + shift rows; this is the input to week 3's conformal step)
    metrics.csv         MAE / RMSE / missed-danger rate per model, split and horizon
    forecast_plot.png   left: error vs horizon; right: one shift scenario, 10-min-ahead forecasts
    splits.csv          which scenario went to which split

Run (from the folder that contains manhole_sim.py and data/)
    pip install torch scikit-learn
    python forecast.py
    python forecast.py --epochs 5          # quicker test run

"missed danger" = the truth is above the exposure limit but the point forecast says it is below.
This is the failure week 3's conformal upper bound is meant to fix.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from manhole_sim import EXPOSURE_LIMIT_PPM, SAMPLE_EVERY

WINDOW_MIN = 5
HORIZONS_MIN = (1, 2, 5, 10, 15, 20)
SENSORS = ("sensor_z1", "sensor_z2", "sensor_z3")
W = WINDOW_MIN * 60 // SAMPLE_EVERY                      # 30 samples
H_STEPS = [h * 60 // SAMPLE_EVERY for h in HORIZONS_MIN]  # 6, 12, 30, 60, 90, 120 samples


# ---------------------------------------------------------------- data

def make_splits(meta, seed):
    rng = np.random.default_rng(seed)
    train_ids = meta.loc[meta.regime == "train", "scenario_id"].to_numpy().copy()
    rng.shuffle(train_ids)
    split = {sid: "train" for sid in train_ids[:200]}
    split.update({sid: "cal" for sid in train_ids[200:250]})
    split.update({sid: "test" for sid in train_ids[250:]})
    split.update({sid: "shift" for sid in meta.loc[meta.regime == "shift", "scenario_id"]})
    return split


def build_windows(ts, split, stride_train=3):
    """Cut every scenario into (window, targets) pairs. Targets at horizons beyond the
    60-min record are NaN and are ignored in training and metrics."""
    X, Y, info = [], [], []
    for sid, d in ts.groupby("scenario_id", sort=True):
        s = d[list(SENSORS)].to_numpy(np.float32)
        y = d["truth_bz"].to_numpy(np.float32)
        t = d["t_s"].to_numpy()
        stride = stride_train if split[sid] == "train" else 1
        for k in range(W - 1, len(d), stride):
            tgt = [y[k + h] if k + h < len(d) else np.nan for h in H_STEPS]
            X.append(s[k - W + 1: k + 1])
            Y.append(tgt)
            info.append((sid, split[sid], t[k]))
    info = pd.DataFrame(info, columns=["scenario_id", "split", "t_s"])
    return np.stack(X), np.array(Y, np.float32), info


def summary_features(X, t_s):
    """Hand-made features for gradient boosting: per sensor last value, mean, max,
    slope over the window, slope over the last minute; plus bottom-minus-middle gap and time."""
    last = X[:, -1, :]
    feats = [last, X.mean(1), X.max(1),
             (X[:, -1, :] - X[:, 0, :]) / WINDOW_MIN,
             (X[:, -1, :] - X[:, -7, :]),
             (last[:, 0] - last[:, 1])[:, None],
             np.asarray(t_s, np.float32)[:, None] / 60.0]
    return np.log1p(np.clip(np.concatenate(feats[:3], 1), 0, None)), np.concatenate(feats[3:], 1)


def gbr_features(X, t_s):
    a, b = summary_features(X, t_s)
    return np.concatenate([a, b], 1)


# ---------------------------------------------------------------- models

def fit_gbr(Xf, Y, seed):
    models = []
    for j in range(Y.shape[1]):
        ok = ~np.isnan(Y[:, j])
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          random_state=seed)
        m.fit(Xf[ok], np.log1p(Y[ok, j]))
        models.append(m)
    return models


def predict_gbr(models, Xf):
    return np.stack([np.expm1(m.predict(Xf)) for m in models], 1).clip(0, None)


def fit_lstm(X, Y, Xval, Yval, epochs, seed, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(SENSORS), 64, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(HORIZONS_MIN)))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    def prep(Xa, Ya):
        x = torch.tensor(np.log1p(np.clip(Xa, 0, None)), dtype=torch.float32)
        y = torch.tensor(np.log1p(np.nan_to_num(Ya, nan=0.0)), dtype=torch.float32)
        m = torch.tensor(~np.isnan(Ya), dtype=torch.float32)
        return x, y, m

    x, y, m = prep(X, Y)
    xv, yv, mv = prep(Xval, Yval)
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            b = perm[i:i + 256]
            pred = net(x[b].to(device))
            loss = (((pred - y[b].to(device)) ** 2) * m[b].to(device)).sum() / m[b].sum().to(device)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(xv[i:i + 2048].to(device)).cpu() for i in range(0, len(xv), 2048)])
            vloss = float(((((pv - yv) ** 2) * mv).sum() / mv.sum()))
        if vloss < best:
            best, best_state = vloss, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  lstm epoch {ep + 1:2d}/{epochs}  train {loss.item():.4f}  val {vloss:.4f}")
    net.load_state_dict(best_state)
    return net


def predict_lstm(net, X, device):
    import torch
    net.eval()
    x = torch.tensor(np.log1p(np.clip(X, 0, None)), dtype=torch.float32)
    with torch.no_grad():
        out = torch.cat([net(x[i:i + 4096].to(device)).cpu() for i in range(0, len(x), 4096)])
    return np.expm1(out.numpy()).clip(0, None)


# ---------------------------------------------------------------- evaluation

def metrics(pred_df):
    rows = []
    for (split, h), d in pred_df.groupby(["split", "horizon_min"]):
        danger = d.y_true > EXPOSURE_LIMIT_PPM
        for m in ("persist", "gbr", "lstm"):
            e = d[f"pred_{m}"] - d.y_true
            missed = float(((d[f"pred_{m}"] <= EXPOSURE_LIMIT_PPM) & danger).sum() / max(danger.sum(), 1))
            rows.append(dict(split=split, horizon_min=h, model=m, MAE=float(e.abs().mean()),
                             RMSE=float(np.sqrt((e ** 2).mean())), missed_danger=missed,
                             n=len(d), n_danger=int(danger.sum())))
    return pd.DataFrame(rows)


def plot(pred_df, met, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    style = {"persist": ("tab:gray", "--"), "gbr": ("tab:blue", "-"), "lstm": ("tab:orange", "-")}
    for split, marker in (("test", "o"), ("shift", "s")):
        for m, (c, ls) in style.items():
            d = met[(met.split == split) & (met.model == m)]
            ax[0].plot(d.horizon_min, d.MAE, color=c, ls=ls, marker=marker, ms=4,
                       label=f"{m} ({split})")
    ax[0].set_xlabel("forecast horizon [min]")
    ax[0].set_ylabel("MAE [ppm]")
    ax[0].set_title("Forecast error vs horizon")
    ax[0].legend(fontsize=7, ncol=2)

    sh = pred_df[(pred_df.split == "shift") & (pred_df.horizon_min == 10)]
    sid = sh.groupby("scenario_id").y_true.max().sort_values().index[-len(sh.scenario_id.unique()) // 4]
    d = sh[sh.scenario_id == sid].sort_values("t_s")
    tm = (d.t_s + 600) / 60
    ax[1].plot(tm, d.y_true, "k", lw=2, label="truth")
    for m, (c, ls) in style.items():
        ax[1].plot(tm, d[f"pred_{m}"], color=c, ls=ls, lw=1.2, label=m)
    ax[1].axhline(EXPOSURE_LIMIT_PPM, color="r", ls=":", lw=1, label="limit (placeholder)")
    ax[1].set_xlabel("time since lid opened [min]")
    ax[1].set_ylabel("H2S at breathing zone [ppm]")
    ax[1].set_title(f"Shift scenario #{sid}: forecasts made 10 min earlier")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot saved to {path}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_week2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    meta = pd.read_csv(os.path.join(a.data, "scenarios_meta.csv"))
    ts = pd.read_csv(os.path.join(a.data, "timeseries.csv"))
    split = make_splits(meta, a.seed)
    pd.Series(split, name="split").rename_axis("scenario_id").to_csv(os.path.join(a.out, "splits.csv"))

    t0 = time.time()
    X, Y, info = build_windows(ts, split)
    print(f"windows: {len(X)}  ({info.split.value_counts().to_dict()})  [{time.time() - t0:.0f}s]")
    tr = (info.split == "train").to_numpy()
    cal = (info.split == "cal").to_numpy()

    print("training gradient boosting ...")
    Xf = gbr_features(X, info.t_s)
    gbr = fit_gbr(Xf[tr], Y[tr], a.seed)
    print("training LSTM ...")
    net = fit_lstm(X[tr], Y[tr], X[cal], Y[cal], a.epochs, a.seed, device)

    keep = (info.split != "train").to_numpy()
    preds = {
        "persist": np.repeat(X[keep][:, -1, 1:2], len(HORIZONS_MIN), 1),
        "gbr": predict_gbr(gbr, Xf[keep]),
        "lstm": predict_lstm(net, X[keep], device),
    }
    rows = []
    base = info[keep].reset_index(drop=True)
    for j, h in enumerate(HORIZONS_MIN):
        d = base.copy()
        d["horizon_min"] = h
        d["y_true"] = Y[keep][:, j]
        for m, p in preds.items():
            d[f"pred_{m}"] = p[:, j]
        rows.append(d)
    pred_df = pd.concat(rows, ignore_index=True).dropna(subset=["y_true"])
    pred_df.to_csv(os.path.join(a.out, "predictions.csv"), index=False, float_format="%.4f")

    met = metrics(pred_df)
    met.to_csv(os.path.join(a.out, "metrics.csv"), index=False, float_format="%.4f")
    show = met[met.split.isin(["test", "shift"])].pivot_table(
        index=["split", "horizon_min"], columns="model", values=["MAE", "missed_danger"])
    print("\nMAE [ppm] and missed-danger rate (truth above limit, forecast below):")
    print(show.round(3).to_string())
    plot(pred_df, met, os.path.join(a.out, "forecast_plot.png"))
    print(f"done in {time.time() - t0:.0f}s, outputs in {a.out}/")


if __name__ == "__main__":
    main()
