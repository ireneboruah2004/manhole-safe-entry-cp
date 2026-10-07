"""
manhole_sim.py  -  Week 1: synthetic H2S build-up data for a manhole shaft.

Model (1-D vertical, finite volume, backward Euler):
    dC/dt = d/dz( D dC/dz ) - w dC/dz
    C(z, t): H2S concentration in ppm, z = 0 at the sludge (bottom), z = L at the opening (top)

Boundary conditions
    bottom (z = 0): gas released from sludge, flux q(t) [ppm*m/s]
                    q(t) = q0 * (1 + small noise) * burst(t)
                    bursts = sludge disturbed by cleaning work (short, large spikes)
    top (z = L):    exchange with outside air through the open lid, -D dC/dz = h * C
                    plus optional upward fan velocity w [m/s] (forced ventilation)

Story of one scenario
    The lid has been closed, so gas has built up (initial profile C0(z)).
    At t = 0 the lid is opened (and maybe a fan switched on).
    We simulate 60 minutes. Sensors at 3 heights record noisy, lagged, drifting readings.
    The quantity to forecast is the TRUE concentration at the worker's breathing zone.

Two regimes
    train : few, moderate bursts, small sensor drift   (use for training + calibration)
    shift : more, larger bursts, larger sensor drift    (use ONLY for testing robustness)

Outputs (in --out folder)
    scenarios_meta.csv   one row per scenario with every sampled parameter
    timeseries.csv       scenario_id, t_s, sensor_z1..z3 (ppm), truth_bz (ppm), q_rel (burst factor)
    example_plot.png     4 example scenarios, for a visual sanity check

Run
    python manhole_sim.py                      # 300 train + 100 shift scenarios
    python manhole_sim.py --selftest           # numerical checks only
    python manhole_sim.py --n-train 50 --n-shift 20 --out data_small

ASSUMPTIONS (all parameter ranges are plausible guesses, not measured values;
state them as such in the paper and replace with literature values where you can):
    shaft depth 3 m, eddy diffusivity 2e-4 to 5e-3 m^2/s, lid exchange velocity 1e-3 to 1e-2 m/s,
    fan velocity 0 or 2e-3 to 1e-2 m/s, sensor first-order lag 30 s, sensor noise 5% + 0.2 ppm.
EXPOSURE_LIMIT_PPM is a placeholder: set it from a published standard (e.g. NIOSH / ACGIH for H2S)
and cite the source.
"""

import argparse
import os

import numpy as np
import pandas as pd

# ---------------- fixed settings ----------------
L = 3.0                          # shaft depth [m]
N = 60                           # number of cells
DZ = L / N
Z = (np.arange(N) + 0.5) * DZ    # cell centres [m]
DT = 1.0                         # time step [s]
T_END = 3600.0                   # 60 minutes
SAMPLE_EVERY = 10                # sensor sampling period [s]
SENSOR_Z = (0.3, 1.5, 2.7)       # sensor heights [m]
BREATHING_Z = 1.5                # worker breathing zone height above sludge [m]
SENSOR_TAU = 30.0                # sensor response lag [s]
EXPOSURE_LIMIT_PPM = 10.0        # PLACEHOLDER - set from a cited standard

REGIMES = {
    #           bursts/hour  burst size range     sensor drift range (ppm per hour)
    "train": dict(burst_rate=0.7, burst_mag=(3.0, 10.0), drift=(-0.5, 0.5)),
    "shift": dict(burst_rate=2.0, burst_mag=(10.0, 40.0), drift=(-2.0, 2.0)),
}


def cell_index(z):
    return int(np.clip(round(z / DZ - 0.5), 0, N - 1))


def build_matrix(D, w, h):
    """Return M such that dC/dt = M @ C + s (s = bottom source term)."""
    a = D / DZ**2
    b = w / DZ
    M = np.zeros((N, N))
    for i in range(N - 1):               # interior face between i and i+1
        M[i, i] += -a - b
        M[i, i + 1] += a
        M[i + 1, i + 1] += -a
        M[i + 1, i] += a + b
    M[N - 1, N - 1] += -h / DZ - b       # top: exchange with outside air + fan outflow
    return M


def simulate(D, w, h, C0, q_fn, t_end=T_END, dt=DT):
    """Backward Euler. Returns times and concentration field C[t, z]."""
    M = build_matrix(D, w, h)
    A_inv = np.linalg.inv(np.eye(N) - dt * M)
    steps = int(round(t_end / dt))
    C = C0.copy()
    out = np.empty((steps + 1, N))
    out[0] = C
    for n in range(1, steps + 1):
        rhs = C.copy()
        rhs[0] += dt * q_fn(n * dt) / DZ
        C = A_inv @ rhs
        out[n] = C
    return np.arange(steps + 1) * dt, out


def sample_scenario(rng, regime, sid):
    p = REGIMES[regime]
    D = 10 ** rng.uniform(np.log10(2e-4), np.log10(5e-3))
    h = 10 ** rng.uniform(np.log10(1e-3), np.log10(1e-2))
    fan_on = rng.random() < 0.4
    w = rng.uniform(2e-3, 1e-2) if fan_on else 0.0
    q0 = 10 ** rng.uniform(np.log10(1e-4), np.log10(5e-3))

    # gas built up while the lid was closed: higher near the sludge
    c_bottom = 10 ** rng.uniform(np.log10(5.0), np.log10(100.0))
    decay_len = rng.uniform(0.5, 3.0)
    C0 = c_bottom * np.exp(-Z / decay_len)

    n_bursts = rng.poisson(p["burst_rate"])
    bursts = []
    for _ in range(n_bursts):
        start = rng.uniform(300, T_END - 300)
        dur = rng.uniform(60, 600)
        mag = rng.uniform(*p["burst_mag"])
        bursts.append((start, dur, mag))

    noise_t = np.arange(0, T_END + 61, 60.0)
    noise_v = np.exp(rng.normal(0, 0.15, size=noise_t.size))  # slow +-15% wander in release

    def q_rel(t):
        f = np.interp(t, noise_t, noise_v)
        for s, d, m in bursts:
            if s <= t < s + d:
                f *= m
        return f

    meta = dict(scenario_id=sid, regime=regime, D_m2s=D, h_ms=h, fan_on=fan_on, w_ms=w,
                q0=q0, c0_bottom_ppm=c_bottom, c0_decay_m=decay_len, n_bursts=n_bursts,
                bursts=";".join(f"{s:.0f}/{d:.0f}/{m:.1f}" for s, d, m in bursts))
    return meta, D, w, h, C0, q0, q_rel


def sensor_readings(rng, t, C, drift_range):
    """Lagged, noisy, drifting readings at SENSOR_Z, sampled every SAMPLE_EVERY seconds."""
    readings = {}
    for k, z in enumerate(SENSOR_Z, start=1):
        true = C[:, cell_index(z)]
        lagged = np.empty_like(true)
        lagged[0] = true[0]
        alpha = DT / (SENSOR_TAU + DT)
        for n in range(1, true.size):
            lagged[n] = lagged[n - 1] + alpha * (true[n] - lagged[n - 1])
        drift = rng.uniform(*drift_range) * t / 3600.0
        idx = np.arange(0, t.size, SAMPLE_EVERY)
        noise = rng.normal(0, 1, idx.size) * (0.05 * lagged[idx] + 0.2)
        readings[f"sensor_z{k}"] = np.clip(lagged[idx] + drift[idx] + noise, 0, None)
    return readings


def generate(n_train, n_shift, seed, out):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    metas, frames = [], []
    plan = [("train", n_train), ("shift", n_shift)]
    sid = 0
    for regime, n in plan:
        for _ in range(n):
            meta, D, w, h, C0, q0, q_rel = sample_scenario(rng, regime, sid)
            t, C = simulate(D, w, h, C0, lambda tt: q0 * q_rel(tt))
            idx = np.arange(0, t.size, SAMPLE_EVERY)
            df = pd.DataFrame({"scenario_id": sid, "t_s": t[idx]})
            for k, v in sensor_readings(rng, t, C, REGIMES[regime]["drift"]).items():
                df[k] = v
            df["truth_bz"] = C[idx, cell_index(BREATHING_Z)]
            df["q_rel"] = [q_rel(tt) for tt in t[idx]]
            meta["peak_truth_ppm"] = float(df["truth_bz"].max())
            meta["frac_time_unsafe"] = float((df["truth_bz"] > EXPOSURE_LIMIT_PPM).mean())
            metas.append(meta)
            frames.append(df)
            sid += 1
        print(f"{regime}: {n} scenarios done")
    meta_df = pd.DataFrame(metas)
    ts_df = pd.concat(frames, ignore_index=True)
    meta_df.to_csv(os.path.join(out, "scenarios_meta.csv"), index=False)
    ts_df.to_csv(os.path.join(out, "timeseries.csv"), index=False, float_format="%.4f")
    print(f"saved {len(meta_df)} scenarios, {len(ts_df)} rows to {out}/")
    print(meta_df.groupby("regime")[["peak_truth_ppm", "frac_time_unsafe", "n_bursts"]]
          .describe().T.round(2).to_string())
    plot_examples(meta_df, ts_df, os.path.join(out, "example_plot.png"))


def plot_examples(meta_df, ts_df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = []
    for regime in ("train", "shift"):
        sub = meta_df[meta_df.regime == regime]
        picks += list(sub.sort_values("n_bursts", ascending=False).scenario_id[:2])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, sid in zip(axes.flat, picks):
        d = ts_df[ts_df.scenario_id == sid]
        m = meta_df[meta_df.scenario_id == sid].iloc[0]
        tm = d.t_s / 60
        for k, z in enumerate(SENSOR_Z, start=1):
            ax.plot(tm, d[f"sensor_z{k}"], lw=0.8, alpha=0.7, label=f"sensor {z} m")
        ax.plot(tm, d.truth_bz, "k", lw=2, label=f"truth, breathing zone {BREATHING_Z} m")
        ax.axhline(EXPOSURE_LIMIT_PPM, color="r", ls="--", lw=1, label="limit (placeholder)")
        ax.set_title(f"#{sid} {m.regime}, bursts={m.n_bursts}, fan={'on' if m.fan_on else 'off'}")
        ax.set_ylabel("H2S [ppm]")
    for ax in axes[-1]:
        ax.set_xlabel("time since lid opened [min]")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot saved to {path}")


def selftest():
    # 1) closed top, no fan, constant source: total gas must grow exactly as q*t
    q = 2e-3
    t, C = simulate(D=1e-3, w=0.0, h=0.0, C0=np.zeros(N), q_fn=lambda tt: q, t_end=600)
    mass = C.sum(axis=1) * DZ
    err1 = abs(mass[-1] - q * t[-1]) / (q * t[-1])
    # 2) no source, open top: gas must only decrease and stay non-negative
    t, C = simulate(D=1e-3, w=5e-3, h=5e-3, C0=np.full(N, 50.0), q_fn=lambda tt: 0.0, t_end=600)
    mass = C.sum(axis=1) * DZ
    ok2 = np.all(np.diff(mass) <= 1e-9) and C.min() >= -1e-9
    # 3) steady state vs analytic (w = 0)
    D, h = 1e-3, 5e-3
    t, C = simulate(D=D, w=0.0, h=h, C0=np.zeros(N), q_fn=lambda tt: q, t_end=200000, dt=50.0)
    analytic = q * (L - DZ / 2 - Z) / D + q / h   # top-cell exchange, cell-centred
    err3 = np.max(np.abs(C[-1] - analytic) / analytic)
    print(f"mass conservation error: {err1:.2e}  (should be < 1e-6)")
    print(f"decay monotone and non-negative: {ok2}")
    print(f"steady state vs analytic, max rel error: {err3:.2e}  (should be < 1e-2)")
    assert err1 < 1e-6 and ok2 and err3 < 1e-2
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-shift", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        generate(a.n_train, a.n_shift, a.seed, a.out)
