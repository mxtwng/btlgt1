"""
experiments.py - Tinh toan / mo phong / thuc nghiem (Muc 8 cua bao cao)
Chay:  python experiments.py
Ket qua: output/tables/*.csv  va  output/figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model import build_models, linear_demand, elastic_demand
from src.sensitivity import (error_table, elasticity, convergence_order,
                             exact_change, differential, DEFAULT_PCTS)
from src.montecarlo import simulate, DEFAULT_SEED, DEFAULT_N

ROOT = Path(__file__).parent
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = DEFAULT_SEED
PCTS = DEFAULT_PCTS
PLOT_RANGE = {"linear": (5, 115), "elastic": (10, 150)}
OUTS = ["q", "R", "P"]
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3,
                     "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def to_csv(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig", float_format="%.6g")


# ------------------------------------------------ TN1: bang sai so
def exp1_tables(models):
    print("\n=== TN1: Δy thực vs dy ===")
    for m in models.values():
        for out in OUTS:
            for tag, x0 in (("p0", m.p0), ("popt", m.p_opt)):
                if out != "P" and tag == "popt":
                    continue
                df = error_table(m.funcs[out], x0, PCTS)
                to_csv(df, f"TN1_{m.key}_{out}_{tag}.csv")
        df = error_table(m.funcs["P"], m.p0, PCTS)
        print(f"\n{m.title}  |  P tại p0={m.p0:g}")
        print(df[["Δx/x0 (%)", "Δy thực", "dy (vi phân)",
                  "Sai số tuyệt đối", "Sai số tương đối"]].to_string(index=False))
        d = error_table(m.funcs["P"], m.p_opt, [5, 10])
        print(f"P tại p*={m.p_opt:g} (P'=0): dy = {d['dy (vi phân)'].iloc[0]:.3g}, "
              f"Δy thực (+5%) = {d['Δy thực'].iloc[0]:.6g}, "
              f"E_rel = {d['Sai số tương đối'].iloc[0]:.3g}")


# ------------------------------------------------ TN2: tiep tuyen
def exp2_tangent(models):
    for m in models.values():
        lo, hi = PLOT_RANGE[m.key]
        ps = np.linspace(lo, hi, 400)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, out in zip(axes, OUTS):
            fn = m.funcs[out]
            y = fn.f(ps)
            tang = fn.f(m.p0) + fn.df(m.p0) * (ps - m.p0)
            ax.plot(ps, y, lw=2, label=f"{fn.symbol}(p)")
            ax.plot(ps, tang, "--", label="tiếp tuyến tại $p_0$")
            ax.axvspan(.8 * m.p0, 1.2 * m.p0, color="orange", alpha=.15,
                       label="$p_0\\pm20\\%$")
            ax.scatter([m.p0], [fn.f(m.p0)], c="k", zorder=5)
            pad = .1 * (y.max() - y.min())
            ax.set_ylim(y.min() - pad, y.max() + pad)
            ax.set_xlabel("giá p (nghìn đồng/sp)")
            ax.set_ylabel(f"{fn.name} ({fn.unit_y})")
            ax.legend(fontsize=8)
        fig.suptitle(m.title)
        save(fig, f"TN2_tangent_{m.key}.png")


# ------------------------------------------------ TN3: bac hoi tu
def exp3_order(models):
    print("\n=== TN3: bậc của sai số (kỳ vọng ≈ 2) ===")
    grid = np.logspace(-4, np.log10(.3), 40)
    rows = []
    fig, axes = plt.subplots(1, len(models), figsize=(11, 4))
    for ax, m in zip(np.atleast_1d(axes), models.values()):
        for out, x0, tag in (("q", m.p0, "p0"), ("R", m.p0, "p0"),
                             ("P", m.p0, "p0"), ("P", m.p_opt, "p*")):
            slope, dxs, errs = convergence_order(m.funcs[out], x0, grid)
            rows.append({"mô hình": m.key, "đại lượng": out, "điểm": tag,
                         "hệ số góc log-log": slope})
            if np.isnan(slope):
                continue
            ax.loglog(dxs, errs, "o-", ms=3,
                      label=f"{out} tại {tag}  (slope={slope:.3f})")
        ref = grid * m.p0
        ax.loglog(ref, ref ** 2 * 0.1 * np.max(errs) / np.max(ref ** 2),
                  "k:", label="tham chiếu $\\propto\\Delta x^2$")
        ax.set_xlabel("|Δx|"); ax.set_ylabel("|Δy − dy|")
        ax.set_title(m.title, fontsize=9); ax.legend(fontsize=8)
    to_csv(pd.DataFrame(rows), "TN3_convergence_order.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    save(fig, "TN3_error_loglog.png")


# ------------------------------------------------ TN4: do nhay
def exp4_elasticity(models):
    fig, axes = plt.subplots(1, len(models), figsize=(11, 4))
    for ax, m in zip(np.atleast_1d(axes), models.values()):
        lo, hi = PLOT_RANGE[m.key]
        ps = np.linspace(lo, hi, 800)
        for out, st in (("q", "-"), ("R", "--"), ("P", "-.")):
            S = elasticity(m.funcs[out], ps)
            S = np.where(np.abs(S) > 12, np.nan, S)    # cat de de nhin
            ax.plot(ps, S, st, label=f"$S_{out}(p)$")
        ax.axvline(m.p0, c="gray", ls=":", label="$p_0$")
        ax.axvline(m.p_opt, c="red", ls=":", label="$p^*$")
        ax.axhline(0, c="k", lw=.6)
        ax.set_ylim(-12, 12)
        ax.set_xlabel("giá p"); ax.set_ylabel("độ nhạy tương đối S")
        ax.set_title(m.title, fontsize=9); ax.legend(fontsize=8)
    save(fig, "TN4_elasticity.png")
    rows = []
    for m in models.values():
        for out in OUTS:
            for tag, x0 in (("p0", m.p0), ("p*", m.p_opt)):
                rows.append({"mô hình": m.key, "đại lượng": out, "điểm": tag,
                             "p": x0, "S": float(elasticity(m.funcs[out], x0))})
    to_csv(pd.DataFrame(rows), "TN4_elasticity_values.csv")
    print("\n=== TN4: độ nhạy tại p0 và p* ===")
    print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------ TN5: tham so
def exp5_params():
    print("\n=== TN5: ảnh hưởng tham số (Δp = +10%) ===")
    sweeps = [("linear", "b", np.linspace(5, 20, 7),
               lambda v: linear_demand(b=v)),
              ("elastic", "k", np.array([1.2, 1.5, 2, 2.5, 3, 4]),
               lambda v: elastic_demand(k=v))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for ax, (key, pname, vals, build) in zip(axes, sweeps):
        res = {o: [] for o in OUTS}
        for v in vals:
            m = build(v)
            for o in OUTS:
                dx = .1 * m.p0
                dt, dd = exact_change(m.funcs[o], m.p0, dx), differential(m.funcs[o], m.p0, dx)
                e = abs(dt - dd) / abs(dt) * 100 if dt != 0 else np.nan
                res[o].append(e)
                rows.append({"mô hình": key, "tham số": pname, "giá trị": v,
                             "đại lượng": o, "Sai số tương đối (%)": e})
        for o in OUTS:
            ax.plot(vals, res[o], "o-", label=o)
        ax.set_xlabel(pname); ax.set_ylabel("sai số tương đối của xấp xỉ (%)")
        ax.set_title(f"{key}: Δp = +10%"); ax.legend()
    df = pd.DataFrame(rows)
    to_csv(df, "TN5_param_study.csv")
    print(df.pivot_table(index=["mô hình", "giá trị"], columns="đại lượng",
                         values="Sai số tương đối (%)").round(3).to_string())
    save(fig, "TN5_param_study.png")


# ------------------------------------------------ TN6: Monte Carlo
def exp6_montecarlo(models, n=DEFAULT_N):
    print("\n=== TN6: Monte Carlo (nhiễu giá ε~N(0,σ²)) ===")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in models.values():
        fn = m.funcs["P"]
        for tag, x0 in (("p0", m.p0), ("p*", m.p_opt)):
            for sig in (.01, .05, .10):
                r = simulate(fn, x0, sig, n=n, seed=SEED)
                dy = r["dy"]
                rows.append({
                    "mô hình": m.key, "điểm": tag, "σ": sig,
                    "std ΔP thực": r["std"],
                    "std dự đoán |P'|·x0·σ": r["std_pred"],
                    "mean ΔP thực": r["mean"],
                    "mean dự đoán ½P''(x0σ)²": r["mean_pred"],
                })
                if m.key == "linear" and sig == .10:
                    ax = axes[0 if tag == "p0" else 1]
                    ax.hist(dy, bins=80, alpha=.8)
                    ax.axvline(0, c="k", lw=.8)
                    ax.set_title(f"ΔP tại {tag} = {x0:g}, σ=10%")
                    ax.set_xlabel("ΔP (nghìn đồng/tuần)")
    df = pd.DataFrame(rows)
    to_csv(df, "TN6_montecarlo.csv")
    print(df.round(3).to_string(index=False))
    save(fig, "TN6_montecarlo_hist.png")


if __name__ == "__main__":
    models = build_models()
    exp1_tables(models)
    exp2_tangent(models)
    exp3_order(models)
    exp4_elasticity(models)
    exp5_params()
    exp6_montecarlo(models)
    print(f"\nXong. Bảng: {TAB}\n      Hình: {FIG}")
