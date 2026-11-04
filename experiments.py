"""
experiments.py - Tính toán / mô phỏng / thực nghiệm (Mục 8 của báo cáo, Đề tài 9)

Chạy:  uv run experiments.py     (hoặc: python experiments.py)

Kết quả: output/tables/*.csv  và  output/figures/*.png
(chạy lại sẽ ghi đè các tệp cùng tên).

Tám thí nghiệm, mỗi thí nghiệm trả lời một câu hỏi của đề tài:

    TN1  Vi phân dy dự báo thay đổi thực Δy tốt đến đâu?        (bảng sai số)
    TN2  Xấp xỉ tuyến tính trông như thế nào trên đồ thị?       (tiếp tuyến)
    TN3  Sai số |Δy - dy| giảm theo Δp với bậc bao nhiêu?       (log-log)
    TN4  Đại lượng nào nhạy nhất với giá?                       (độ nhạy S)
    TN5  Kết luận có đổi khi đổi tham số mô hình không?         (quét b, k)
    TN6  Khi giá nhiễu ngẫu nhiên thì ΔP phân bố ra sao?        (Monte Carlo)
    TN7  Vi phân dùng được đến mức Δp nào?                      (vùng tin cậy)
    TN8  Thêm số hạng bậc hai thì cải thiện bao nhiêu?          (xấp xỉ bậc 2)

Ký hiệu: p = giá (nghìn đồng/sp), q = lượng cầu, R = doanh thu, P = lợi nhuận,
p0 = giá hiện tại, p* = giá tối ưu (P'(p*) = 0).
Tất cả phép tính đều lấy từ src/ nên khớp với app.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model import build_models, linear_demand, elastic_demand
from src.sensitivity import (error_table, elasticity, convergence_order,
                             exact_change, differential, DEFAULT_PCTS,
                             second_order, safe_range)
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
    """Lưu hình vào output/figures/<name> rồi đóng hình."""
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def to_csv(df, name):
    """Ghi bảng vào output/tables/<name> (UTF-8 có BOM để Excel đọc đúng tiếng Việt)."""
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig", float_format="%.6g")


# ------------------------------------------------ TN1: bang sai so
def exp1_tables(models):
    """TN1 - Bảng so sánh thay đổi thực Δy và vi phân dy.

    Với mỗi mô hình và mỗi đại lượng (q, R, P), thay giá p0 một lượng
    Δp/p0 ∈ DEFAULT_PCTS (±1% ... ±20%) rồi tính Δy thực, dy = f'(p0)Δp,
    sai số tuyệt đối/tương đối và cận Taylor (M/2)Δp². Với P còn khảo sát
    thêm tại p* (nơi P' = 0 nên dy = 0).

    Ghi: output/tables/TN1_<mô hình>_<q|R|P>_<p0|popt>.csv
    """
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
    """TN2 - Minh họa hình học: vi phân chính là tiếp tuyến.

    Vẽ q(p), R(p), P(p) cùng tiếp tuyến tại p0 và dải p0 ± 20%. Khoảng cách
    giữa đường cong và tiếp tuyến chính là sai số của xấp xỉ vi phân.

    Ghi: output/figures/TN2_tangent_<linear|elastic>.png
    """
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
    """TN3 - Bậc của sai số xấp xỉ vi phân (kỳ vọng ≈ 2).

    Vẽ |Δy - dy| theo |Δp| trên thang log-log với Δp/p0 từ 1e-4 đến 30%.
    Hệ số góc hồi quy log-log là bậc của sai số: ≈ 2 nghĩa là giảm Δp 10 lần
    thì sai số giảm ~100 lần (do sai số ≈ ½f''Δp²). Bỏ qua trường hợp sai số
    bằng 0 (q tuyến tính, f'' = 0).

    Ghi: output/tables/TN3_convergence_order.csv, output/figures/TN3_error_loglog.png
    """
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
    """TN4 - Độ nhạy tương đối S(p) = f'(p)·p / f(p).

    Giá tăng 1% thì đại lượng đầu ra đổi xấp xỉ S%. Vẽ S_q, S_R, S_P theo p
    (cắt ở |S| > 12 cho dễ nhìn) và lập bảng giá trị tại p0 và p*. Tại p* thì
    S_P = 0 vì P' = 0.

    Ghi: output/figures/TN4_elasticity.png, output/tables/TN4_elasticity_values.csv
    """
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
    """TN5 - Ảnh hưởng của tham số mô hình (cố định Δp = +10%).

    Quét độ dốc cầu b (mô hình tuyến tính, b = 5..20) và độ co giãn k (mô hình
    co giãn, k = 1.2..4), mỗi giá trị tính sai số tương đối của xấp xỉ vi phân
    cho q, R, P. Trả lời: kết luận về độ chính xác của vi phân có phụ thuộc
    vào bộ tham số cụ thể hay không.

    Ghi: output/tables/TN5_param_study.csv, output/figures/TN5_param_study.png
    """
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
    """TN6 - Mô phỏng Monte Carlo với giá nhiễu p = p0(1 + ε), ε ~ N(0, σ²).

    Với σ ∈ {1%, 5%, 10%}, mô phỏng n mẫu (seed cố định để tái lập) và so sánh
    độ lệch chuẩn và trung bình của ΔP thực với dự đoán từ khai triển Taylor:
    std ≈ |P'|·p0·σ, mean ≈ ½P''(p0σ)². Tại p* thì P' = 0 nên std dự đoán bậc
    một bằng 0, còn phân bố ΔP lệch về phía âm (P'' < 0).

    Ghi: output/tables/TN6_montecarlo.csv, output/figures/TN6_montecarlo_hist.png
    """
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


# ------------------------------------------------ TN7: vung tin cay
def exp7_safe_range(models):
    """TN7 - Vùng tin cậy của vi phân: Δp tối đa để sai số tương đối ≤ ngưỡng.

    Với ngưỡng E_rel ∈ {1%, 5%, 10%}, tìm mức tăng/giảm giá lớn nhất (tính theo
    % của p0) sao cho E_rel ≤ ngưỡng với MỌI mức nhỏ hơn (src.sensitivity.
    safe_range, quét lưới bước 0.05%, tối đa 50%). Cách đọc kết quả:
      - 50% nghĩa là không chạm ngưỡng (q tuyến tính: vi phân chính xác tuyệt đối);
      - 0% tại p* vì dy = 0 nên E_rel = 100% với mọi Δp ≠ 0.
    Hình vẽ E_rel theo Δp/p0 ∈ [-30%, 30%] tại p0 cho q, R, P (đường ngưỡng 5%).

    Ghi: output/tables/TN7_safe_range.csv, output/figures/TN7_erel_vs_delta.png
    """
    print("\n=== TN7: vùng tin cậy của vi phân (E_rel <= ngưỡng) ===")
    rows = []
    for m in models.values():
        for out in OUTS:
            for tag, x0 in (("p0", m.p0), ("p*", m.p_opt)):
                if out != "P" and tag == "p*":
                    continue
                for tol in (.01, .05, .10):
                    up, dn = safe_range(m.funcs[out], x0, tol, m.domain)
                    rows.append({"mô hình": m.key, "đại lượng": out, "điểm": tag,
                                 "ngưỡng E_rel (%)": 100 * tol,
                                 "Δp tăng tối đa (%)": 100 * up,
                                 "Δp giảm tối đa (%)": 100 * dn})
    df = pd.DataFrame(rows)
    to_csv(df, "TN7_safe_range.csv")
    print(df.to_string(index=False))

    deltas = np.linspace(-.30, .30, 1201)
    deltas = deltas[deltas != 0]
    fig, axes = plt.subplots(1, len(models), figsize=(11, 4))
    for ax, m in zip(np.atleast_1d(axes), models.values()):
        for out in OUTS:
            fn, x0 = m.funcs[out], m.p0
            erel = []
            for d in deltas:
                x1 = x0 * (1 + d)
                if not (m.domain[0] < x1 < m.domain[1]):
                    erel.append(np.nan)
                    continue
                dx = x1 - x0
                dt = exact_change(fn, x0, dx)
                e = abs(dt - differential(fn, x0, dx))
                erel.append(100 * e / abs(dt) if dt != 0 else np.nan)
            ax.plot(100 * deltas, np.clip(erel, 0, 100), label=out)
        ax.axhline(5, c="k", ls=":", label="ngưỡng 5%")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Δp/p₀ (%)")
        ax.set_ylabel("sai số tương đối E_rel (%)")
        ax.set_title(m.title, fontsize=9)
        ax.legend(fontsize=8)
    save(fig, "TN7_erel_vs_delta.png")


# ------------------------------------------------ TN8: xap xi bac hai
def exp8_second_order(models):
    """TN8 - Xấp xỉ bậc hai dy2 = f'(p0)Δp + ½f''(p0)Δp² so với vi phân bậc một.

    Phần 1 (bảng): với lợi nhuận P tại p0 và p*, Δp/p0 ∈ {±5%, ±10%, ±20%},
    so sánh Δy thực với dy bậc 1 và bậc 2, kèm sai số tuyệt đối/tương đối.
    Tại p* xấp xỉ bậc 1 cho dy = 0 (sai 100%) còn bậc 2 bắt được thay đổi.
    Phần 2 (log-log): bậc hội tụ của hai xấp xỉ, kỳ vọng ≈ 2 (bậc 1) và ≈ 3
    (bậc 2). Mô hình tuyến tính có R, P là đa thức bậc hai nên xấp xỉ bậc 2
    chính xác tuyệt đối (sai số 0, hệ số góc để trống/NaN).

    Ghi: output/tables/TN8_second_order.csv, output/tables/TN8_orders.csv,
         output/figures/TN8_second_order_loglog.png
    """
    print("\n=== TN8: xấp xỉ bậc hai dy2 = f'Δp + ½f''Δp² ===")
    pcts = [-20, -10, -5, 5, 10, 20]
    rows = []
    for m in models.values():
        fn = m.funcs["P"]
        for tag, x0 in (("p0", m.p0), ("p*", m.p_opt)):
            for pct in pcts:
                dx = x0 * pct / 100.0
                dt = exact_change(fn, x0, dx)
                d1 = differential(fn, x0, dx)
                d2 = second_order(fn, x0, dx)
                e1, e2 = abs(dt - d1), abs(dt - d2)
                rows.append({"mô hình": m.key, "điểm": tag, "Δp/p0 (%)": pct,
                             "Δy thực": dt, "dy bậc 1": d1, "dy bậc 2": d2,
                             "E_abs bậc 1": e1, "E_abs bậc 2": e2,
                             "E_rel bậc 1 (%)": 100 * e1 / abs(dt),
                             "E_rel bậc 2 (%)": 100 * e2 / abs(dt)})
    df = pd.DataFrame(rows)
    to_csv(df, "TN8_second_order.csv")
    print(df.round(4).to_string(index=False))

    grid = np.logspace(-4, np.log10(.3), 40)
    orows = []
    fig, axes = plt.subplots(1, len(models), figsize=(11, 4))
    for ax, m in zip(np.atleast_1d(axes), models.values()):
        for out, x0, tag in (("R", m.p0, "p0"), ("P", m.p0, "p0"),
                             ("P", m.p_opt, "p*")):
            s1, dxs, e1 = convergence_order(m.funcs[out], x0, grid)
            s2, _, e2 = convergence_order(m.funcs[out], x0, grid,
                                          approx=second_order)
            orows.append({"mô hình": m.key, "đại lượng": out, "điểm": tag,
                          "hệ số góc bậc 1": s1, "hệ số góc bậc 2": s2})
            ax.loglog(dxs, e1, "o-", ms=3, label=f"{out} tại {tag}: bậc 1 (slope={s1:.2f})")
            if np.isfinite(s2):
                ax.loglog(dxs, e2, "s--", ms=3,
                          label=f"{out} tại {tag}: bậc 2 (slope={s2:.2f})")
        ax.set_xlabel("|Δp|")
        ax.set_ylabel("|Δy − xấp xỉ|")
        ax.set_title(m.title, fontsize=9)
        ax.legend(fontsize=7)
    odf = pd.DataFrame(orows)
    to_csv(odf, "TN8_orders.csv")
    print("\nHệ số góc log-log (NaN = sai số bằng 0, xấp xỉ chính xác):")
    print(odf.to_string(index=False))
    save(fig, "TN8_second_order_loglog.png")


if __name__ == "__main__":
    models = build_models()
    exp1_tables(models)
    exp2_tangent(models)
    exp3_order(models)
    exp4_elasticity(models)
    exp5_params()
    exp6_montecarlo(models)
    exp7_safe_range(models)
    exp8_second_order(models)
    print(f"\nXong. Bảng: {TAB}\n      Hình: {FIG}")
