"""
app.py - Ứng dụng trình diễn cho Đề tài 9 (Độ nhạy và sai số trong mô hình kinh tế)

Chạy:  uv run streamlit run app.py

App dùng CHUNG code với báo cáo (src/model.py, src/sensitivity.py, src/montecarlo.py)
nên số liệu trong app và trong output/ khớp nhau khi cùng tham số.

Các tab: xấp xỉ vi phân, bảng sai số (TN1), bậc của sai số (TN3), độ nhạy (TN4),
Monte Carlo (TN6), vùng tin cậy (TN7), xấp xỉ bậc hai (TN8), kết quả đã xuất.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.model import linear_demand, elastic_demand
from src.sensitivity import (DEFAULT_PCTS, differential, elasticity, error_table,
                             exact_change, taylor_bound, convergence_order,
                             second_order, safe_range)
from src.montecarlo import simulate, DEFAULT_SEED

ROOT = Path(__file__).parent
OUT_TAB = ROOT / "output" / "tables"
OUT_FIG = ROOT / "output" / "figures"

st.set_page_config(page_title="Đề 9 – Độ nhạy và sai số", layout="wide")
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .3,
                     "font.size": 10})

OUT_LABEL = {"q": "Lượng cầu q", "R": "Doanh thu R", "P": "Lợi nhuận P"}


# ===================================================== sidebar
st.sidebar.header("Mô hình")
mkind = st.sidebar.radio("Hàm cầu", ["Tuyến tính  q = a − b·p",
                                     "Co giãn không đổi  q = A·p^(−k)"])
is_linear = mkind.startswith("Tuyến")
c = st.sidebar.slider("Chi phí biên c (nghìn đồng/sp)", 5.0, 45.0, 30.0, 1.0)

if is_linear:
    a = st.sidebar.slider("a (nhu cầu tối đa, sp/tuần)", 600, 2000, 1200, 50)
    b = st.sidebar.slider("b (độ dốc cầu)", 5.0, 20.0, 10.0, 0.5)
    if c >= a / b:
        st.error(f"Cần c < a/b = {a / b:.1f} để giá tối ưu nằm trong miền xác định.")
        st.stop()
    build = lambda p0: linear_demand(a=a, b=b, c=c, p0=p0)
    default_p0 = 50.0
else:
    A5 = st.sidebar.slider("A (×10⁵)", 2.0, 20.0, 10.0, 0.5)
    k = st.sidebar.slider("k (độ co giãn của cầu)", 1.2, 4.0, 2.0, 0.1)
    build = lambda p0: elastic_demand(A=A5 * 1e5, k=k, c=c, p0=p0)
    default_p0 = 40.0

ref = build(1.0)                              # chi de lay mien xac dinh va p*
lo, hi = ref.domain
pmin = round(lo + 0.05 * (hi - lo), 1)
pmax = round(hi - 0.05 * (hi - lo), 1)

st.sidebar.header("Điểm khảo sát")
at_opt = st.sidebar.checkbox("Đặt p₀ = p* (giá tối ưu lợi nhuận)", value=False)
if at_opt:
    p0 = float(ref.p_opt)
    st.sidebar.caption(f"p₀ = p* = {p0:.4g}")
else:
    p0 = st.sidebar.slider("Giá hiện tại p₀", float(pmin), float(pmax),
                           float(min(max(default_p0, pmin), pmax)), 0.5)
pct = st.sidebar.slider("Sai số giá Δp/p₀ (%)", -30.0, 30.0, 5.0, 0.5)
out = st.sidebar.radio("Đại lượng đầu ra", ["q", "R", "P"], index=2,
                       format_func=lambda o: OUT_LABEL[o], horizontal=False)

m = build(p0)
fn = m.funcs[out]
dx = p0 * pct / 100.0
x1 = p0 + dx
if not (lo < x1 < hi):
    st.error(f"Giá mới {x1:.3g} nằm ngoài miền xác định ({lo:g}, {hi:g}). Giảm |Δp|.")
    st.stop()

# ===================================================== tinh toan (src/*)
y0 = float(fn.f(p0))
dy_true = exact_change(fn, p0, dx)
dy_diff = differential(fn, p0, dx)
e_abs = abs(dy_true - dy_diff)
e_rel = e_abs / abs(dy_true) if dy_true != 0 else float("nan")
S = float(elasticity(fn, p0))
bound = taylor_bound(fn, p0, dx)


def fmt_pct(v):
    return "—" if not np.isfinite(v) else f"{v:.2f}%"


def plot_range():
    return (max(lo, 0.4 * min(p0, x1)), min(hi, 1.6 * max(p0, x1)))


# ===================================================== giao dien
st.title("Đề 9 · Độ nhạy và sai số trong mô hình kinh tế")
st.caption(f"{m.title}  ·  c = {c:g}  ·  p* = {m.p_opt:.4g}  ·  "
           "p tính bằng nghìn đồng/sp; q: sp/tuần; R, P: nghìn đồng/tuần")

tabs = st.tabs(["Xấp xỉ vi phân", "Bảng sai số", "Bậc của sai số",
                "Độ nhạy S(p)", "Monte Carlo", "Vùng tin cậy",
                "Xấp xỉ bậc hai", "Kết quả báo cáo"])

# ---------------------------------------------------- tab 1
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Δy thực", f"{dy_true:+.5g}")
    c2.metric("dy = f′(p₀)·Δp", f"{dy_diff:+.5g}")
    c3.metric("Sai số tuyệt đối |Δy − dy|", f"{e_abs:.5g}")
    c4.metric("Sai số tương đối", fmt_pct(100 * e_rel))
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("y₀ = f(p₀)", f"{y0:.5g}")
    d2.metric("Độ nhạy S = f′·p/f", "—" if not np.isfinite(S) else f"{S:+.4f}")
    d3.metric("Δy/y₀ thực", fmt_pct(100 * dy_true / y0) if y0 else "—")
    d4.metric("S·Δp/p₀ (xấp xỉ)", fmt_pct(S * pct) if np.isfinite(S) else "—")
    ok = e_abs <= bound + 1e-9 * max(1.0, abs(bound))
    st.caption(f"Cận Taylor (M/2)·Δp² = {bound:.5g}  →  "
               f"{'|Δy − dy| ≤ cận ✓' if ok else 'vi phạm cận (kiểm tra lại)'}")

    unit = fn.unit_y
    msg = (f"Giá đổi từ {p0:.4g} → {x1:.4g} ({pct:+g}%). **{fn.name}** thực tế thay đổi "
           f"**{dy_true:+.5g}** {unit}")
    if y0:
        msg += f" ({100 * dy_true / y0:+.2f}%)"
    msg += f"; vi phân dự báo **{dy_diff:+.5g}** {unit}. "
    msg += (f"Dự báo lệch {e_abs:.4g} {unit}" +
            (f" ({100 * e_rel:.2f}% so với thay đổi thực)." if np.isfinite(e_rel) else "."))
    st.markdown(msg)
    if np.isfinite(S) and abs(S) < 0.05 and out != "q":
        st.warning("S ≈ 0: p₀ gần điểm tới hạn (f′ ≈ 0). Vi phân dự báo gần như "
                   "không đổi nhưng thay đổi thực là bậc hai (≈ ½f″Δp²), nên sai số "
                   "tương đối rất lớn. Xấp xỉ bậc một không đáng tin ở đây.")

    ps = np.linspace(*plot_range(), 400)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(ps, fn.f(ps), lw=2, label=f"{out}(p)")
    ax.plot(ps, y0 + float(fn.df(p0)) * (ps - p0), "--", label="tiếp tuyến tại p₀")
    ax.scatter([p0], [y0], c="k", zorder=5, label="p₀")
    ax.scatter([x1], [y0 + dy_diff], c="orange", zorder=5, label="vi phân (xấp xỉ)")
    ax.scatter([x1], [y0 + dy_true], c="red", zorder=5, label="giá trị thực")
    ax.plot([x1, x1], [y0 + dy_diff, y0 + dy_true], c="red", ls=":", lw=1.5)
    if plot_range()[0] < m.p_opt < plot_range()[1]:
        ax.axvline(m.p_opt, c="red", alpha=.4, ls=":", label="p*")
    ax.set_xlabel("giá p (nghìn đồng/sp)")
    ax.set_ylabel(f"{fn.name} ({unit})")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------- tab 2
with tabs[1]:
    st.subheader(f"{fn.name} tại p₀ = {p0:.4g}: thay đổi thực vs vi phân")
    df = error_table(fn, p0, DEFAULT_PCTS)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Tải CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"error_table_{m.key}_{out}.csv", mime="text/csv")
    st.caption("Sai số tương đối = |Δy − dy| / |Δy|. Cột cuối là cận Taylor (M/2)Δx².")

# ---------------------------------------------------- tab 3
with tabs[2]:
    st.subheader("Sai số |Δy − dy| theo |Δp| (log-log)")
    grid = np.logspace(-4, np.log10(.3), 40)
    slope, dxs, errs = convergence_order(fn, p0, grid)
    if not np.isfinite(slope):
        st.info("Sai số bằng 0 với mọi Δp (f″ = 0, ví dụ q tuyến tính): "
                "vi phân cho kết quả chính xác tuyệt đối.")
    else:
        st.metric("Hệ số góc log-log (kỳ vọng ≈ 2)", f"{slope:.3f}")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.loglog(dxs, errs, "o-", ms=3, label=f"{out} tại p₀  (slope = {slope:.3f})")
        ref_line = dxs ** 2 * errs[-1] / dxs[-1] ** 2
        ax.loglog(dxs, ref_line, "k:", label="∝ Δp²")
        ax.axvline(abs(dx), c="red", alpha=.5, ls="--", label="Δp hiện tại")
        ax.set_xlabel("|Δp|"); ax.set_ylabel("|Δy − dy|"); ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Sai số cỡ (Δp)²: giảm sai số đầu vào 10 lần thì sai số xấp xỉ "
                   "giảm khoảng 100 lần.")

# ---------------------------------------------------- tab 4
with tabs[3]:
    st.subheader("Độ nhạy tương đối S(p) = f′(p)·p / f(p)")
    st.caption("Giá tăng 1% thì đại lượng đầu ra thay đổi xấp xỉ S%. "
               "Đồ thị cắt ở |S| > 12 để dễ nhìn.")
    r_lo, r_hi = (0.04 * hi, 0.96 * hi) if is_linear else (10.0, 150.0)
    ps2 = np.linspace(r_lo, r_hi, 800)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for o, st_ in (("q", "-"), ("R", "--"), ("P", "-.")):
        Sv = elasticity(m.funcs[o], ps2)
        ax.plot(ps2, np.where(np.abs(Sv) > 12, np.nan, Sv), st_, label=f"S_{o}(p)")
    ax.axvline(p0, c="gray", ls=":", label="p₀")
    ax.axvline(m.p_opt, c="red", ls=":", label="p*")
    ax.axhline(0, c="k", lw=.6)
    ax.set_ylim(-12, 12)
    ax.set_xlabel("giá p"); ax.set_ylabel("S"); ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    rows = [{"Đại lượng": OUT_LABEL[o], "S tại p₀": float(elasticity(m.funcs[o], p0)),
             "S tại p*": float(elasticity(m.funcs[o], m.p_opt))} for o in ("q", "R", "P")]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------- tab 5
with tabs[4]:
    st.subheader("Mô phỏng nhiễu giá: p = p₀(1 + ε), ε ~ N(0, σ²)")
    sig = st.slider("σ (độ lệch chuẩn của sai số giá, %)", 0.5, 15.0, 5.0, 0.5) / 100
    n = st.select_slider("Số mẫu", options=[20_000, 100_000, 200_000], value=200_000)
    st.caption(f"seed = {DEFAULT_SEED} (cố định để khớp với báo cáo). Dự đoán dùng khai "
               "triển Taylor: std ≈ |f′|·p₀·σ, trung bình ≈ ½f″(p₀σ)².")
    colA, colB = st.columns(2)
    for col, tag, x0 in ((colA, "p₀", p0), (colB, "p*", float(m.p_opt))):
        with col:
            r = simulate(fn, x0, sig, n=n)
            st.markdown(f"**Tại {tag} = {x0:.4g}**")
            m1, m2 = st.columns(2)
            m1.metric("std Δy (thực)", f"{r['std']:.5g}",
                      f"dự đoán {r['std_pred']:.5g}", delta_color="off")
            m2.metric("mean Δy (thực)", f"{r['mean']:.5g}",
                      f"dự đoán {r['mean_pred']:.5g}", delta_color="off")
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.hist(r["dy"], bins=80, alpha=.8)
            ax.axvline(0, c="k", lw=.8)
            ax.axvline(r["mean"], c="red", ls="--", label="trung bình")
            ax.set_xlabel(f"Δ{out} ({fn.unit_y})"); ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

# ---------------------------------------------------- tab 6 (TN7)
with tabs[5]:
    st.subheader("Vùng tin cậy của vi phân")
    st.caption("Mức tăng/giảm giá lớn nhất (theo % của p₀) sao cho sai số tương đối "
               "E_rel = |Δy − dy| / |Δy| không vượt ngưỡng với mọi mức nhỏ hơn.")
    tol = st.slider("Ngưỡng sai số tương đối E_rel (%)", 1.0, 20.0, 5.0, 0.5) / 100
    up, dn = safe_range(fn, p0, tol, m.domain)
    u1, u2 = st.columns(2)
    u1.metric("Giá tăng tối đa", f"+{100 * up:.2f}%")
    u2.metric("Giá giảm tối đa", f"−{100 * dn:.2f}%")
    if up >= 0.4995 and dn >= 0.4995:
        st.info("Không chạm ngưỡng trong khoảng ±50% (quét tối đa 50%): vi phân chính "
                "xác tuyệt đối ở đây, ví dụ q tuyến tính với f″ = 0.")
    elif up == 0 and dn == 0:
        st.warning("Vùng tin cậy bằng 0: tại điểm tới hạn (f′ = 0) vi phân dự báo "
                   "dy = 0 nên E_rel = 100% với mọi Δp ≠ 0. Xem tab xấp xỉ bậc hai.")
    else:
        st.markdown(f"Với {fn.name.lower()} tại p₀ = {p0:.4g}, vi phân sai không quá "
                    f"{100 * tol:g}% khi giá tăng tới **{100 * up:.2f}%** hoặc giảm "
                    f"tới **{100 * dn:.2f}%**. Ngoài vùng này nên dùng giá trị thực "
                    "hoặc xấp xỉ bậc hai.")

    deltas = np.linspace(-.30, .30, 1201)
    deltas = deltas[deltas != 0]
    xs = p0 * (1 + deltas)
    in_dom = (xs > lo) & (xs < hi)
    dt = np.where(in_dom, fn.f(np.where(in_dom, xs, p0)) - fn.f(p0), np.nan)
    err = np.abs(dt - float(fn.df(p0)) * (xs - p0))
    with np.errstate(divide="ignore", invalid="ignore"):
        erel_curve = np.where(dt != 0, 100 * err / np.abs(dt), np.nan)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(100 * deltas, np.clip(erel_curve, 0, 100), label=f"E_rel của {out}")
    ax.axhline(100 * tol, c="k", ls=":", label=f"ngưỡng {100 * tol:g}%")
    if up > 0 or dn > 0:
        ax.axvspan(-100 * dn, 100 * up, color="green", alpha=.12, label="vùng tin cậy")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Δp/p₀ (%)"); ax.set_ylabel("sai số tương đối E_rel (%)")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------- tab 7 (TN8)
with tabs[6]:
    st.subheader("Xấp xỉ bậc hai: dy₂ = f′(p₀)Δp + ½f″(p₀)Δp²")
    dy2 = second_order(fn, p0, dx)
    e2 = abs(dy_true - dy2)
    e2_rel = e2 / abs(dy_true) if dy_true != 0 else float("nan")
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Δy thực", f"{dy_true:+.5g}")
    v2.metric("dy bậc 1", f"{dy_diff:+.5g}")
    v3.metric("dy₂ bậc 2", f"{dy2:+.5g}")
    v4.metric("E_rel bậc 1", fmt_pct(100 * e_rel))
    v5.metric("E_rel bậc 2", fmt_pct(100 * e2_rel))
    if e2 <= 1e-9 * max(1.0, abs(dy_true)):
        st.info("Xấp xỉ bậc hai chính xác tuyệt đối: hàm này là đa thức bậc ≤ 2 "
                "(ví dụ R và P của cầu tuyến tính).")
    elif np.isfinite(e_rel) and np.isfinite(e2_rel):
        st.markdown(f"Thêm số hạng bậc hai đưa sai số tương đối từ "
                    f"**{100 * e_rel:.2f}%** xuống **{100 * e2_rel:.2f}%**.")
    elif dy_diff == 0:
        st.markdown("Vi phân bậc 1 cho dy = 0 (điểm tới hạn), còn số hạng bậc hai "
                    "bắt được thay đổi thực.")

    ps3 = np.linspace(*plot_range(), 400)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(ps3, fn.f(ps3), lw=2, label=f"{out}(p)")
    ax.plot(ps3, y0 + float(fn.df(p0)) * (ps3 - p0), "--", label="bậc 1 (tiếp tuyến)")
    ax.plot(ps3, y0 + float(fn.df(p0)) * (ps3 - p0)
            + 0.5 * float(fn.d2f(p0)) * (ps3 - p0) ** 2, "-.", label="bậc 2 (parabol)")
    ax.scatter([p0], [y0], c="k", zorder=5, label="p₀")
    ax.scatter([x1], [y0 + dy_true], c="red", zorder=5, label="giá trị thực")
    ax.set_xlabel("giá p (nghìn đồng/sp)")
    ax.set_ylabel(f"{fn.name} ({fn.unit_y})")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Bậc hội tụ** (hệ số góc log-log của sai số theo |Δp|)")
    grid2 = np.logspace(-4, np.log10(.3), 40)
    s1, dxs1, er1 = convergence_order(fn, p0, grid2)
    s2, dxs2, er2 = convergence_order(fn, p0, grid2, approx=second_order)
    h1, h2 = st.columns(2)
    h1.metric("Bậc 1 (kỳ vọng ≈ 2)", "—" if not np.isfinite(s1) else f"{s1:.3f}")
    h2.metric("Bậc 2 (kỳ vọng ≈ 3)", "—" if not np.isfinite(s2) else f"{s2:.3f}")
    if not np.isfinite(s1) and not np.isfinite(s2):
        st.info("Cả hai xấp xỉ đều chính xác tuyệt đối (f″ = 0).")
    else:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if np.isfinite(s1):
            ax.loglog(dxs1, er1, "o-", ms=3, label=f"bậc 1 (slope = {s1:.2f})")
        if np.isfinite(s2):
            ax.loglog(dxs2, er2, "s--", ms=3, label=f"bậc 2 (slope = {s2:.2f})")
        else:
            st.caption("Xấp xỉ bậc hai có sai số bằng 0, nên không có đường bậc 2 "
                       "trên đồ thị.")
        ax.axvline(abs(dx), c="red", alpha=.5, ls="--", label="Δp hiện tại")
        ax.set_xlabel("|Δp|"); ax.set_ylabel("|Δy − xấp xỉ|"); ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------- tab 8
with tabs[7]:
    st.subheader("Kết quả đã xuất bởi experiments.py")
    st.caption("Các tệp này dùng tham số mặc định của báo cáo "
               "(a=1200, b=10, A=10⁶, k=2, c=30), không đổi theo thanh bên.")
    figs = sorted(OUT_FIG.glob("*.png")) if OUT_FIG.exists() else []
    tabs_csv = sorted(OUT_TAB.glob("*.csv")) if OUT_TAB.exists() else []
    if not figs and not tabs_csv:
        st.info("Chưa có kết quả. Chạy `uv run experiments.py` trước.")
    else:
        if figs:
            pick = st.selectbox("Hình", figs, format_func=lambda p: p.name)
            st.image(str(pick), use_container_width=True)
        if tabs_csv:
            pickt = st.selectbox("Bảng", tabs_csv, format_func=lambda p: p.name)
            st.dataframe(pd.read_csv(pickt, encoding="utf-8-sig"),
                         use_container_width=True, hide_index=True)
