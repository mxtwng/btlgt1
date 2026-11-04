# Bài tập lớn Giải tích 1 — Đề tài 9: Độ nhạy và sai số trong mô hình kinh tế

Khi giá bán thực tế lệch so với dự kiến một chút, doanh thu và lợi nhuận thay đổi bao nhiêu, và có thể dự báo nhanh bằng **vi phân** `Δy ≈ f′(x)Δx` đến mức chính xác nào?

Dự án trả lời câu hỏi này trên hai mô hình hàm cầu. Mọi đạo hàm đều **tự tính tay và tự cài đặt** (SymPy và sai phân hữu hạn chỉ dùng để kiểm chứng trong `verify.py`). Chương trình so sánh thay đổi thực với xấp xỉ vi phân bậc một và bậc hai, khảo sát độ nhạy, tìm vùng mà vi phân còn đáng tin và chạy mô phỏng Monte Carlo. Kết quả dùng trong báo cáo và có thể khám phá thêm bằng ứng dụng Streamlit.

## Mô hình

| | Hàm cầu | Điểm khảo sát | Giá tối ưu |
|---|---|---|---|
| (L) Tuyến tính | `q = 1200 − 10p` | `p₀ = 50` | `p* = (a + bc)/(2b) = 75` |
| (E) Co giãn không đổi | `q = 10⁶·p⁻²` | `p₀ = 40` | `p* = ck/(k − 1) = 60` |

Chi phí biên mặc định `c = 30`. Đơn vị: `p` nghìn đồng/sản phẩm; `q` sản phẩm/tuần; `R`, `P` nghìn đồng/tuần. Với `R = p·q` và `P = (p − c)·q`.

## Tám thí nghiệm

`experiments.py` chạy cả tám thí nghiệm và ghi kết quả vào `output/` (chạy lại sẽ ghi đè).

| TN | Câu hỏi | Công cụ Giải tích | Kết quả |
|---|---|---|---|
| 1 | Vi phân dự báo thay đổi thực tốt đến đâu? | vi phân, sai số tuyệt đối/tương đối, cận Taylor | `TN1_*.csv` |
| 2 | Xấp xỉ tuyến tính trông thế nào trên đồ thị? | tiếp tuyến | `TN2_tangent_*.png` |
| 3 | Sai số giảm theo Δp với bậc bao nhiêu? | khai triển Taylor, hồi quy log-log (kỳ vọng ≈ 2) | `TN3_convergence_order.csv`, `TN3_error_loglog.png` |
| 4 | Đại lượng nào nhạy nhất với giá? | độ nhạy tương đối `S = f′·p/f` | `TN4_elasticity_values.csv`, `TN4_elasticity.png` |
| 5 | Kết luận có đổi khi đổi tham số không? | quét `b` (L) và `k` (E) ở Δp = +10% | `TN5_param_study.csv`, `TN5_param_study.png` |
| 6 | Khi giá nhiễu ngẫu nhiên thì ΔP phân bố ra sao? | Monte Carlo so với dự đoán `std ≈ \|f′\|·p₀σ`, `mean ≈ ½f″(p₀σ)²` | `TN6_montecarlo.csv`, `TN6_montecarlo_hist.png` |
| 7 | Vi phân dùng được đến mức Δp nào? | vùng tin cậy: Δp lớn nhất để `E_rel` ≤ 1%, 5%, 10% | `TN7_safe_range.csv`, `TN7_erel_vs_delta.png` |
| 8 | Thêm số hạng bậc hai cải thiện bao nhiêu? | `dy₂ = f′Δp + ½f″Δp²`, bậc hội tụ ≈ 2 so với ≈ 3 | `TN8_second_order.csv`, `TN8_orders.csv`, `TN8_second_order_loglog.png` |

Cách đọc một số kết quả:

- **TN7:** giá trị `50%` nghĩa là không chạm ngưỡng (ví dụ `q` tuyến tính, vi phân chính xác tuyệt đối). Giá trị `0%` tại `p*` vì `P′(p*) = 0` nên `dy = 0` và `E_rel = 100%` với mọi Δp ≠ 0.
- **TN8:** với mô hình tuyến tính, `R` và `P` là đa thức bậc hai nên xấp xỉ bậc hai chính xác tuyệt đối. Bảng bậc hội tụ để trống ở các dòng đó.

## Cấu trúc

```
├── src/
│   ├── model.py         hai mô hình cầu; q, R, P và đạo hàm bậc 1, bậc 2 (tự tính tay)
│   ├── sensitivity.py   Δy thực, vi phân, sai số, độ nhạy, cận Taylor, bậc hội tụ,
│   │                    xấp xỉ bậc hai, vùng tin cậy
│   └── montecarlo.py    mô phỏng nhiễu giá (seed cố định)
├── verify.py            kiểm tra đạo hàm tự cài bằng SymPy và sai phân hữu hạn
├── experiments.py       chạy TN1–TN8, xuất bảng CSV và hình PNG
├── app.py               ứng dụng Streamlit tương tác, dùng chung code trong src/
├── output/
│   ├── tables/          bảng kết quả (CSV)
│   └── figures/         hình kết quả (PNG)
├── report/              phần báo cáo LaTeX (muc6.tex)
├── docs/                đề bài
├── pyproject.toml, uv.lock
└── README.md
```

## Chạy

Cần Python 3.12 trở lên.

Với [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run verify.py
uv run experiments.py
uv run streamlit run app.py
```

Với pip:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
python -m pip install numpy pandas matplotlib sympy streamlit
python verify.py
python experiments.py
streamlit run app.py
```

`verify.py` kiểm chứng các đạo hàm; `experiments.py` tạo lại bảng và hình; `app.py` mở giao diện web để đổi tham số và xem kết quả.

Ứng dụng có 8 tab: xấp xỉ vi phân, bảng sai số (TN1), bậc của sai số (TN3), độ nhạy (TN4), Monte Carlo (TN6), vùng tin cậy (TN7), xấp xỉ bậc hai (TN8) và kết quả đã xuất. Tham số mô hình, giá `p₀` và mức sai số giá chỉnh ở thanh bên.

## Tái lập kết quả

Tham số mặc định và seed mô phỏng (`20260503`, 200 000 mẫu) nằm trong `src/`. Khi cập nhật số liệu trong báo cáo, hãy chạy lại `experiments.py` và dùng cùng tham số và seed với phần phân tích. `app.py` dùng chung code với báo cáo nên số liệu khớp nhau khi cùng tham số.
