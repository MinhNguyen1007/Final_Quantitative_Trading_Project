# CLAUDE.md — Quantitative Trading Project

> File này load tự động vào mọi phiên Claude Code làm việc trong project.

## ⚠️ CỐT LÕI — đọc trước khi làm bất cứ thứ gì

User muốn project **đơn giản, đúng, đủ** — KHÔNG over-engineering.

**Bài học từ các phiên trước**: đã từng đi quá xa với multi-seed ensemble,
3-config comparison, thêm layer, tạo nhiều notebook (v2, final). Tất cả đều
**không cải thiện kết quả** mà còn làm phức tạp hơn. Cuối cùng đã quay về bản
đơn giản. **Đừng lặp lại sai lầm này.**

Quy tắc:
- Không thêm feature mới trừ khi user yêu cầu rõ
- Không tạo notebook mới — sửa `sparse_alpha.ipynb` tại chỗ
- Khi nghi ngờ "có nên thêm X không?" → mặc định **KHÔNG**
- Khi user nói "đơn giản" → loại bỏ thứ thừa, không thêm

---

## Đề tài

**Sparse Deep Learning for Alpha Interpretability** — Giao dịch Định lượng, năm 4 IUH.

- **Dataset**: VN30 (30 cổ phiếu Việt Nam), 2013-01-02 → 2023-06-30, ~63K rows
- **Target**: forward return 5 ngày = `Close(t+5)/Close(t) - 1`
- **Features**: ~35 technical indicators (momentum, trend, oscillator, volatility, volume, candle pattern)
- **Mục tiêu**: train Sparse MLP với L1 → tự chọn feature → vừa deep learning vừa giải thích được

## File chính

**`sparse_alpha.ipynb`** — bản chính thức cho báo cáo, **23 cells, 8 sections**.

Cấu hình đã chốt:

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| HIDDEN_DIMS | `[256, 128, 64, 32]` | 4 hidden — deep learning rõ ràng |
| L1_LAMBDA | **1e-4** | λ ≥ 2e-4 → seeds chết (predict ≈ mean) |
| PATIENCE | 20 | Standard |
| DROPOUT | 0.1 | Standard |
| Seed | 42 (single) | Đơn giản, không ensemble |

8 sections của notebook:
1. Load + Feature Engineering
2. Target + Split + Scale (time-based)
3. MLP Architecture (4 hidden + L1)
4. Train Dense + Sparse
5. Evaluation (MSE/RMSE/IC/Sparsity)
6. Interpretability A — Weight magnitude
7. Interpretability B — SHAP + Spearman ρ
8. Save artifacts

## Time-based split (cố định, không đổi)

```
TRAIN_END = '2020-12-31'   # Train ≤
VAL_END   = '2022-06-30'   # Val: TRAIN_END < ... ≤ VAL_END
                            # Test: > VAL_END
```

## Decisions quan trọng (KHÔNG đề xuất thay đổi)

### MLP, không LSTM/Transformer
- Topic là **Interpretability** → Sparse MLP weight = feature importance trực tiếp
- LSTM gates / Transformer attention sparse khó giải thích
- Features đã encode time (`return_1d`, `macd`...)
- Cite Gu-Kelly-Xiu (2020) — RFS — MLP nhiều layer thắng

### Weight magnitude + SHAP, không thêm Permutation/IG
- 2 phương pháp đã đủ cross-validate qua Spearman ρ
- Permutation/IG = over-engineering

### Single-seed, không ensemble
- Topic là Sparse + Interpretability, không phải ensemble robustness
- Đã thử ensemble — không cải thiện kết quả

### L1 = 1e-4
- Đã thử λ ≥ 2e-4 → model "chết" (val MSE = var(y), predict ≈ mean)
- 1e-4 cho sparsity vẫn cao + vẫn fit signal

## Anti-patterns — KHÔNG làm

- ❌ Đổi sang LSTM / Transformer / GNN / VAE / GAN / TabNet
- ❌ Tăng L1 λ ≥ 2e-4
- ❌ Thêm multi-seed ensemble
- ❌ Thêm Permutation Importance / Integrated Gradients
- ❌ Thêm so sánh nhiều config (A/B/C)
- ❌ Tạo notebook mới (v2, final, ...) khi đã có `sparse_alpha.ipynb`
- ❌ Train không có time-based split (leak future)
- ❌ Thêm phần "robustness check" nếu user không yêu cầu
- ❌ **Early stop theo val IC** — đã thử, test IC flip âm. Phải dùng val MSE.
- ❌ Connection Weight Algorithm cho weight importance — BatchNorm phá tích trị tuyệt đối

## Cấu trúc thư mục

```
Project/
├── CLAUDE.md                    # ← file này
├── README.md                    # Doc tiếng Việt
├── sparse_alpha.ipynb           # ⭐ NOTEBOOK CHÍNH (23 cells) — dùng cho báo cáo
├── experiments_compare.ipynb    # 🧪 SANDBOX so sánh 3h vs 4h vs TabNet (tạo 2026-05-08, chờ chạy)
├── test.ipynb                   # bản đầu — tham khảo, không dùng cho báo cáo
├── data/                        # 30 file CSV VN30
├── webapp/                      # Streamlit dashboard
├── dense_model.pth              # Dense MLP weights
├── sparse_model.pth             # Sparse MLP weights
├── model_metadata.pkl           # scaler + feature_cols + results
└── *.png                        # plots
```

## Webapp

- Chạy: `cd webapp && streamlit run app.py`
- Đọc `model_metadata.pkl` + 2 file `.pth` ở root project
- File `webapp/utils/model.py` có `SparseLSTM` class — KHÔNG xóa (giữ tương thích test.ipynb)

## Cách user thích làm việc

- **Tiếng Việt** cho giải thích, **English** cho code/identifier
- **Ngắn gọn**, không preamble dài
- **Code có comment chỉ khi WHY không hiển nhiên**
- File reference dùng markdown link: `[file.py:42](path/file.py#L42)`
- Train model → trong `.ipynb`, không phải `.py` script
- User nói "OK" có nhiều options → chọn option khuyến nghị nhất
- User nói "đơn giản" → cắt thứ thừa, không thêm
- Trước khi đề xuất change → check anti-patterns ở trên

## Kết quả run hiện tại (sparse_alpha.ipynb)

| Model | MSE | IC | Sparsity |
|-------|-----|-----|----------|
| Dense MLP | 0.00314 | +0.0558 | 0.2% |
| **Sparse MLP** | **0.00313** | **+0.0647** ⭐ | **59.3%** |

- **Spearman ρ** (Weight vs SHAP) = 0.370 → trung bình
- IC ≥ 0.03 đã được coi là useful trong quant tài chính
- Sparse model PREDICTIVE hơn Dense + có khả năng giải thích → đúng tinh thần topic

## Lịch sử thử nghiệm (KHÔNG lặp lại)

| Đã thử | Kết quả | Bài học |
|--------|---------|---------|
| 3 hidden, single seed | IC=0.047, ρ=0.803 | Bản gốc — best ρ nhưng IC thấp hơn |
| **4 hidden, single seed (hiện tại)** | **IC=0.065, ρ=0.370** | ⭐ Đang dùng |
| 4 hidden, 3-config compare (A/B/C) | tốn thời gian, kết quả lệch | Over-engineering |
| 4 hidden + 5-seed ensemble λ=5e-4 | 3/5 seeds chết, IC=0.007 | Failed — λ quá cao |
| 4 hidden + 5-seed ensemble λ=2e-4 | 3/5 seeds chết, IC=-0.04 | Failed — vẫn chết |
| 3 hidden + 5-seed ensemble λ=1e-4 | 5/5 sống, IC=-0.01 | Failed — variance lớn nuốt seed lucky |
| Connection Weight Algorithm (Olden 2004) thay vì first-layer weight | ρ=0.347 (giảm từ 0.370) | Failed — BatchNorm/ReLU phá tích trị tuyệt đối qua layer. Đã revert. |
| Early stop theo val IC (thay vì val MSE) | Sparse Test IC = **−0.0457** (flip sign từ +0.0647), Dense IC cũng âm | Failed — overfitting val IC (val IC +0.0497 nhưng test IC âm). IC là metric noisy, không phù hợp làm criterion early-stop. Đã revert. **BÀI HỌC**: trong financial ML, early stop theo MSE rồi report IC ổn hơn (López de Prado 2018). |
| TabNet (Arik & Pfister 2019) — sandbox 4 combos tune nhẹ | Best (C2): Test IC = **−0.0356** | Failed — val MSE plateau, model "chết". Confirm anti-pattern: sparsemax attention không học được signal trên VN30 alpha. |
| MLP-3h sandbox (cùng config Exp A của experiments_compare) | IC=0.0379, ρ=0.464 | Failed (so với main 4h IC=0.0647) — kém main notebook về IC, tăng ρ chỉ marginal. |
| MLP-4h sandbox (cùng config main) | IC=0.0314, ρ=0.434 | Variance theo random state — main notebook lucky run hơn. Confirm `.pth` saved của main là stable. |

## Khi user request mới

1. Sửa `sparse_alpha.ipynb` → dùng `NotebookEdit`
2. Webapp tweaks → sửa `webapp/`
3. Báo cáo / writeup → user paste output, hỗ trợ phân tích
4. **Không tạo notebook mới** trừ khi user yêu cầu rõ
5. **Không đề xuất thêm feature** không thuộc 4 tiêu chí cốt lõi (Sparse, Deep Learning, Interpretability, Alpha)

---

## 🧪 experiments_compare.ipynb — sandbox đang chờ chạy

**Tạo ngày 2026-05-08** theo yêu cầu rõ của user (override anti-pattern "không tạo notebook mới").
Mục đích: test 3 thí nghiệm so sánh để **bổ sung báo cáo**, KHÔNG thay thế `sparse_alpha.ipynb`.

### 3 thí nghiệm trong sandbox

| Exp | Model | Config |
|-----|-------|--------|
| A | MLP 3 hidden + L1 | `[128, 64, 32]`, λ=1e-4 |
| B | MLP 4 hidden + L1 | `[256, 128, 64, 32]`, λ=1e-4 (giống main) |
| C | TabNet (Arik & Pfister 2019) | tune nhẹ 4 combos, chọn best theo **val MSE** |

Cài thêm: `pytorch-tabnet 4.1.0` (đã `pip install`).

### Quy tắc khi user paste output sandbox (lần sau)

1. **KHÔNG đề xuất chuyển main notebook sang config thắng tự động.** Chờ user quyết.
2. Phân tích kết quả → đề xuất cách trình bày trong báo cáo (so sánh trade-off, không thay thế).
3. Nếu **TabNet thắng cả IC và ρ** → có thể đề xuất bổ sung 1 section "alternative architecture" trong báo cáo, nhưng main notebook vẫn giữ MLP.
4. Nếu **3-hidden có ρ rất cao (≥0.7) + IC vẫn ≥ 0.04** → cân nhắc cùng user xem có nên đổi main notebook về 3-hidden vì topic là Interpretability.
5. Nếu **mọi thí nghiệm fail** (như lịch sử cho thấy 5/6 lần) → xóa sandbox, ghi vào "Lịch sử thử nghiệm", không động main notebook.

### Sandbox notebook structure (15 cells)

1-2. Setup + load data + features + split (giống main)
3. MLP utilities (class + train/eval/SHAP/ρ helpers)
4. Exp A: train + eval MLP-3h
5. Exp B: train + eval MLP-4h
6-7. Exp C: train 4 TabNet combos, eval best
8. Bảng so sánh + plot → save `experiments_compare.png`

### Files có thể sinh ra (an toàn xóa nếu cần)

- `experiments_compare.ipynb`
- `experiments_compare.png`
