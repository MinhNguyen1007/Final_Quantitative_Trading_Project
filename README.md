# Sparse Deep Learning for Alpha Interpretability (VN30)

Project nay trien khai bai toan du doan forward return 5 ngay cho thi truong VN30, ket hop:

- Dense MLP (baseline)
- Sparse MLP voi L1 regularization de tang kha nang giai thich

He thong gom 2 phan:

- Notebook chinh huan luyen va danh gia mo hinh: `sparse_alpha.ipynb` (`test.ipynb` la ban dau, deprecated)
- Web app Streamlit de dashboard, prediction, backtest va phan tich weight: `webapp/`

## 1. Muc tieu bai toan

- Input: du lieu OHLCV theo ngay cua cac ma VN30.
- Feature engineering: technical indicators (momentum, trend, volatility, volume, candlestick pattern).
- Target: forward return 5 ngay.
- So sanh Dense vs Sparse theo:
  - MSE
  - RMSE
  - Information Coefficient (IC)
  - Sparsity ratio
- Giai thich mo hinh qua feature importance va weight analysis.

## 2. Cau truc thu muc

```text
.
|-- sparse_alpha.ipynb             # Notebook chinh huan luyen + luu artifacts (23 cells, 8 sections)
|-- experiments_compare.ipynb      # Sandbox so sanh 3h vs 4h vs TabNet
|-- test.ipynb                     # Ban dau — tham khao, khong dung cho bao cao
|-- dense_model.pth                # Model Dense da train
|-- sparse_model.pth               # Model Sparse da train
|-- model_metadata.pkl             # Scaler, feature cols, metrics, weights
|-- data/
|   |-- ACB.csv
|   |-- ...
|   |-- VN30.csv
|   `-- VRE.csv
|-- webapp/
|   |-- app.py                     # Entry point web app modular
|   |-- config.py                  # Cau hinh DATA_DIR, duong dan model
|   |-- requirements.txt
|   |-- components/
|   |   |-- dashboard.py
|   |   |-- prediction.py
|   |   |-- backtest.py
|   |   `-- weights.py
|   `-- utils/
|       |-- data_loader.py
|       |-- features.py
|       `-- model.py
`-- README.md
```

## 3. Du lieu

Trong thu muc `data/` hien co:

- 31 file CSV
- Header: `Symbol,Value,TradingDate,Time,Open,High,Low,Close,Volume`
- Tong so dong: 66,100
- Khoang thoi gian: 02/01/2013 -> 30/06/2023

Luu y:

- Danh sach ma huan luyen trong code gom 30 ma VN30 (`VN30_SYMBOLS`).
- File `VN30.csv` duoc luu trong data folder nhung khong nam trong danh sach train mac dinh.

## 4. Pipeline huan luyen (sparse_alpha.ipynb)

Notebook `sparse_alpha.ipynb` thuc hien day du pipeline:

1. Load va gop du lieu tu cac file CSV.
2. Chuan hoa cot thoi gian, sap xep theo `Symbol` va `TradingDate`.
3. Tao feature ky thuat:
   - Returns, ROC
   - SMA ratio, MACD
   - RSI, Stochastic, Williams %R, CCI
   - ATR, Bollinger bands, rolling std
   - Volume ratios, OBV
   - Candlestick/gap features
4. Tao target: `Close(t+5)/Close(t)-1`.
5. Chia tap du lieu theo moc thoi gian:
   - Train <= 2020-12-31
   - Val: 2021-01-01 -> 2022-06-30
   - Test > 2022-06-30
6. Scale feature bang `StandardScaler` (fit tren train).
7. Train Dense MLP (baseline).
8. Train Sparse MLP voi L1 lambda = 1e-4 (da chot — lambda lon hon gay model "chet").
9. Cross-validate interpretability bang Spearman rho giua Weight magnitude va SHAP.
10. Danh gia tren test set va ve bieu do so sanh.
11. Trich feature importance tu layer dau (tong `|weights|`).
12. Luu artifacts:
    - `dense_model.pth`
    - `sparse_model.pth`
    - `model_metadata.pkl`
    - `comparison.png`
    - `feature_importance_weight.png`
    - `feature_importance_shap.png`

## 5. Web App (Streamlit)

Web app chinh o `webapp/app.py` (ban modular).

### Cac tab chuc nang

- Dashboard:
  - So sanh metric Dense vs Sparse
  - Bieu do MSE/RMSE, IC/Sparsity
  - Top feature importance + giai thich tai chinh
- Prediction:
  - Chon ma co phieu
  - Du doan forward return 5 ngay (Dense va Sparse)
  - Hien thi top feature tac dong + nennhat 60 ngay
- Backtest:
  - So sanh du doan va thuc te tren lich su
  - IC, direction accuracy, MSE
  - PnL tich luy mo phong theo tin hieu mo hinh
- Weights:
  - Heatmap |weights| layer 1
  - Histogram phan phoi weights
  - Thong ke sparsity

## 6. Cai dat va chay

Yeu cau:

- Python 3.10+ (khuyen nghi 3.10/3.11)

### B1. Tao moi truong ao

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### B2. Cai thu vien cho web app

```bash
pip install -r webapp/requirements.txt
```

Neu can chay notebook huan luyen, cai them:

```bash
pip install jupyter matplotlib seaborn
```

### B3. Chay web app

Paths trong `webapp/config.py` da auto-detect tu vi tri file → khong can chinh tay.

```bash
cd webapp
streamlit run app.py
```

Mac dinh app mo o `http://localhost:8501`.

## 7. Tai tao model tu notebook

De retrain va tao lai artifacts:

1. Mo `sparse_alpha.ipynb`.
2. Chay toan bo cell theo thu tu.
3. Kiem tra da sinh ra cac file `.pth`, `.pkl`, `.png` o root project.
4. Chay lai web app de su dung artifacts moi.

## 8. Luu y quan trong

- Day la du an hoc thuat/nghien cuu, khong phai khuyen nghi dau tu.
- Backtest hien tai la mo phong don gian, chua tinh phi giao dich va slippage.
- Neu gap loi khong tim thay du lieu/model, hay uu tien kiem tra lai cac duong dan trong `webapp/config.py`.

## 9. Huong phat trien

- Them transaction cost, slippage va quan ly vi the vao backtest.
- Thu cac mo hinh sequence (LSTM/Transformer) va so sanh voi MLP.
- Them walk-forward validation va rolling retrain.
- Dong goi pipeline train/eval thanh script CLI de tai lap de hon.
