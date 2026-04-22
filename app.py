"""
Sparse Deep Learning for Alpha Interpretability - Web App
=========================================================
Dashboard + Dự đoán cho cổ phiếu VN30

Cách chạy:
    pip install streamlit plotly pandas-ta-openbb
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Sparse Alpha - VN30",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e8ecf1;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .green { color: #10b981; }
    .red { color: #ef4444; }
    .blue { color: #3b82f6; }
    .purple { color: #8b5cf6; }
    
    .pred-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .pred-box.negative {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-color: #fca5a5;
    }
    
    div[data-testid="stSidebar"] {
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL DEFINITION (same as notebook)
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_sparsity_ratio(self, threshold=1e-4):
        total = sum(p.numel() for p in self.parameters())
        near_zero = sum((p.abs() < threshold).sum().item() for p in self.parameters())
        return near_zero / total if total > 0 else 0

# ============================================================
# FEATURE ENGINEERING (same as notebook)
# ============================================================
def create_features(group):
    df = group.copy().reset_index(drop=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    
    close, high, low = df['Close'], df['High'], df['Low']
    volume, open_price = df['Volume'], df['Open']
    
    for p in [1, 3, 5, 10, 20, 60]:
        df[f'return_{p}d'] = close.pct_change(p)
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = ta.roc(close, length=p)
    for p in [5, 10, 20, 50]:
        sma = ta.sma(close, length=p)
        df[f'close_to_sma_{p}'] = close / sma
    
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd'] = macd_df.iloc[:, 0]
        df['macd_hist'] = macd_df.iloc[:, 1]
        df['macd_signal'] = macd_df.iloc[:, 2]
    
    df['rsi_14'] = ta.rsi(close, length=14)
    df['rsi_7'] = ta.rsi(close, length=7)
    
    stoch_df = ta.stoch(high, low, close, k=14, d=3)
    if stoch_df is not None:
        df['stoch_k'] = stoch_df.iloc[:, 0]
        df['stoch_d'] = stoch_df.iloc[:, 1]
    
    df['willr_14'] = ta.willr(high, low, close, length=14)
    df['cci_20'] = ta.cci(high, low, close, length=20)
    
    atr = ta.atr(high, low, close, length=14)
    df['atr_ratio'] = atr / close
    
    bb_df = ta.bbands(close, length=20, std=2)
    if bb_df is not None:
        df['bb_pct_b'] = (close - bb_df.iloc[:, 0]) / (bb_df.iloc[:, 2] - bb_df.iloc[:, 0])
        df['bb_width'] = (bb_df.iloc[:, 2] - bb_df.iloc[:, 0]) / bb_df.iloc[:, 1]
    
    df['std_ratio'] = close.rolling(20).std() / close
    
    vol_sma_5 = ta.sma(volume, length=5)
    vol_sma_20 = ta.sma(volume, length=20)
    df['vol_ratio_5_20'] = vol_sma_5 / vol_sma_20
    df['vol_change_1d'] = volume.pct_change(1)
    df['vol_change_5d'] = volume.pct_change(5)
    obv = ta.obv(close, volume)
    if obv is not None:
        df['obv_norm'] = obv / vol_sma_20
    
    df['hl_ratio'] = (high - low) / close
    df['co_ratio'] = (close - open_price) / open_price
    df['gap'] = (open_price - close.shift(1)) / close.shift(1)
    df['upper_shadow'] = (high - np.maximum(close, open_price)) / close
    df['lower_shadow'] = (np.minimum(close, open_price) - low) / close
    
    return df

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_resource
def load_models_and_data():
    """Load saved models và metadata."""
    with open('model_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    device = torch.device('cpu')
    input_dim = meta['input_dim']
    
    dense_model = MLP(input_dim).to(device)
    dense_model.load_state_dict(torch.load('dense_model.pth', map_location=device, weights_only=True))
    dense_model.eval()
    
    sparse_model = MLP(input_dim).to(device)
    sparse_model.load_state_dict(torch.load('sparse_model.pth', map_location=device, weights_only=True))
    sparse_model.eval()
    
    return dense_model, sparse_model, meta

@st.cache_data
def load_stock_data(data_dir, symbols):
    """Load tất cả CSV files."""
    all_dfs = []
    for symbol in symbols:
        fp = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            all_dfs.append(df)
    if not all_dfs:
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    combined['TradingDate'] = pd.to_datetime(combined['TradingDate'], format='%d/%m/%Y')
    combined = combined.drop(columns=['Time', 'Value'], errors='ignore')
    combined = combined.sort_values(['Symbol', 'TradingDate']).reset_index(drop=True)
    return combined

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình")
    
    DATA_DIR = st.text_input("📂 Đường dẫn data", value="D:/S2_Year4/Quantitative Trading/Project/data",
                              help="Thư mục chứa 30 file CSV VN30")
    
    st.markdown("---")
    st.markdown("### 📖 Về đề tài")
    st.markdown("""
    **Sparse Deep Learning for Alpha Interpretability**
    
    Dùng mô hình MLP với L1 Regularization để tìm và giải thích 
    tín hiệu alpha trên thị trường VN30.
    
    - 🔴 **Dense MLP**: Baseline (không sparse)
    - 🟢 **Sparse MLP**: L1 Regularization
    """)
    
    st.markdown("---")
    st.markdown("*IUH - Giao dịch Định lượng*")

# ============================================================
# LOAD
# ============================================================
try:
    dense_model, sparse_model, meta = load_models_and_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Không tìm thấy model files. Hãy chạy notebook trước để tạo `dense_model.pth`, `sparse_model.pth`, `model_metadata.pkl`.\n\nLỗi: {e}")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📈 Sparse Deep Learning for Alpha Interpretability</h1>
    <p>Học sâu thưa cho khả năng giải thích tín hiệu Alpha — Thị trường VN30</p>
</div>
""", unsafe_allow_html=True)

if models_loaded:
    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Dự đoán", "🔬 Phân tích Weights"])
    
    # ============================================================
    # TAB 1: DASHBOARD
    # ============================================================
    with tab1:
        st.markdown("### 📊 So sánh Dense MLP vs Sparse MLP")
        
        dense_res = meta['dense_res']
        sparse_res = meta['sparse_res']
        
        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            diff_mse = ((sparse_res['mse'] - dense_res['mse']) / dense_res['mse']) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="value green">{sparse_res['mse']:.6f}</div>
                <div class="label">MSE (Sparse)</div>
                <div style="font-size:0.8rem; color:#6b7280; margin-top:4px;">Dense: {dense_res['mse']:.6f} ({diff_mse:+.1f}%)</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value green">{sparse_res['rmse']:.6f}</div>
                <div class="label">RMSE (Sparse)</div>
                <div style="font-size:0.8rem; color:#6b7280; margin-top:4px;">Dense: {dense_res['rmse']:.6f}</div>
            </div>""", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value blue">{sparse_res['ic']:.4f}</div>
                <div class="label">IC (Sparse)</div>
                <div style="font-size:0.8rem; color:#6b7280; margin-top:4px;">Dense: {dense_res['ic']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value purple">{sparse_res['sparsity']:.1%}</div>
                <div class="label">Sparsity Ratio</div>
                <div style="font-size:0.8rem; color:#6b7280; margin-top:4px;">Dense: {dense_res['sparsity']:.1%}</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Bar chart: Metrics comparison
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'RMSE', 'IC', 'Sparsity'],
                'Dense MLP': [dense_res['mse'], dense_res['rmse'], dense_res['ic'], dense_res['sparsity']],
                'Sparse MLP': [sparse_res['mse'], sparse_res['rmse'], sparse_res['ic'], sparse_res['sparsity']]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Dense MLP', x=['MSE', 'RMSE'], 
                                y=[dense_res['mse'], dense_res['rmse']], 
                                marker_color='#ef4444', opacity=0.85))
            fig.add_trace(go.Bar(name='Sparse MLP', x=['MSE', 'RMSE'], 
                                y=[sparse_res['mse'], sparse_res['rmse']], 
                                marker_color='#10b981', opacity=0.85))
            fig.update_layout(title='So sánh MSE & RMSE', barmode='group',
                            template='plotly_white', height=400,
                            font=dict(family='Space Grotesk'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            # Bar chart: IC & Sparsity
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Dense MLP', x=['IC', 'Sparsity Ratio'],
                                y=[dense_res['ic'], dense_res['sparsity']],
                                marker_color='#ef4444', opacity=0.85))
            fig.add_trace(go.Bar(name='Sparse MLP', x=['IC', 'Sparsity Ratio'],
                                y=[sparse_res['ic'], sparse_res['sparsity']],
                                marker_color='#10b981', opacity=0.85))
            fig.update_layout(title='So sánh IC & Sparsity', barmode='group',
                            template='plotly_white', height=400,
                            font=dict(family='Space Grotesk'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("### 🏆 Feature Importance (Sparse MLP)")
        
        sparse_fi = meta['sparse_fi']
        top_n = st.slider("Số features hiển thị", 5, 35, 15)
        top_fi = sparse_fi.head(top_n)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_fi['feature'][::-1],
            x=top_fi['importance'][::-1],
            orientation='h',
            marker=dict(
                color=top_fi['importance'][::-1],
                colorscale='Emrld',
                showscale=True,
                colorbar=dict(title="Importance")
            )
        ))
        fig.update_layout(
            title=f'Top {top_n} Features quan trọng nhất (Sparse MLP, λ={meta["best_lambda"]})',
            xaxis_title='Sum of |Weights|',
            template='plotly_white', height=max(400, top_n * 28),
            font=dict(family='Space Grotesk')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature explanation table
        explanations = {
            'upper_shadow': 'Bóng trên nến — áp lực bán',
            'lower_shadow': 'Bóng dưới nến — áp lực mua',
            'willr_14': 'Williams %R — quá mua/bán',
            'atr_ratio': 'ATR Ratio — biến động tương đối',
            'stoch_d': 'Stochastic %D — tín hiệu dao động',
            'stoch_k': 'Stochastic %K — dao động giá',
            'cci_20': 'CCI — chỉ số kênh hàng hóa',
            'vol_ratio_5_20': 'Volume Ratio — thay đổi khối lượng',
            'close_to_sma_50': 'Giá/SMA50 — xu hướng dài hạn',
            'bb_pct_b': 'Bollinger %B — vị trí trong dải',
            'return_10d': 'Momentum 10 ngày',
            'roc_20': 'Rate of Change 20 ngày',
            'return_60d': 'Momentum quý',
            'return_20d': 'Momentum tháng',
            'roc_10': 'Rate of Change 10 ngày',
            'macd': 'MACD — xu hướng momentum',
        }
        
        st.markdown("### 📋 Giải thích Top Features")
        explain_df = top_fi.head(10).copy()
        explain_df['Ý nghĩa tài chính'] = explain_df['feature'].map(
            lambda x: explanations.get(x, '—')
        )
        explain_df.columns = ['Feature', 'Importance', 'Ý nghĩa tài chính']
        explain_df.index = range(1, len(explain_df) + 1)
        st.dataframe(explain_df, use_container_width=True)
    
    # ============================================================
    # TAB 2: DỰ ĐOÁN
    # ============================================================
    with tab2:
        st.markdown("### 🔮 Dự đoán Forward Return 5 ngày")
        st.markdown("Chọn mã cổ phiếu → Mô hình sẽ dự đoán return 5 ngày tới và giải thích dựa vào features nào.")
        
        df_all = load_stock_data(DATA_DIR, meta['vn30_symbols'])
        
        if df_all is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                available = sorted(df_all['Symbol'].unique())
                symbol = st.selectbox("📌 Chọn mã cổ phiếu", available, index=available.index('VCB') if 'VCB' in available else 0)
            
            with col2:
                st.markdown("")
            
            if st.button("🚀 Dự đoán", type="primary", use_container_width=True):
                with st.spinner("Đang tính toán..."):
                    # Lấy data của mã đã chọn
                    df_symbol = df_all[df_all['Symbol'] == symbol].copy()
                    
                    # Tạo features
                    df_feat = create_features(df_symbol)
                    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(df_feat) == 0:
                        st.error("Không đủ dữ liệu để tính features!")
                    else:
                        # Lấy dòng cuối cùng (mới nhất)
                        latest = df_feat.iloc[-1]
                        latest_date = latest['TradingDate']
                        latest_close = latest['Close']
                        
                        # Chuẩn bị input
                        feature_cols = meta['feature_cols']
                        X = latest[feature_cols].values.reshape(1, -1).astype(float)
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        X_scaled = meta['scaler'].transform(X)
                        X_tensor = torch.FloatTensor(X_scaled)
                        
                        # Dự đoán
                        with torch.no_grad():
                            dense_pred = dense_model(X_tensor).item()
                            sparse_pred = sparse_model(X_tensor).item()
                        
                        # Hiển thị kết quả
                        st.markdown("---")
                        st.markdown(f"#### 📅 Dữ liệu mới nhất: {latest_date.strftime('%d/%m/%Y')} | Giá đóng cửa: **{latest_close:,.0f} VNĐ**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            css_class = "pred-box" if dense_pred >= 0 else "pred-box negative"
                            arrow = "📈" if dense_pred >= 0 else "📉"
                            st.markdown(f"""
                            <div class="{css_class}">
                                <h3 style="margin:0; color:#6b7280;">Dense MLP</h3>
                                <div style="font-size:2.5rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:0.5rem 0;">
                                    {arrow} {dense_pred:+.2%}
                                </div>
                                <div style="color:#6b7280;">
                                    Giá dự đoán sau 5 ngày: <b>{latest_close * (1 + dense_pred):,.0f} VNĐ</b>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            css_class = "pred-box" if sparse_pred >= 0 else "pred-box negative"
                            arrow = "📈" if sparse_pred >= 0 else "📉"
                            st.markdown(f"""
                            <div class="{css_class}">
                                <h3 style="margin:0; color:#6b7280;">Sparse MLP ⭐</h3>
                                <div style="font-size:2.5rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:0.5rem 0;">
                                    {arrow} {sparse_pred:+.2%}
                                </div>
                                <div style="color:#6b7280;">
                                    Giá dự đoán sau 5 ngày: <b>{latest_close * (1 + sparse_pred):,.0f} VNĐ</b>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Giải thích: hiển thị feature values
                        st.markdown("---")
                        st.markdown("#### 🔍 Giải thích dự đoán (Sparse MLP)")
                        st.markdown("Dựa trên top features quan trọng nhất:")
                        
                        sparse_fi = meta['sparse_fi']
                        top_features = sparse_fi.head(10)
                        
                        feat_values = []
                        for _, row in top_features.iterrows():
                            fname = row['feature']
                            fval = latest[fname] if fname in latest.index else 0
                            feat_values.append({
                                'Feature': fname,
                                'Giá trị hiện tại': f"{fval:.4f}",
                                'Importance': f"{row['importance']:.4f}",
                                'Ý nghĩa': explanations.get(fname, '—')
                            })
                        
                        st.dataframe(pd.DataFrame(feat_values), use_container_width=True)
                        
                        # Biểu đồ giá gần đây
                        st.markdown(f"#### 📈 Biểu đồ giá {symbol} (60 ngày gần nhất)")
                        recent = df_symbol.tail(60)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=recent['TradingDate'],
                            open=recent['Open'], high=recent['High'],
                            low=recent['Low'], close=recent['Close'],
                            name=symbol
                        ))
                        fig.update_layout(
                            template='plotly_white', height=450,
                            xaxis_rangeslider_visible=False,
                            font=dict(family='Space Grotesk'),
                            title=f'{symbol} - Nến Nhật 60 ngày'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("⚠️ Lưu ý: Đây là kết quả từ mô hình nghiên cứu, không phải khuyến nghị đầu tư.")
        
        else:
            st.warning(f"⚠️ Không tìm thấy dữ liệu tại `{DATA_DIR}`. Hãy kiểm tra đường dẫn ở sidebar.")
    
    # ============================================================
    # TAB 3: PHÂN TÍCH WEIGHTS
    # ============================================================
    with tab3:
        st.markdown("### 🔬 Phân tích Weight Heatmap")
        st.markdown("So sánh ma trận weights của layer đầu tiên giữa Dense MLP và Sparse MLP.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Dense MLP")
            fig = px.imshow(
                np.abs(meta['dense_w']),
                color_continuous_scale='YlOrRd',
                labels=dict(x="Input Features", y="Hidden Neurons", color="|Weight|"),
                aspect='auto'
            )
            fig.update_layout(title='Dense MLP - |Weights| Layer 1',
                            template='plotly_white', height=500,
                            font=dict(family='Space Grotesk'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🟢 Sparse MLP")
            fig = px.imshow(
                np.abs(meta['sparse_w']),
                color_continuous_scale='YlOrRd',
                labels=dict(x="Input Features", y="Hidden Neurons", color="|Weight|"),
                aspect='auto'
            )
            fig.update_layout(title=f'Sparse MLP (λ={meta["best_lambda"]}) - |Weights| Layer 1',
                            template='plotly_white', height=500,
                            font=dict(family='Space Grotesk'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Weight Distribution
        st.markdown("### 📊 Phân phối Weights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=meta['dense_w'].flatten(), nbinsx=100,
                marker_color='#ef4444', opacity=0.75, name='Dense'
            ))
            fig.update_layout(title='Dense MLP - Phân phối Weights',
                            xaxis_title='Weight Value', yaxis_title='Count',
                            template='plotly_white', height=350,
                            font=dict(family='Space Grotesk'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=meta['sparse_w'].flatten(), nbinsx=100,
                marker_color='#10b981', opacity=0.75, name='Sparse'
            ))
            fig.update_layout(title='Sparse MLP - Phân phối Weights',
                            xaxis_title='Weight Value', yaxis_title='Count',
                            template='plotly_white', height=350,
                            font=dict(family='Space Grotesk'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("### 📋 Thống kê Sparsity")
        
        feature_cols = meta['feature_cols']
        sparse_fi = meta['sparse_fi']
        
        total_params_dense = sum(p.numel() for p in dense_model.parameters())
        total_params_sparse = sum(p.numel() for p in sparse_model.parameters())
        
        stats_df = pd.DataFrame({
            'Thống kê': [
                'Tổng tham số', 
                'Weights ≈ 0 (< 1e-4)', 
                'Sparsity Ratio',
                'Features importance > 0.01',
                'Features bị loại (importance < 0.01)'
            ],
            'Dense MLP': [
                f"{total_params_dense:,}",
                f"{int(total_params_dense * dense_model.get_sparsity_ratio()):,}",
                f"{dense_model.get_sparsity_ratio():.1%}",
                f"{len(meta['dense_fi'][meta['dense_fi']['importance'] >= 0.01])}/{len(feature_cols)}",
                f"{len(meta['dense_fi'][meta['dense_fi']['importance'] < 0.01])}/{len(feature_cols)}"
            ],
            'Sparse MLP': [
                f"{total_params_sparse:,}",
                f"{int(total_params_sparse * sparse_model.get_sparsity_ratio()):,}",
                f"{sparse_model.get_sparsity_ratio():.1%}",
                f"{len(sparse_fi[sparse_fi['importance'] >= 0.01])}/{len(feature_cols)}",
                f"{len(sparse_fi[sparse_fi['importance'] < 0.01])}/{len(feature_cols)}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)