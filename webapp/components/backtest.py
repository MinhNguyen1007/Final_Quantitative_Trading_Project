"""
Backtest - Tab so sánh dự đoán vs thực tế trên dữ liệu quá khứ
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import load_single_stock
from utils.features import prepare_features
from utils.model import predict
from config import DATA_DIR, FORWARD_DAYS


def render(dense_model, sparse_model, meta):
    """Render tab Backtest."""
    st.markdown("### 🧪 Backtest: Dự đoán vs Thực tế")
    st.markdown("So sánh kết quả dự đoán với return thực tế trên dữ liệu quá khứ.")

    symbols = meta['vn30_symbols']
    default_idx = symbols.index('VCB') if 'VCB' in symbols else 0

    col1, col2 = st.columns([1, 1])
    with col1:
        symbol = st.selectbox("📌 Chọn mã", symbols, index=default_idx, key="bt_symbol")
    with col2:
        n_days = st.slider("Số ngày backtest", 30, 200, 100, key="bt_days")

    if st.button("🧪 Chạy Backtest", type="primary", use_container_width=True):
        with st.spinner(f"Đang backtest {symbol}..."):
            df_symbol = load_single_stock(DATA_DIR, symbol)

            if df_symbol is None:
                st.error(f"Không tìm thấy dữ liệu cho {symbol}")
                return

            df_feat, X_scaled = prepare_features(
                df_symbol, meta['scaler'], meta['feature_cols']
            )

            if df_feat is None or len(df_feat) < n_days + FORWARD_DAYS:
                st.error("Không đủ dữ liệu!")
                return

            # Tính actual return
            df_feat['actual_return'] = df_feat['Close'].shift(-FORWARD_DAYS) / df_feat['Close'] - 1

            # Chỉ lấy n_days cuối (trừ FORWARD_DAYS cuối vì không có actual)
            df_bt = df_feat.iloc[-(n_days + FORWARD_DAYS):-FORWARD_DAYS].copy().reset_index(drop=True)
            X_bt = X_scaled[-(n_days + FORWARD_DAYS):-FORWARD_DAYS]

            # Dự đoán
            df_bt['dense_pred'] = predict(dense_model, X_bt)
            df_bt['sparse_pred'] = predict(sparse_model, X_bt)

            # Loại NaN
            df_bt = df_bt.dropna(subset=['actual_return']).reset_index(drop=True)

            # --- Metrics ---
            st.markdown("---")
            st.markdown(f"#### 📊 Kết quả Backtest {symbol} ({len(df_bt)} ngày)")

            def calc_metrics(pred, actual):
                mse = np.mean((pred - actual) ** 2)
                ic = np.corrcoef(pred, actual)[0, 1]
                direction_acc = np.mean(np.sign(pred) == np.sign(actual))
                return mse, ic, direction_acc

            d_mse, d_ic, d_acc = calc_metrics(df_bt['dense_pred'].values, df_bt['actual_return'].values)
            s_mse, s_ic, s_acc = calc_metrics(df_bt['sparse_pred'].values, df_bt['actual_return'].values)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">IC (Tương quan)</div>
                    <div class="value red">{d_ic:.4f}</div>
                    <div style="font-size:0.8rem; color:#6b7280;">Dense MLP</div>
                    <div class="value green" style="margin-top:8px;">{s_ic:.4f}</div>
                    <div style="font-size:0.8rem; color:#6b7280;">Sparse MLP</div>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Đúng hướng (%)</div>
                    <div class="value red">{d_acc:.1%}</div>
                    <div style="font-size:0.8rem; color:#6b7280;">Dense MLP</div>
                    <div class="value green" style="margin-top:8px;">{s_acc:.1%}</div>
                    <div style="font-size:0.8rem; color:#6b7280;">Sparse MLP</div>
                </div>""", unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">MSE</div>
                    <div class="value red">{d_mse:.6f}</div>
                    <div style="font-size:0.8rem; color:#6b7280;">Dense MLP</div>
                    <div class="value green" style="margin-top:8px;">{s_mse:.6f}</div>
                    <div style="font-size:0.8rem; color:#6b7280;">Sparse MLP</div>
                </div>""", unsafe_allow_html=True)

            # --- Chart: Predicted vs Actual ---
            st.markdown("")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Sparse MLP: Dự đoán vs Thực tế",
                                              "Dense MLP: Dự đoán vs Thực tế"),
                               vertical_spacing=0.12)

            # Sparse
            fig.add_trace(go.Scatter(x=df_bt['TradingDate'], y=df_bt['actual_return'],
                                    name='Thực tế', line=dict(color='#374151', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_bt['TradingDate'], y=df_bt['sparse_pred'],
                                    name='Sparse MLP', line=dict(color='#10b981', width=2, dash='dot')), row=1, col=1)

            # Dense
            fig.add_trace(go.Scatter(x=df_bt['TradingDate'], y=df_bt['actual_return'],
                                    name='Thực tế', line=dict(color='#374151', width=2),
                                    showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_bt['TradingDate'], y=df_bt['dense_pred'],
                                    name='Dense MLP', line=dict(color='#ef4444', width=2, dash='dot')), row=2, col=1)

            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

            fig.update_layout(template='plotly_white', height=600,
                             title=f'{symbol} — Backtest {len(df_bt)} ngày')
            fig.update_yaxes(title_text="Return", row=1, col=1)
            fig.update_yaxes(title_text="Return", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # --- Bảng chi tiết ---
            st.markdown("#### 📋 Bảng chi tiết (20 ngày gần nhất)")

            display_df = df_bt.tail(20)[['TradingDate', 'Close', 'actual_return', 'dense_pred', 'sparse_pred']].copy()
            display_df['TradingDate'] = display_df['TradingDate'].dt.strftime('%d/%m/%Y')
            display_df['Close'] = display_df['Close'].apply(lambda x: f"{x:,.0f}")
            display_df['actual_return'] = display_df['actual_return'].apply(lambda x: f"{x:+.2%}")
            display_df['dense_pred'] = display_df['dense_pred'].apply(lambda x: f"{x:+.2%}")
            display_df['sparse_pred'] = display_df['sparse_pred'].apply(lambda x: f"{x:+.2%}")

            # Đúng hướng
            df_check = df_bt.tail(20).copy()
            display_df['Sparse đúng?'] = (np.sign(df_check['sparse_pred'].values) == np.sign(df_check['actual_return'].values))
            display_df['Sparse đúng?'] = display_df['Sparse đúng?'].map({True: '✅', False: '❌'})

            display_df.columns = ['Ngày', 'Giá', 'Thực tế', 'Dense', 'Sparse', 'Đúng hướng?']
            display_df.index = range(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)

            # --- Cumulative Return ---
            st.markdown("#### 💰 Lợi nhuận tích lũy (giả lập)")
            st.markdown("Nếu mỗi ngày đặt cược theo hướng dự đoán của mô hình:")

            df_bt['sparse_daily_pnl'] = np.sign(df_bt['sparse_pred']) * df_bt['actual_return']
            df_bt['dense_daily_pnl'] = np.sign(df_bt['dense_pred']) * df_bt['actual_return']
            df_bt['sparse_cum'] = (1 + df_bt['sparse_daily_pnl']).cumprod() - 1
            df_bt['dense_cum'] = (1 + df_bt['dense_daily_pnl']).cumprod() - 1
            df_bt['hold_cum'] = (1 + df_bt['actual_return']).cumprod() - 1

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_bt['TradingDate'], y=df_bt['sparse_cum'],
                                    name='Sparse MLP', line=dict(color='#10b981', width=2.5)))
            fig.add_trace(go.Scatter(x=df_bt['TradingDate'], y=df_bt['dense_cum'],
                                    name='Dense MLP', line=dict(color='#ef4444', width=2)))
            fig.add_trace(go.Scatter(x=df_bt['TradingDate'], y=df_bt['hold_cum'],
                                    name='Buy & Hold', line=dict(color='#6b7280', width=1.5, dash='dash')))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
            fig.update_layout(
                title=f'{symbol} — Lợi nhuận tích lũy',
                yaxis_title='Cumulative Return',
                template='plotly_white', height=400,
                yaxis_tickformat='.1%'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("⚠️ Đây là backtest đơn giản, chưa tính phí giao dịch và slippage.")
