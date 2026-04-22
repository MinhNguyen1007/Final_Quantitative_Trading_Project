"""
Prediction - Tab dự đoán Forward Return cho từng mã
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data_loader import load_single_stock
from utils.features import create_features, prepare_features, FEATURE_EXPLANATIONS
from utils.model import predict
from config import DATA_DIR, FORWARD_DAYS


def render(dense_model, sparse_model, meta):
    """Render tab Dự đoán."""
    st.markdown("### 🔮 Dự đoán Forward Return 5 ngày")
    st.markdown("Chọn mã cổ phiếu → Mô hình dự đoán return 5 ngày tới và giải thích.")

    symbols = meta['vn30_symbols']
    default_idx = symbols.index('VCB') if 'VCB' in symbols else 0
    symbol = st.selectbox("📌 Chọn mã cổ phiếu", symbols, index=default_idx)

    if st.button("🚀 Dự đoán", type="primary", use_container_width=True):
        with st.spinner("Đang tính toán..."):
            df_symbol = load_single_stock(DATA_DIR, symbol)

            if df_symbol is None:
                st.error(f"Không tìm thấy dữ liệu cho {symbol}")
                return

            df_feat, X_scaled = prepare_features(
                df_symbol, meta['scaler'], meta['feature_cols']
            )

            if df_feat is None:
                st.error("Không đủ dữ liệu để tính features!")
                return

            # Lấy dòng cuối (mới nhất)
            latest = df_feat.iloc[-1]
            latest_date = latest['TradingDate']
            latest_close = latest['Close']
            X_last = X_scaled[-1:, :]

            # Dự đoán
            dense_pred = predict(dense_model, X_last)[0]
            sparse_pred = predict(sparse_model, X_last)[0]

            # --- Hiển thị ---
            st.markdown("---")
            st.markdown(f"#### 📅 Dữ liệu mới nhất: {latest_date.strftime('%d/%m/%Y')} | "
                       f"Giá đóng cửa: **{latest_close:,.0f} VNĐ**")

            col1, col2 = st.columns(2)

            for col, pred, name, star in [(col1, dense_pred, "Dense MLP", ""),
                                           (col2, sparse_pred, "Sparse MLP", " ⭐")]:
                css = "pred-box" if pred >= 0 else "pred-box negative"
                arrow = "📈" if pred >= 0 else "📉"
                price = latest_close * (1 + pred)

                with col:
                    st.markdown(f"""
                    <div class="{css}">
                        <h3 style="margin:0; color:#6b7280;">{name}{star}</h3>
                        <div style="font-size:2.5rem; font-weight:700; font-family:'JetBrains Mono',monospace; margin:0.5rem 0;">
                            {arrow} {pred:+.2%}
                        </div>
                        <div style="color:#6b7280;">
                            Giá dự đoán sau 5 ngày: <b>{price:,.0f} VNĐ</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # --- Giải thích ---
            st.markdown("---")
            st.markdown("#### 🔍 Giải thích dự đoán (Sparse MLP)")

            sparse_fi = meta['sparse_fi']
            top_feats = sparse_fi.head(10)

            rows = []
            for _, row in top_feats.iterrows():
                fname = row['feature']
                fval = latest[fname] if fname in latest.index else 0
                rows.append({
                    'Feature': fname,
                    'Giá trị hiện tại': f"{fval:.4f}",
                    'Importance': f"{row['importance']:.4f}",
                    'Ý nghĩa': FEATURE_EXPLANATIONS.get(fname, '—')
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # --- Biểu đồ nến ---
            st.markdown(f"#### 📈 Biểu đồ giá {symbol} (60 ngày gần nhất)")
            recent = df_symbol.tail(60)

            fig = go.Figure(go.Candlestick(
                x=recent['TradingDate'],
                open=recent['Open'], high=recent['High'],
                low=recent['Low'], close=recent['Close'],
                name=symbol
            ))
            fig.update_layout(
                template='plotly_white', height=450,
                xaxis_rangeslider_visible=False,
                title=f'{symbol} - Nến Nhật 60 ngày'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("⚠️ Đây là kết quả từ mô hình nghiên cứu, không phải khuyến nghị đầu tư.")
