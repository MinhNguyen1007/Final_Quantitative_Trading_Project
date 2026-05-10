"""
Interpretability — Cross-validate Weight magnitude vs SHAP qua Spearman ρ.
Dùng artifacts đã có trong model_metadata.pkl (sparse_fi, sparse_shap, spearman_rho).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def _topk_bar(df, value_col, title, color, top_n=15):
    top = df.head(top_n).copy()
    fig = go.Figure(go.Bar(
        x=top[value_col][::-1],
        y=top['feature'][::-1],
        orientation='h',
        marker=dict(color=color, opacity=0.85),
    ))
    fig.update_layout(title=title, template='plotly_white',
                      height=max(380, top_n * 26),
                      xaxis_title=value_col)
    return fig


def render(meta, ext=None):
    """Render tab Interpretability — Weight vs SHAP cross-validation."""
    st.markdown("### 🧠 Interpretability — Weight magnitude vs SHAP")
    st.caption(
        "Cross-validate 2 phương pháp giải thích model bằng Spearman ρ. "
        "ρ ≥ 0.7: đáng tin cậy. 0.3 ≤ ρ < 0.7: có tương quan. ρ < 0.3: mâu thuẫn."
    )

    sparse_fi   = meta.get('sparse_fi')
    sparse_shap = meta.get('sparse_shap')
    rho         = meta.get('spearman_rho')

    if sparse_fi is None or sparse_shap is None:
        st.warning("Thiếu artifact interpretability trong `model_metadata.pkl`. "
                   "Chạy lại `sparse_alpha.ipynb` để sinh.")
        return

    # --- Spearman ρ banner ---
    rho_color = '#10b981' if rho >= 0.7 else '#f59e0b' if rho >= 0.3 else '#ef4444'
    rho_label = ('ĐÁNG TIN CẬY' if rho >= 0.7
                 else 'CÓ TƯƠNG QUAN' if rho >= 0.3
                 else 'MÂU THUẪN')

    st.markdown(f"""
    <div style="background: {rho_color}20; border-left: 4px solid {rho_color};
                padding: 1rem 1.5rem; border-radius: 8px; margin: 1rem 0;">
        <div style="font-size: 0.85rem; color: #6b7280; text-transform: uppercase;
                    letter-spacing: 1px;">Spearman ρ (Weight ↔ SHAP)</div>
        <div style="font-size: 2rem; font-weight: 700; color: {rho_color};
                    font-family: 'JetBrains Mono', monospace;">{rho:.3f}</div>
        <div style="font-size: 0.9rem; color: #374151; margin-top: 0.3rem;">{rho_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Side-by-side top features ---
    st.markdown("#### So sánh Top features")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _topk_bar(sparse_fi, 'importance',
                      'Weight Magnitude — Sparse MLP', '#2ECC71'),
            use_container_width=True)
    with col2:
        st.plotly_chart(
            _topk_bar(sparse_shap, 'importance',
                      'SHAP — Sparse MLP', '#3498DB'),
            use_container_width=True)

    # --- Top 10 ranking comparison ---
    st.markdown("#### Bảng so sánh ranking Top 10")
    side = pd.DataFrame({
        'Rank':   range(1, 11),
        'Weight': sparse_fi['feature'].head(10).values,
        'SHAP':   sparse_shap['feature'].head(10).values,
    })
    st.dataframe(side, use_container_width=True, hide_index=True)

    st.caption(
        "**Diễn giải**: Weight magnitude đo *structural importance* (model cấu trúc "
        "ưu tiên feature nào ở input layer). SHAP đo *functional importance* "
        "(feature đóng góp gì cho prediction trên test set). Hai cách đo khác nhau "
        "→ ranking không trùng hoàn toàn là kỳ vọng được, đặc biệt với deep network."
    )
