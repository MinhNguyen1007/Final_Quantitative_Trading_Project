"""
Dashboard - Tab hiển thị kết quả so sánh Dense vs Sparse
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.features import FEATURE_EXPLANATIONS


def render(meta):
    """Render tab Dashboard."""
    st.markdown("### 📊 So sánh Dense MLP vs Sparse MLP trên Test Set")

    dense_res = meta['dense_res']
    sparse_res = meta['sparse_res']

    # --- Metric Cards ---
    col1, col2, col3, col4 = st.columns(4)

    def fmt_val(val, key):
        if key == 'sparsity':
            return f"{val:.1%}"
        elif key == 'ic':
            return f"{val:.4f}"
        else:
            return f"{val:.6f}"

    cards = [
        ("MSE", 'mse', 'green', col1),
        ("RMSE", 'rmse', 'green', col2),
        ("IC", 'ic', 'blue', col3),
        ("Sparsity", 'sparsity', 'purple', col4),
    ]

    for label, key, color, col in cards:
        sv_str = fmt_val(sparse_res[key], key)
        dv_str = fmt_val(dense_res[key], key)

        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value {color}">{sv_str}</div>
                <div class="label">{label} (Sparse)</div>
                <div style="font-size:0.8rem; color:#6b7280; margin-top:4px;">Dense: {dv_str}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # --- Charts ---
    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Dense MLP', x=['MSE', 'RMSE'],
                            y=[dense_res['mse'], dense_res['rmse']],
                            marker_color='#ef4444', opacity=0.85))
        fig.add_trace(go.Bar(name='Sparse MLP', x=['MSE', 'RMSE'],
                            y=[sparse_res['mse'], sparse_res['rmse']],
                            marker_color='#10b981', opacity=0.85))
        fig.update_layout(title='So sánh MSE & RMSE', barmode='group',
                         template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Dense MLP', x=['IC', 'Sparsity Ratio'],
                            y=[dense_res['ic'], dense_res['sparsity']],
                            marker_color='#ef4444', opacity=0.85))
        fig.add_trace(go.Bar(name='Sparse MLP', x=['IC', 'Sparsity Ratio'],
                            y=[sparse_res['ic'], sparse_res['sparsity']],
                            marker_color='#10b981', opacity=0.85))
        fig.update_layout(title='So sánh IC & Sparsity', barmode='group',
                         template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Feature Importance ---
    st.markdown("### 🏆 Feature Importance (Sparse MLP)")

    sparse_fi = meta['sparse_fi']
    top_n = st.slider("Số features hiển thị", 5, 35, 15)
    top_fi = sparse_fi.head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_fi['feature'][::-1],
        x=top_fi['importance'][::-1],
        orientation='h',
        marker=dict(color=top_fi['importance'][::-1], colorscale='Emrld', showscale=True)
    ))
    fig.update_layout(
        title=f'Top {top_n} Features (Sparse MLP, λ={meta["best_lambda"]})',
        xaxis_title='Sum of |Weights|',
        template='plotly_white', height=max(400, top_n * 28)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Feature Explanation Table ---
    st.markdown("### 📋 Giải thích Top Features")
    explain_df = top_fi.head(10).copy()
    explain_df['Ý nghĩa'] = explain_df['feature'].map(lambda x: FEATURE_EXPLANATIONS.get(x, '—'))
    explain_df.columns = ['Feature', 'Importance', 'Ý nghĩa tài chính']
    explain_df.index = range(1, len(explain_df) + 1)
    st.dataframe(explain_df, use_container_width=True)
