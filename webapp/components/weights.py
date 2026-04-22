"""
Weights - Tab phân tích Weight Heatmap và phân phối
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render(dense_model, sparse_model, meta):
    """Render tab Phân tích Weights."""
    st.markdown("### 🔬 Phân tích Weight Heatmap")

    col1, col2 = st.columns(2)

    for col, weights, title, model in [
        (col1, meta['dense_w'], 'Dense MLP', dense_model),
        (col2, meta['sparse_w'], f'Sparse MLP (λ={meta["best_lambda"]})', sparse_model),
    ]:
        with col:
            fig = px.imshow(
                np.abs(weights), color_continuous_scale='YlOrRd',
                labels=dict(x="Input Features", y="Hidden Neurons", color="|Weight|"),
                aspect='auto'
            )
            fig.update_layout(title=f'{title} — |Weights| Layer 1',
                            template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)

    # --- Weight Distribution ---
    st.markdown("### 📊 Phân phối Weights")

    col1, col2 = st.columns(2)

    for col, weights, title, color in [
        (col1, meta['dense_w'], 'Dense MLP', '#ef4444'),
        (col2, meta['sparse_w'], f'Sparse MLP (λ={meta["best_lambda"]})', '#10b981'),
    ]:
        with col:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=weights.flatten(), nbinsx=100,
                marker_color=color, opacity=0.75
            ))
            fig.update_layout(title=f'{title} — Phân phối Weights',
                            xaxis_title='Weight Value', yaxis_title='Count',
                            template='plotly_white', height=350)
            st.plotly_chart(fig, use_container_width=True)

    # --- Sparsity Stats ---
    st.markdown("### 📋 Thống kê Sparsity")

    feature_cols = meta['feature_cols']
    sparse_fi = meta['sparse_fi']
    dense_fi = meta['dense_fi']

    total_params = sum(p.numel() for p in dense_model.parameters())

    stats = pd.DataFrame({
        'Thống kê': [
            'Tổng tham số',
            'Weights ≈ 0 (< 1e-4)',
            'Sparsity Ratio',
            'Features quan trọng (importance ≥ 0.01)',
            'Features bị loại (importance < 0.01)',
        ],
        'Dense MLP': [
            f"{total_params:,}",
            f"{int(total_params * dense_model.get_sparsity_ratio()):,}",
            f"{dense_model.get_sparsity_ratio():.1%}",
            f"{len(dense_fi[dense_fi['importance'] >= 0.01])}/{len(feature_cols)}",
            f"{len(dense_fi[dense_fi['importance'] < 0.01])}/{len(feature_cols)}",
        ],
        'Sparse MLP': [
            f"{total_params:,}",
            f"{int(total_params * sparse_model.get_sparsity_ratio()):,}",
            f"{sparse_model.get_sparsity_ratio():.1%}",
            f"{len(sparse_fi[sparse_fi['importance'] >= 0.01])}/{len(feature_cols)}",
            f"{len(sparse_fi[sparse_fi['importance'] < 0.01])}/{len(feature_cols)}",
        ]
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)
