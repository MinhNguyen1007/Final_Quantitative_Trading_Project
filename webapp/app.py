"""
Sparse Deep Learning for Alpha Interpretability — Web App
=========================================================
Chạy: streamlit run app.py
"""

import streamlit as st
from utils.model import load_models, load_metadata, load_extended_metadata
from components import dashboard, prediction, backtest, weights, interpretability

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sparse Alpha — VN30",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { font-family: 'Space Grotesk', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }

    .metric-card {
        background: white; border: 1px solid #e8ecf1; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card .value {
        font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .label {
        font-size: 0.85rem; color: #6b7280; margin-top: 0.3rem;
        text-transform: uppercase; letter-spacing: 1px;
    }

    .green { color: #10b981; }
    .red { color: #ef4444; }
    .blue { color: #3b82f6; }
    .purple { color: #8b5cf6; }

    .pred-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac; border-radius: 16px; padding: 2rem; text-align: center;
    }
    .pred-box.negative {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-color: #fca5a5;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Sparse Alpha VN30")
    st.markdown("---")
    st.markdown("""
    **Sparse Deep Learning for Alpha Interpretability**

    Mô hình MLP + L1 Regularization cho dự đoán
    và giải thích tín hiệu alpha trên VN30.

    - 🔴 **Dense MLP** — Baseline
    - 🟢 **Sparse MLP** — L1 Regularization
    """)
    st.markdown("---")
    st.markdown("*IUH — Giao dịch Định lượng*")

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def init():
    meta = load_metadata()
    dense, sparse = load_models(meta['input_dim'])
    ext = load_extended_metadata()  # có thể là None nếu chưa chạy phần mở rộng
    return dense, sparse, meta, ext

try:
    dense_model, sparse_model, meta, ext_meta = init()
    loaded = True
except Exception as e:
    loaded = False
    st.error(f"⚠️ Không load được model. Chạy notebook trước để tạo file `.pth` và `.pkl`.\n\n{e}")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📈 Sparse Deep Learning for Alpha Interpretability</h1>
    <p>Học sâu thưa cho khả năng giải thích tín hiệu Alpha — Thị trường VN30</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
if loaded:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔮 Dự đoán",
        "🧪 Backtest",
        "🔬 Phân tích Weights",
        "🧠 Interpretability",
    ])

    with tab1:
        dashboard.render(meta)

    with tab2:
        prediction.render(dense_model, sparse_model, meta)

    with tab3:
        backtest.render(dense_model, sparse_model, meta)

    with tab4:
        weights.render(dense_model, sparse_model, meta)

    with tab5:
        interpretability.render(meta, ext_meta)
