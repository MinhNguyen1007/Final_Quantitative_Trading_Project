"""
Config - Cấu hình chung cho project.

Paths được auto-detect từ vị trí file này → portable, không cần chỉnh tay
khi clone về máy khác.
"""

from pathlib import Path

# Auto-detect project root (parent of webapp/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = str(_PROJECT_ROOT / "data")

# Danh sách 30 mã VN30
VN30_SYMBOLS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# Model params (phải khớp với sparse_alpha.ipynb)
HIDDEN_DIMS = [256, 128, 64, 32]
DROPOUT = 0.1
FORWARD_DAYS = 5

# Artifact paths (auto-detect, không cần chỉnh tay)
DENSE_MODEL_PATH  = str(_PROJECT_ROOT / "dense_model.pth")
SPARSE_MODEL_PATH = str(_PROJECT_ROOT / "sparse_model.pth")
METADATA_PATH     = str(_PROJECT_ROOT / "model_metadata.pkl")

# Legacy LSTM paths — không dùng cho topic hiện tại, giữ tương thích test.ipynb cũ
DENSE_LSTM_PATH        = str(_PROJECT_ROOT / "dense_lstm.pth")
SPARSE_LSTM_PATH       = str(_PROJECT_ROOT / "sparse_lstm.pth")
EXTENDED_METADATA_PATH = str(_PROJECT_ROOT / "model_metadata_extended.pkl")

# LSTM hyperparams (legacy)
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.1
LSTM_WINDOW = 20
