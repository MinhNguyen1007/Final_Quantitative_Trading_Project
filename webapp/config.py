"""
Config - Cấu hình chung cho project
"""

# ⚠️ THAY ĐỔI ĐƯỜNG DẪN NÀY cho phù hợp với máy bạn
DATA_DIR = r"D:/S2_Year4/Quantitative Trading/Project/data"

# Danh sách 30 mã VN30
VN30_SYMBOLS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# Model params
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.1
FORWARD_DAYS = 5

# File paths (tương đối với thư mục chạy app)
DENSE_MODEL_PATH = r"D:\S2_Year4\Quantitative Trading\Project\dense_model.pth"
SPARSE_MODEL_PATH = r"D:\S2_Year4\Quantitative Trading\Project\sparse_model.pth"
METADATA_PATH = r"D:\S2_Year4\Quantitative Trading\Project\model_metadata.pkl"
