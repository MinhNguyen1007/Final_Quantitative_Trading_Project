"""
Model - Định nghĩa MLP và các hàm load/predict
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
from config import HIDDEN_DIMS, DROPOUT, DENSE_MODEL_PATH, SPARSE_MODEL_PATH, METADATA_PATH


class MLP(nn.Module):
    """
    Multi-Layer Perceptron: Input → 128 → 64 → 32 → 1
    """
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
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
        """Tỉ lệ weights ≈ 0."""
        total = sum(p.numel() for p in self.parameters())
        near_zero = sum((p.abs() < threshold).sum().item() for p in self.parameters())
        return near_zero / total if total > 0 else 0


def load_metadata():
    """Load scaler, feature_cols, kết quả đánh giá, ..."""
    with open(METADATA_PATH, 'rb') as f:
        return pickle.load(f)


def load_models(input_dim):
    """Load cả Dense và Sparse model."""
    device = torch.device('cpu')

    dense = MLP(input_dim).to(device)
    dense.load_state_dict(torch.load(DENSE_MODEL_PATH, map_location=device, weights_only=True))
    dense.eval()

    sparse = MLP(input_dim).to(device)
    sparse.load_state_dict(torch.load(SPARSE_MODEL_PATH, map_location=device, weights_only=True))
    sparse.eval()

    return dense, sparse


def predict(model, X_scaled):
    """
    Dự đoán từ features đã scaled.
    X_scaled: numpy array shape (n, num_features)
    Return: numpy array shape (n,)
    """
    X_tensor = torch.FloatTensor(X_scaled)
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).numpy().flatten()
    return pred


def get_feature_importance(model, feature_names):
    """Feature importance = tổng |weights| từ input layer."""
    for module in model.network:
        if isinstance(module, nn.Linear):
            weights = module.weight.data.cpu().numpy()
            break
    importance = np.sum(np.abs(weights), axis=0)
    import pandas as pd
    fi = pd.DataFrame({'feature': feature_names, 'importance': importance})
    return fi.sort_values('importance', ascending=False).reset_index(drop=True), weights
