# ============================================================================
# ISOLATED MODEL TRAINING
# ============================================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Any
import math
import numpy as np
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.task import (
    load_and_partition_data,
    create_features_and_labels,
    get_global_feature_space,
    get_model,
    train,
    test,
    MimicDataset,
    TOP_ICD_CODES,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class LogisticRegressionNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegressionNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class MLPNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super(MLPNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

class TabNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_d: int = 256, n_steps: int = 6):
        super(TabNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_steps = n_steps
        
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_dim, n_d * 4),
            nn.BatchNorm1d(n_d * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_d * 4, n_d * 2),
            nn.BatchNorm1d(n_d * 2),
            nn.ReLU(),
        )
        
        self.attentive_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_d * 2, n_d),
                nn.BatchNorm1d(n_d),
                nn.ReLU(),
                nn.Dropout(0.2),
            ) for _ in range(n_steps)
        ])
        
        self.feature_selectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_d, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.Sigmoid(),
            ) for _ in range(n_steps)
        ])
        
        self.classifier = nn.Linear(n_d, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        aggregated = torch.zeros(batch_size, self.n_d).to(x.device)
        prior_scales = torch.ones(batch_size, self.input_dim).to(x.device)
        
        for step in range(self.n_steps):
            feature_mask = self.feature_selectors[step](aggregated) * prior_scales
            masked_features = x * feature_mask
            
            transformed = self.feature_transformer(masked_features)
            
            decision = self.attentive_transformers[step](transformed)
            
            aggregated = aggregated + decision
            
            prior_scales = prior_scales * (1 - feature_mask)
        
        return self.classifier(aggregated)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 12, expert_dim: int = 512):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.BatchNorm1d(expert_dim),
                nn.Dropout(0.3),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU(),
                nn.BatchNorm1d(expert_dim),
                nn.Dropout(0.3),
                nn.Linear(expert_dim, expert_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(expert_dim // 2),
                nn.Dropout(0.3),
                nn.Linear(expert_dim // 2, output_dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        gate_weights = self.gate(x)  # [batch_size, num_experts]
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # [batch_size, output_dim]
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch_size, output_dim, num_experts]
        
        gate_weights = gate_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
        output = torch.sum(expert_outputs * gate_weights, dim=2)  # [batch_size, output_dim]
        
        return output

class TransformerNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 512, nhead: int = 16, num_layers: int = 6):
        super(TransformerNet, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.token_size = 8
        
        self.input_projection = nn.Sequential(
            nn.Linear(self.token_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        max_tokens = (input_dim + self.token_size - 1) // self.token_size
        self.pos_encoding = nn.Parameter(torch.randn(1, max_tokens, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.15,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x):
        batch_size, input_dim = x.size()
        
        num_tokens = (input_dim + self.token_size - 1) // self.token_size
        
        if input_dim % self.token_size != 0:
            padding_size = self.token_size - (input_dim % self.token_size)
            x = F.pad(x, (0, padding_size), value=0)
            input_dim = x.size(1)
        
        x = x.view(batch_size, num_tokens, self.token_size)
        
        x = self.input_projection(x)
        
        pos_encoding = self.pos_encoding[:, :num_tokens, :]
        x = x + pos_encoding
        
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        
        return self.classifier(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, num_blocks: int = 4):
        super(ResidualMLP, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ) for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = F.relu(x)
        
        return self.classifier(x)

class LSTMNet(nn.Module):
    """LSTM network with attention for sequential processing of tabular data."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super(LSTMNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.seq_len = min(32, max(1, input_dim // 4))
        if self.seq_len == 0:
            self.seq_len = 1
        self.feature_dim = max(1, input_dim // self.seq_len)
        
        if input_dim % self.seq_len != 0:
            self.input_projection = nn.Linear(input_dim, self.seq_len * self.feature_dim)
            actual_input_dim = self.seq_len * self.feature_dim
        else:
            self.input_projection = None
            actual_input_dim = input_dim
            self.feature_dim = actual_input_dim // self.seq_len
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        lstm_out, _ = self.lstm(x)
        
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        return self.classifier(attended_output)

class ULCDNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 64):
        super(ULCDNet, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
        
        self.consensus_weights = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.Softmax(dim=1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        
        self.distillation_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        
        consensus = self.consensus_weights(latent)
        weighted_latent = latent * consensus
        
        main_output = self.decoder(weighted_latent)
        
        distill_output = self.distillation_head(latent)
        
        final_output = 0.7 * main_output + 0.3 * distill_output
        
        return final_output

# ============================================================================
# SKLEARN WRAPPER
# ============================================================================

class SklearnWrapper(nn.Module):
    """Wrapper to make sklearn models compatible with PyTorch training loop."""
    def __init__(self, sklearn_model, input_dim: int, output_dim: int):
        super().__init__()
        self.sklearn_model = sklearn_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_fitted = False
        
    def forward(self, x):
        if not self.is_fitted:
            batch_size = x.shape[0]
            return torch.randn(batch_size, self.output_dim, device=x.device)
        
        x_np = x.detach().cpu().numpy()
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            x_np = self.scaler.transform(x_np)
        
        probabilities = self.sklearn_model.predict_proba(x_np)
        
        if probabilities.shape[0] != x_np.shape[0]:
            print(f"      ERROR: XGBoost returned wrong batch dimension: expected {x_np.shape[0]}, got {probabilities.shape[0]}")
            if probabilities.shape[1] == x_np.shape[0]:
                probabilities = probabilities.T
                print(f"      Fixed by transposing to: {probabilities.shape}")
            else:
                print(f"      Fallback: creating uniform probabilities")
                n_classes = len(self.sklearn_model.classes_) if hasattr(self.sklearn_model, 'classes_') else self.output_dim
                probabilities = np.ones((x_np.shape[0], n_classes)) / n_classes
        
        if probabilities.shape[1] != self.output_dim:
            learned_classes = self.sklearn_model.classes_
            
            # Create full probability matrix 
            full_probs = np.zeros((probabilities.shape[0], self.output_dim))
            
            # Fill in the actual probabilities for classes the model learned
            for i, cls in enumerate(learned_classes):
                if cls < self.output_dim:  # Ensure class index is valid
                    full_probs[:, cls] = probabilities[:, i]
            
            # For unseen classes, assign very small uniform probability
            # But don't renormalize - keep the original learned probabilities intact
            unseen_mask = np.ones(self.output_dim, dtype=bool)
            unseen_mask[learned_classes[learned_classes < self.output_dim]] = False
            
            # Assign minimal probability to unseen classes so they sum to ~0.001
            n_unseen = np.sum(unseen_mask)
            if n_unseen > 0:
                unseen_prob_per_class = 0.001 / n_unseen
                full_probs[:, unseen_mask] = unseen_prob_per_class
                
                # Reduce learned class probabilities slightly to maintain sum=1
                learned_mask = ~unseen_mask
                full_probs[:, learned_mask] = full_probs[:, learned_mask] * 0.999
            
            probabilities = full_probs
            
        return torch.tensor(probabilities, dtype=torch.float32, device=x.device)
    
    def fit(self, X, y):
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else y
        
        unique_classes = len(np.unique(y_np))
        if unique_classes == 1 and hasattr(self.sklearn_model, 'n_classes_'):
            print(f"    WARNING: Single-class partition detected. Sklearn models may have limited performance.")
        
        self.sklearn_model.fit(X_np, y_np)
        self.is_fitted = True
        
    def predict(self, X):
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_np = self.scaler.transform(X_np)
            
        return self.sklearn_model.predict(X_np)
        
    def predict_proba(self, X):
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_np = self.scaler.transform(X_np)
            
        return self.sklearn_model.predict_proba(X_np)

# ============================================================================
# MODEL FACTORY
# ============================================================================
def get_isolated_model(model_name: str, input_dim: int, output_dim: int):
    if model_name == "logistic":
        return LogisticRegressionNet(input_dim, output_dim)
    elif model_name == "random_forest":
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        return SklearnWrapper(rf, input_dim, output_dim)
    elif model_name == "extra_trees":
        et_model = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,              # Slightly deeper for medical complexity
            min_samples_split=5,       # Conservative splitting
            min_samples_leaf=2,        # Small leaf size for medical precision
            max_features='sqrt',       # Feature sampling for regularization
            random_state=42,           # Reproducible results
            n_jobs=-1,                 # Use all cores
            bootstrap=True             # Bootstrap sampling like Random Forest
        )
        return SklearnWrapper(et_model, input_dim, output_dim)
    elif model_name == "xgboost":
        if not HAS_XGBOOST:
            raise ValueError("XGBoost not available. Install with: pip install xgboost")
        # XGBoost with medical-tuned parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,          # Balance between performance and speed
            max_depth=6,               # Prevent overfitting
            learning_rate=0.1,         # Conservative learning rate
            subsample=0.8,             # Row sampling for robustness
            colsample_bytree=0.8,      # Feature sampling
            random_state=42,
            n_jobs=-1,
            objective='multi:softprob',  # Multi-class with probabilities
            num_class=output_dim       # Explicitly set number of classes
        )
        return SklearnWrapper(xgb_model, input_dim, output_dim)
    elif model_name == "gradient_boosting":
        # Gradient Boosting Classifier - robust alternative to XGBoost
        gb_model = GradientBoostingClassifier(
            n_estimators=100,          # Balance between performance and speed
            max_depth=6,               # Prevent overfitting on medical data
            learning_rate=0.1,         # Conservative learning rate
            subsample=0.8,             # Row sampling for regularization
            max_features='sqrt',       # Feature sampling for regularization
            random_state=42,           # Reproducible results
            verbose=0                  # Quiet training
        )
        return SklearnWrapper(gb_model, input_dim, output_dim)
    elif model_name == "lightgbm":
        # LightGBM - excellent for medical tabular data
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,          # More trees for complex medical patterns
                max_depth=8,               # Moderate depth for medical complexity
                learning_rate=0.05,        # Conservative learning rate
                num_leaves=64,             # Balanced leaf count
                min_child_samples=10,      # Prevent overfitting on small medical subgroups
                subsample=0.8,             # Row sampling for robustness
                colsample_bytree=0.8,      # Feature sampling
                reg_alpha=0.1,             # L1 regularization
                reg_lambda=0.1,            # L2 regularization
                random_state=42,
                n_jobs=-1,
                verbosity=-1               # Quiet training
            )
            return SklearnWrapper(lgb_model, input_dim, output_dim)
        except ImportError:
            print("Warning: LightGBM not available. Install with: pip install lightgbm")
            # Fallback to Random Forest
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            return SklearnWrapper(rf, input_dim, output_dim)
    elif model_name == "adaboost":
        # AdaBoost with enhanced parameters
        from sklearn.tree import DecisionTreeClassifier
        ada_model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
            n_estimators=100,          # Balanced number of estimators
            learning_rate=0.8,         # Moderate learning rate
            random_state=42
        )
        return SklearnWrapper(ada_model, input_dim, output_dim)
    elif model_name == "mixture_of_experts":
        return MixtureOfExperts(input_dim, output_dim, num_experts=6, expert_dim=256)
    elif model_name == "mlp":
        return MLPNet(input_dim, output_dim)
    elif model_name == "lstm":
        return LSTMNet(input_dim, output_dim)
    elif model_name == "ulcd":
        return ULCDNet(input_dim, output_dim)
    elif model_name == "tabnet":
        return TabNet(input_dim, output_dim)
    elif model_name == "transformer":
        return TransformerNet(input_dim, output_dim)
    elif model_name == "residual_mlp":
        return ResidualMLP(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ============================================================================
# CONFIGURATION
# ============================================================================

MIMIC_DATA_DIR = "mimic-iv-3.1"
MIN_PARTITION_SIZE = 500
BATCH_SIZE = 128
LOCAL_EPOCHS = 40
LEARNING_RATE = 0.0005
TOP_K_CODES = 100
ACTUAL_NUM_CLASSES = None

AVAILABLE_MODELS: List[str] = [
    "logistic",
    "mlp",
    "lstm",
    "ulcd",
    "tabnet",
    "mixture_of_experts",
    "transformer",
    "residual_mlp",
    "random_forest",
    "gradient_boosting",
    "extra_trees",
    "lightgbm",
    "adaboost"
]

MODELS_TO_TEST: List[str] = [
    "logistic",
    "random_forest",
    "mlp",
    "lstm",
    "mixture_of_experts"
]

# Advanced options
PARTITION_SCHEME = "heterogeneous"  # Use same partitioning as federated experiments
USE_ICD_FEATURES = False  # Match federated learning feature set (no severity features)
SAVE_PLOTS = True  # Save loss plots to files

# ==============================================================================
# FEATURE PREPROCESSING FOR MEDICAL DATA
# ==============================================================================

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def preprocess_medical_features(X_train, X_test, method='robust'):
    """
    Preprocess medical features with proper scaling and outlier handling.
    
    Args:
        X_train: Training features (numpy array or tensor)
        X_test: Test features (numpy array or tensor) 
        method: Scaling method ('robust', 'standard', or 'none')
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    # Convert to numpy if tensors
    X_train_np = X_train.detach().cpu().numpy() if torch.is_tensor(X_train) else X_train
    X_test_np = X_test.detach().cpu().numpy() if torch.is_tensor(X_test) else X_test
    
    if method == 'none':
        return X_train, X_test, None
    
    print(f"    Preprocessing features with {method} scaling...")
    print(f"    Original range: [{X_train_np.min():.2f}, {X_train_np.max():.2f}]")
    
    # Handle extreme outliers (cap at 99.9th percentile)
    for col in range(X_train_np.shape[1]):
        col_data = X_train_np[:, col]
        if col_data.std() > 0:  # Only process non-constant columns
            # Cap extreme outliers at 99.9th percentile
            upper_limit = np.percentile(col_data, 99.9)
            lower_limit = np.percentile(col_data, 0.1)
            
            # Apply caps to both train and test
            X_train_np[:, col] = np.clip(col_data, lower_limit, upper_limit)
            X_test_np[:, col] = np.clip(X_test_np[:, col], lower_limit, upper_limit)
    
    # Apply scaling
    if method == 'robust':
        # RobustScaler is better for medical data with outliers
        scaler = RobustScaler()
    elif method == 'standard':
        # StandardScaler for normal distribution assumption
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    
    print(f"    Scaled range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    print(f"    Features with zero variance: {np.sum(X_train_scaled.std(axis=0) == 0)}")
    
    # Convert back to original type
    if torch.is_tensor(X_train):
        X_train_scaled = torch.tensor(X_train_scaled, dtype=X_train.dtype, device=X_train.device)
        X_test_scaled = torch.tensor(X_test_scaled, dtype=X_test.dtype, device=X_test.device)
    
    return X_train_scaled, X_test_scaled, scaler

# ==============================================================================
# ADVANCED LOSS FUNCTIONS AND TRAINING COMPONENTS
# ==============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss for better generalization on large label spaces."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=1)
        
        nll_loss = F.nll_loss(log_probs, target, reduction='none')
        smooth_loss = -log_probs.mean(dim=1)
        
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ==============================================================================
# TRAINING FUNCTIONS WITH LOSS TRACKING
# ==============================================================================

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_sklearn_model(net, trainloader, testloader, epochs: int, learning_rate: float, device: torch.device):
    import time
    start_time = time.time()
    
    # Collect all training data
    all_features = []
    all_labels = []
    
    for features, labels in trainloader:
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    X_train = np.concatenate(all_features)
    y_train = np.concatenate(all_labels)
    
    # Collect all test data
    all_test_features = []
    all_test_labels = []
    
    for features, labels in testloader:
        all_test_features.append(features.cpu().numpy())
        all_test_labels.append(labels.cpu().numpy())
    
    X_test = np.concatenate(all_test_features)
    y_test = np.concatenate(all_test_labels)
    
    # Apply feature preprocessing for sklearn models
    X_train_scaled, X_test_scaled, scaler = preprocess_medical_features(X_train, X_test, method='robust')
    
    # Fit the sklearn model on preprocessed data
    net.fit(X_train_scaled, y_train)
    
    # Get predictions on preprocessed test data
    predictions = net.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    # Store scaler for later use in enhanced_test
    net.scaler = scaler
    
    training_time = time.time() - start_time
    return None, None, training_time

def train_with_loss_and_accuracy_tracking(net, trainloader, testloader, epochs: int, learning_rate: float, device: torch.device):
    start_time = time.time()  # Start timing
    net.to(device)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    epoch_losses = []
    epoch_accuracies = []
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    # Debug: Check label ranges before training
    all_labels = []
    for _, labels in trainloader:
        all_labels.extend(labels.numpy())
    
    min_label = min(all_labels) if all_labels else 0
    max_label = max(all_labels) if all_labels else 0
    unique_labels = len(set(all_labels))
    
    print(f"    Label range: {min_label} to {max_label}, unique labels: {unique_labels}")
    
    # Check for single-class partitions (common in specialized medical domains)
    if unique_labels == 1:
        if TOP_K_CODES is not None and max_label == TOP_K_CODES:
            print(f"    SPECIALIST PARTITION: All samples are rare ICD codes (not in top-{TOP_K_CODES})")
            print(f"    This represents a medical specialty that would benefit greatly from federated collaboration!")
        else:
            print(f"    SINGLE-CLASS PARTITION: All samples have label {max_label}")
    elif unique_labels <= 3:
        print(f"    LOW-DIVERSITY PARTITION: Only {unique_labels} different ICD code classes")
        print(f"    This partition would benefit from federated learning with other specialties")
    
    # Validate and fix labels if needed
    if hasattr(net, 'linear') and hasattr(net.linear, 'out_features'):
        expected_classes = net.linear.out_features
    elif hasattr(net, 'layers') and isinstance(net.layers, nn.Sequential):
        expected_classes = net.layers[-1].out_features
    elif hasattr(net, 'experts') and hasattr(net.experts, '__len__') and len(net.experts) > 0:
        expected_classes = net.experts[0][-1].out_features
    elif hasattr(net, 'decoder') and isinstance(net.decoder, nn.Sequential):
        expected_classes = net.decoder[-1].out_features
    elif hasattr(net, 'classifier') and hasattr(net.classifier, 'out_features'):
        expected_classes = net.classifier.out_features
    elif hasattr(net, 'classifier') and isinstance(net.classifier, nn.Sequential):
        expected_classes = net.classifier[-1].out_features
    elif hasattr(net, 'sklearn_model'):
        expected_classes = len(getattr(net.sklearn_model, 'classes_', []))
        if expected_classes == 0:
            expected_classes = ACTUAL_NUM_CLASSES
    else:
        expected_classes = ACTUAL_NUM_CLASSES
    if max_label >= expected_classes or min_label < 0:
        print(f"    WARNING: Invalid labels detected! Max label: {max_label}, Expected classes: {expected_classes}")
        print(f"    Labels will be clipped during training to valid range [0, {expected_classes-1}]")
    
    print(f"    Training for {epochs} epochs on {device}...")
    print(f"    Model output dim: {expected_classes}, Dataset classes: {unique_labels}")
    
    # Enable anomaly detection for gradient issues (only for debugging)
    # torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        num_batches = 0
        
        try:
            for batch_idx, (features, labels) in enumerate(trainloader):
                features, labels = features.to(device), labels.to(device)
                
                # Validate input tensors
                if torch.isnan(features).any() or torch.isinf(features).any():
                    print(f"Warning: Invalid features detected in batch {batch_idx} (NaN/Inf), skipping batch")
                    continue
                    
                # Clip invalid labels to valid range
                labels = torch.clamp(labels, 0, expected_classes - 1)
                
                optimizer.zero_grad()
                
                if epoch < 3:
                    current_lr = learning_rate * (epoch + 1) / 3
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                
                outputs = net(features)
                
                # Validate output dimensions
                if outputs.shape[1] != expected_classes:
                    print(f"Error: Model output dim {outputs.shape[1]} != expected classes {expected_classes}")
                    raise ValueError(f"Model output dimension mismatch")
                
                # Validate outputs for NaN/Inf
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Warning: Invalid outputs detected in batch {batch_idx} (NaN/Inf), skipping batch")
                    continue
                    
                loss = criterion(outputs, labels)
                
                # Check for NaN/Inf values before backprop
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected in batch {batch_idx} (NaN/Inf), skipping batch")
                    continue
                    
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            print(f"Features shape: {features.shape if 'features' in locals() else 'Unknown'}")
            print(f"Labels shape: {labels.shape if 'labels' in locals() else 'Unknown'}")
            print(f"Labels range: {labels.min().item() if 'labels' in locals() else 'Unknown'} - {labels.max().item() if 'labels' in locals() else 'Unknown'}")
            print(f"Expected classes: {expected_classes}")
            raise
        
        epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(epoch_loss)
        
        # Evaluation phase - calculate accuracy on test set
        net.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in testloader:
                features, labels = features.to(device), labels.to(device)
                
                # Clip invalid labels for evaluation too
                labels = torch.clamp(labels, 0, expected_classes - 1)
                
                outputs = net(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_accuracy = correct / total if total > 0 else 0.0
        epoch_accuracies.append(epoch_accuracy)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        scheduler.step(epoch_loss)
            
        if patience_counter >= patience:
            break
        
        # Print progress every few epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f} ({epoch_accuracy*100:.2f}%)")
    
    training_time = time.time() - start_time
    return epoch_losses, epoch_accuracies, training_time

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def enhanced_test(net, testloader, device, icd_frequency_tiers=None):
    """Enhanced test function with additional medical-specific metrics."""
    import torch.nn as nn
    
    criterion = nn.CrossEntropyLoss()
    net.to(device)
    net.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    if len(testloader) == 0:
        print("Warning: Empty test loader")
        return 0.0, create_empty_metrics()
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            
            # Clip invalid labels
            if hasattr(net, 'linear') and hasattr(net.linear, 'out_features'):
                expected_classes = net.linear.out_features
            elif hasattr(net, 'layers') and isinstance(net.layers, nn.Sequential):
                expected_classes = net.layers[-1].out_features
            elif hasattr(net, 'experts') and hasattr(net.experts, '__len__') and len(net.experts) > 0:
                expected_classes = net.experts[0][-1].out_features
            elif hasattr(net, 'decoder') and isinstance(net.decoder, nn.Sequential):
                expected_classes = net.decoder[-1].out_features
            elif hasattr(net, 'classifier') and hasattr(net.classifier, 'out_features'):
                expected_classes = net.classifier.out_features
            elif hasattr(net, 'classifier') and isinstance(net.classifier, nn.Sequential):
                expected_classes = net.classifier[-1].out_features
            elif hasattr(net, 'sklearn_model'):
                expected_classes = len(getattr(net.sklearn_model, 'classes_', []))
                if expected_classes == 0:
                    expected_classes = ACTUAL_NUM_CLASSES
            else:
                expected_classes = ACTUAL_NUM_CLASSES
            labels = torch.clamp(labels, 0, expected_classes - 1)
            
            outputs = net(features)
            
            # Handle sklearn models differently (they output probabilities, not logits)
            if isinstance(net, SklearnWrapper):
                # For sklearn models, outputs are already probabilities
                probabilities = outputs
                # Compute loss using negative log likelihood since we have probabilities
                loss = -torch.sum(torch.log(probabilities[range(len(labels)), labels] + 1e-8)) / len(labels)
                # Get predictions from probabilities (not expanded outputs)
                _, preds = torch.max(probabilities, 1)
            else:
                # For neural networks, outputs are logits
                loss = criterion(outputs, labels)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
            total_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    
    avg_loss = total_loss / len(testloader)
    
    # Standard metrics
    accuracy = accuracy_score(all_labels, all_predictions) 
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Enhanced metrics
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    top5_accuracy = calculate_top_k_accuracy(all_labels, all_probabilities, k=5)
    top20_accuracy = calculate_top_k_accuracy(all_labels, all_probabilities, k=20)
    top50_accuracy = calculate_top_k_accuracy(all_labels, all_probabilities, k=50)
    
    # Higher Top-K values for large label spaces
    num_classes = all_probabilities.shape[1] if len(all_probabilities.shape) > 1 else len(set(all_labels))
    top500_accuracy = calculate_top_k_accuracy(all_labels, all_probabilities, k=500) if num_classes >= 500 else 0.0
    top1000_accuracy = calculate_top_k_accuracy(all_labels, all_probabilities, k=1000) if num_classes >= 1000 else 0.0
    
    # Medical domain analysis
    rare_common_metrics = None
    if icd_frequency_tiers is not None:
        rare_common_metrics = calculate_rare_common_performance(
            all_labels, all_predictions, icd_frequency_tiers
        )
    
    return avg_loss, {
        "accuracy": accuracy,
        "f1": f1_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "top5_accuracy": top5_accuracy,
        "top20_accuracy": top20_accuracy,
        "top50_accuracy": top50_accuracy,
        "top500_accuracy": top500_accuracy,
        "top1000_accuracy": top1000_accuracy,
        "rare_common_performance": rare_common_metrics
    }

def calculate_top_k_accuracy(labels, probabilities, k):
    """Calculate top-k accuracy for ICD code prediction.
    
    When TOP_K_CODES is limited: Only consider the top-K most frequent ICD codes, excluding 'other' class.
    When TOP_K_CODES is unlimited: Consider all available classes.
    """
    # Handle unlimited case (all ICD codes)
    if TOP_K_CODES is None or TOP_K_CODES in ['none', 'None']:
        max_k = probabilities.shape[1]
        if k > max_k:
            k = max_k
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = sum(1 for i, label in enumerate(labels) if label in top_k_preds[i])
        return correct / len(labels)
    
    # Handle limited TOP_K case
    if k > TOP_K_CODES:
        k = TOP_K_CODES
    
    # Get top-k predictions, but only consider valid ICD code indices (0 to TOP_K_CODES-1)
    # Exclude the 'other' class (index TOP_K_CODES) from top-k consideration
    valid_probabilities = probabilities[:, :TOP_K_CODES]  # Only first TOP_K_CODES classes
    
    if valid_probabilities.shape[1] < k:
        k = valid_probabilities.shape[1]
    
    top_k_preds = np.argsort(valid_probabilities, axis=1)[:, -k:]
    correct = 0
    
    for i, label in enumerate(labels):
        # Only count as correct if:
        # 1. The true label is in the valid ICD range (0 to TOP_K_CODES-1), AND  
        # 2. The true label is among the top-k predictions
        if label < TOP_K_CODES and label in top_k_preds[i]:
            correct += 1
    
    return correct / len(labels)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def create_icd_frequency_tiers():
    from collections import Counter
    
    try:
        # Handle unlimited case (all ICD codes)
        if TOP_K_CODES is None or TOP_K_CODES in ['none', 'None']:
            total_codes = ACTUAL_NUM_CLASSES if ACTUAL_NUM_CLASSES is not None else 9554
            # Split into tiers based on position in frequency-sorted list
            tier_size = total_codes // 4  # Roughly 25% each for top/middle/bottom
            
            frequency_tiers = {
                "common": list(range(0, tier_size)),                    # Top 25% most frequent
                "moderate": list(range(tier_size, 2 * tier_size)),      # Next 25%  
                "rare": list(range(2 * tier_size, total_codes)),        # Bottom 50%
            }
            
            print(f"ICD Frequency Tiers Created (ALL ICD codes):")
            print(f"  Common (top 25%): {len(frequency_tiers['common'])} codes (indices 0-{tier_size-1})")
            print(f"  Moderate (mid 25%): {len(frequency_tiers['moderate'])} codes (indices {tier_size}-{2*tier_size-1})")
            print(f"  Rare (bottom 50%): {len(frequency_tiers['rare'])} codes (indices {2*tier_size}-{total_codes-1})")
            
            return frequency_tiers
        
        # Handle limited TOP_K case
        total_codes = TOP_K_CODES
        
        if total_codes == 0:
            return None
            
        # Split into tiers based on position in frequency-sorted list
        tier_size = total_codes // 4
        
        frequency_tiers = {
            "common": list(range(0, tier_size)),                    # Top 25% most frequent
            "moderate": list(range(tier_size, 2 * tier_size)),      # Next 25%  
            "rare": list(range(2 * tier_size, total_codes)),        # Bottom 50%
            "other": [TOP_K_CODES]
        }
        
        print(f"ICD Frequency Tiers Created:")
        print(f"  Common (top 25%): {len(frequency_tiers['common'])} codes (indices 0-{tier_size-1})")
        print(f"  Moderate (mid 25%): {len(frequency_tiers['moderate'])} codes (indices {tier_size}-{2*tier_size-1})")
        print(f"  Rare (bottom 50%): {len(frequency_tiers['rare'])} codes (indices {2*tier_size}-{total_codes-1})")
        print(f"  Other (outside top-{total_codes}): 1 code (index {TOP_K_CODES})")
        
        return frequency_tiers
        
    except Exception as e:
        print(f"Error creating frequency tiers: {e}")
        return None

def calculate_rare_common_performance(labels, predictions, frequency_tiers):
    """Calculate performance metrics for rare vs common diseases."""
    if frequency_tiers is None:
        return None
    
    results = {}
    total_classified_samples = 0
    
    for tier_name, icd_indices in frequency_tiers.items():
        # Find samples belonging to this tier
        tier_mask = np.isin(labels, icd_indices)
        matching_count = np.sum(tier_mask)
        total_classified_samples += matching_count
        
        if matching_count == 0:
            results[tier_name] = {"accuracy": 0.0, "f1": 0.0, "count": 0}
            continue
            
        tier_labels = labels[tier_mask]
        tier_predictions = predictions[tier_mask]
        
        tier_accuracy = accuracy_score(tier_labels, tier_predictions)
        tier_f1 = f1_score(tier_labels, tier_predictions, average='weighted', zero_division=0)
        
        results[tier_name] = {
            "accuracy": tier_accuracy,
            "f1": tier_f1,
            "count": len(tier_labels)
        }
    
    # Debug check: ensure no double counting
    total_input_samples = len(labels)
    if total_classified_samples != total_input_samples:
        print(f"    WARNING: Sample count mismatch! Input: {total_input_samples}, Classified: {total_classified_samples}")
    
    return results

def create_empty_metrics():
    """Create empty metrics dict for error cases."""
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "top5_accuracy": 0.0,
        "top20_accuracy": 0.0,
        "top50_accuracy": 0.0,
        "top500_accuracy": 0.0,
        "top1000_accuracy": 0.0,
        "rare_common_performance": None
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_training_metrics(all_results, partition_name):
    """Plot enhanced training metrics with additional medical-specific performance graphs."""
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    models = list(all_results.keys())
    
    # Plot 1: Training Loss (Neural Networks Only)
    ax1 = fig.add_subplot(gs[0, 0])
    neural_models = {name: result for name, result in all_results.items() 
                     if name not in ['random_forest', 'lightgbm', 'gradient_boosting', 'extra_trees', 'adaboost']}
    
    for i, (model_name, result) in enumerate(neural_models.items()):
        if 'loss_history' in result:
            epochs = range(1, len(result['loss_history']) + 1)
            ax1.plot(epochs, result['loss_history'], 
                    color=colors[i % len(colors)], 
                    linewidth=2, marker='o', markersize=3,
                    label=f'{model_name.title()}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'Training Loss - {partition_name.title()}')
    ax1.set_ylim(0, 25.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    # Dynamic x-axis based on actual epochs (neural networks only)
    if neural_models:
        max_epochs = max(len(result.get('loss_history', [])) for result in neural_models.values())
        ax1.set_xticks(range(5, max_epochs + 1, 5))
    
    # Plot 2: Accuracy Progression (Neural Networks Only)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (model_name, result) in enumerate(neural_models.items()):
        if 'accuracy_history' in result:
            epochs = range(1, len(result['accuracy_history']) + 1)
            ax2.plot(epochs, [acc * 100 for acc in result['accuracy_history']], 
                    color=colors[i % len(colors)], 
                    linewidth=2, marker='s', markersize=3,
                    label=f'{model_name.title()}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title(f'Test Accuracy - {partition_name.title()}')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    if neural_models:
        ax2.set_xticks(range(5, max_epochs + 1, 5))
    
    # Plot 3: Top-K Accuracy Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if all('metrics' in result for result in all_results.values()):
        top5_acc = [result['metrics']['top5_accuracy']*100 for result in all_results.values()]
        top20_acc = [result['metrics']['top20_accuracy']*100 for result in all_results.values()]
        top50_acc = [result['metrics']['top50_accuracy']*100 for result in all_results.values()]
        top500_acc = [result['metrics']['top500_accuracy']*100 for result in all_results.values()]
        top1000_acc = [result['metrics']['top1000_accuracy']*100 for result in all_results.values()]
        
        x = np.arange(len(models))
        width = 0.15  # Narrower bars to fit all
        ax3.bar(x - 2*width, top5_acc, width, label='Top-5', alpha=0.8, color='skyblue')
        ax3.bar(x - width, top20_acc, width, label='Top-20', alpha=0.8, color='lightgreen')
        ax3.bar(x, top50_acc, width, label='Top-50', alpha=0.8, color='lightcoral')
        ax3.bar(x + width, top500_acc, width, label='Top-500', alpha=0.8, color='orange')
        ax3.bar(x + 2*width, top1000_acc, width, label='Top-1000', alpha=0.8, color='purple')
        
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title(f'Top-K Accuracy - {partition_name.title()}')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
        ax3.set_ylim(0, 100)  # No padding for bar charts
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: F1 Score Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    if all('metrics' in result for result in all_results.values()):
        f1_macro = [result['metrics']['f1_macro'] for result in all_results.values()]
        f1_weighted = [result['metrics']['f1_weighted'] for result in all_results.values()]
        
        x = np.arange(len(models))
        width = 0.35
        ax4.bar(x - width/2, f1_macro, width, label='F1 Macro', alpha=0.8, color='gold')
        ax4.bar(x + width/2, f1_weighted, width, label='F1 Weighted', alpha=0.8, color='orange')
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('F1 Score')
        ax4.set_title(f'F1 Score Comparison - {partition_name.title()}')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
        ax4.set_ylim(0, 1.0)  # Standardized F1 scale: 0-1.0
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Medical Domain Performance (if available)
    ax5 = fig.add_subplot(gs[1, 1])
    has_domain_data = any(result['metrics'].get('rare_common_performance') is not None 
                         for result in all_results.values() if 'metrics' in result)
    
    if has_domain_data:
        tier_names = ['Common', 'Moderate', 'Rare', 'Other']
        tier_data = {tier: [] for tier in tier_names}
        
        for model_name in models:
            if 'metrics' in all_results[model_name]:
                rare_common = all_results[model_name]['metrics']['rare_common_performance']
                if rare_common:
                    for tier in tier_names:
                        tier_key = tier.lower()
                        if tier_key in rare_common:
                            tier_data[tier].append(rare_common[tier_key]['accuracy']*100)
                        else:
                            tier_data[tier].append(0)
                else:
                    for tier in tier_names:
                        tier_data[tier].append(0)
        
        x = np.arange(len(models))
        width = 0.2
        for i, (tier, accuracies) in enumerate(tier_data.items()):
            ax5.bar(x + i*width - 1.5*width, accuracies, width, label=tier, alpha=0.8)
        
        ax5.set_xlabel('Model')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title(f'Disease Frequency Performance - {partition_name.title()}')
        ax5.set_xticks(x)
        ax5.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
        ax5.set_ylim(0, 100)  # Standardized percentage scale: 0-100%
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Medical Domain\\nData Not Available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=10)
        ax5.set_title(f'Disease Frequency Performance - {partition_name.title()}')
    
    # Plot 6: Training Time Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    if all('training_time' in result for result in all_results.values()):
        training_times = [result['training_time'] for result in all_results.values()]
        
        x = np.arange(len(models))
        bars = ax6.bar(x, training_times, alpha=0.8, 
                      color=[colors[i % len(colors)] for i in range(len(models))])
        
        ax6.set_xlabel('Model')
        ax6.set_ylabel('Training Time (seconds)')
        ax6.set_title(f'Training Time Comparison - {partition_name.title()}')
        ax6.set_xticks(x)
        ax6.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars, training_times)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 7: Final Performance Summary
    ax7 = fig.add_subplot(gs[2, 0])
    if all('metrics' in result for result in all_results.values()):
        accuracies = [result['metrics']['accuracy'] for result in all_results.values()]
        f1_scores = [result['metrics']['f1_weighted'] for result in all_results.values()]
        
        x = np.arange(len(models))
        width = 0.35
        bars1 = ax7.bar(x - width/2, [acc*100 for acc in accuracies], width, 
                       label='Accuracy', alpha=0.8, color='skyblue')
        bars2 = ax7.bar(x + width/2, [f1*100 for f1 in f1_scores], width, 
                       label='F1 Score', alpha=0.8, color='lightcoral')
        
        ax7.set_xlabel('Model')
        ax7.set_ylabel('Performance (%)')
        ax7.set_title(f'Final Performance Summary - {partition_name.title()}')
        ax7.set_xticks(x)
        ax7.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
        ax7.set_ylim(0, 100)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    plt.suptitle(f'{partition_name.title()} Partition', 
                 fontsize=16, y=0.95)
    
    if SAVE_PLOTS:
        os.makedirs('isolated_plots', exist_ok=True)
        
        filename_tag = ""
        if all('metrics' in result for result in all_results.values()):
            sample_result = next(iter(all_results.values()))
            if 'metrics' in sample_result and sample_result['metrics'].get('rare_common_performance'):
                rare_common = sample_result['metrics']['rare_common_performance']
                if 'other' in rare_common:
                    other_count = rare_common['other']['count']
                    total_samples = sum(tier['count'] for tier in rare_common.values())
                    other_percentage = other_count / total_samples if total_samples > 0 else 0
                    
                    if other_percentage >= 0.9:
                        filename_tag = "_SPECIALIST"
                    elif other_percentage >= 0.7:
                        filename_tag = "_RARE_HEAVY"
        
        filename = f'isolated_plots/enhanced_metrics_{partition_name.replace(" ", "_")}{filename_tag}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Enhanced metrics plot saved: {filename}")
    
    plt.close()  # Close figure to prevent memory issues

# ============================================================================
# SUMMARY FUNCTIONS
# ============================================================================
def create_summary_comparison(all_partition_results):
    """Create a comprehensive comparison chart across all partitions and models."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    
    # Prepare data for plotting
    partitions = list(all_partition_results.keys())
    models = MODELS_TO_TEST
    num_models = len(models)
    
    # 1. Final Accuracy Comparison
    accuracy_data = []
    for model in models:
        model_accuracies = []
        for partition in partitions:
            if partition in all_partition_results and model in all_partition_results[partition]:
                acc = all_partition_results[partition][model]['metrics']['accuracy']
                model_accuracies.append(acc)
            else:
                model_accuracies.append(0)
        accuracy_data.append(model_accuracies)
    
    x = np.arange(len(partitions))
    width = 0.8 / num_models  # Adjust width based on number of models
    
    for i, (model, accuracies) in enumerate(zip(models, accuracy_data)):
        offset = (i - (num_models - 1) / 2) * width
        ax1.bar(x + offset, accuracies, width, label=f'{model.title()}', alpha=0.8)
    
    ax1.set_xlabel('Medical Specialty (Partition)')
    ax1.set_ylabel('Final Accuracy')
    ax1.set_title('Final Accuracy by Partition and Model Architecture')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace('_', ' ').title() for p in partitions], rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score Comparison
    f1_data = []
    for model in models:
        model_f1s = []
        for partition in partitions:
            if partition in all_partition_results and model in all_partition_results[partition]:
                f1 = all_partition_results[partition][model]['metrics']['f1']
                model_f1s.append(f1)
            else:
                model_f1s.append(0)
        f1_data.append(model_f1s)
    
    for i, (model, f1s) in enumerate(zip(models, f1_data)):
        offset = (i - (num_models - 1) / 2) * width
        ax2.bar(x + offset, f1s, width, label=f'{model.title()}', alpha=0.8)
    
    ax2.set_xlabel('Medical Specialty (Partition)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score by Partition and Model Architecture')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace('_', ' ').title() for p in partitions], rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Loss Final Values
    loss_data = []
    for model in models:
        model_losses = []
        for partition in partitions:
            if partition in all_partition_results and model in all_partition_results[partition]:
                if 'loss_history' in all_partition_results[partition][model]:
                    final_loss = all_partition_results[partition][model]['loss_history'][-1]
                    model_losses.append(final_loss)
                else:
                    model_losses.append(0)
            else:
                model_losses.append(0)
        loss_data.append(model_losses)
    
    for i, (model, losses) in enumerate(zip(models, loss_data)):
        offset = (i - (num_models - 1) / 2) * width
        ax3.bar(x + offset, losses, width, label=f'{model.title()}', alpha=0.8)
    
    ax3.set_xlabel('Medical Specialty (Partition)')
    ax3.set_ylabel('Final Training Loss')
    ax3.set_title('Final Training Loss by Partition and Model Architecture')
    ax3.set_xticks(x)
    ax3.set_xticklabels([p.replace('_', ' ').title() for p in partitions], rotation=45, ha='right', fontsize=8)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Sample Size per Partition
    sample_sizes = []
    for partition in partitions:
        if partition in all_partition_results:
            # Get sample size from any model result
            for model in models:
                if model in all_partition_results[partition] and 'sample_size' in all_partition_results[partition][model]:
                    sample_sizes.append(all_partition_results[partition][model]['sample_size'])
                    break
            else:
                sample_sizes.append(0)
        else:
            sample_sizes.append(0)
    
    ax4.bar(range(len(partitions)), sample_sizes, alpha=0.8, color='skyblue')
    ax4.set_xlabel('Medical Specialty (Partition)')
    ax4.set_ylabel('Number of Training Samples')
    ax4.set_title('Training Data Size by Medical Specialty')
    ax4.set_xticks(range(len(partitions)))
    ax4.set_xticklabels([p.replace('_', ' ').title() for p in partitions], rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        os.makedirs('isolated_plots', exist_ok=True)
        filename = 'isolated_plots/comprehensive_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Summary comparison plot saved: {filename}")
    
    plt.close()  # Close figure to prevent memory issues

# ============================================================================
# GLOBAL EVALUATION
# ============================================================================
def evaluate_on_global_dataset(partitions, global_feature_space, icd_frequency_tiers):
    """Train each model on the complete combined dataset for ultimate baseline comparison."""
    print(f"Combining all partition data for global evaluation...")
    
    # Combine all partition data into a single DataFrame
    all_dataframes = []
    for partition_name, partition_data in partitions.items():
        all_dataframes.append(partition_data)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Combined dataset: {len(combined_df)} samples from {len(partitions)} partitions")
    
    # Create features and labels for combined data
    features, labels = create_features_and_labels(
        combined_df, "global_dataset", global_feature_space, 
        include_icd_features=USE_ICD_FEATURES
    )
    
    print(f"Global feature dimensions: {features.shape}")
    print(f"Global unique labels: {len(np.unique(labels))}")
    
    # Fix label mapping for global dataset
    print(f"Original global label range: {labels.min()} to {labels.max()}")
    
    # Remap labels to valid range
    if TOP_K_CODES is None:
        # When using all ICD codes, no remapping needed
        valid_labels = np.array(labels, dtype=np.int64)  # Convert to numpy array
        print("Using all ICD codes - no label remapping needed for global dataset")
    else:
        # Remap labels to valid range [0, TOP_K_CODES] for limited case
        valid_labels = []
        for label in labels:
            if label < TOP_K_CODES:
                valid_labels.append(label)  # Keep valid top-K labels
            else:
                valid_labels.append(TOP_K_CODES)
        
        valid_labels = np.array(valid_labels, dtype=np.int64)
    print(f"Remapped global label range: {valid_labels.min()} to {valid_labels.max()}")
    
    # Split into train/test - check if stratification is possible
    from collections import Counter
    global_label_counts = Counter(valid_labels)
    global_min_count = min(global_label_counts.values())
    
    # Only stratify if all classes have at least 2 samples
    global_use_stratification = len(np.unique(valid_labels)) > 1 and global_min_count >= 2
    
    if global_use_stratification:
        print(f"Global dataset: Using stratified split (min class count: {global_min_count})")
        global_stratify_param = valid_labels
    else:
        print(f"Global dataset: Using random split (some classes have only {global_min_count} sample(s))")
        global_stratify_param = None
    
    X_train, X_test, y_train, y_test = train_test_split(
        features.values.astype(np.float32),
        valid_labels,
        test_size=0.2,
        random_state=42,
        stratify=global_stratify_param
    )

    train_dataset = MimicDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = MimicDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[1]
    # Use actual number of unique classes (like in partition training)
    if TOP_K_CODES is not None and TOP_K_CODES != 'none':
        output_dim = TOP_K_CODES + 1  # +1 for "other" class  
    else:
        output_dim = 9554  # Use full ICD code space
    
    global_results = {}
    
    # Train each model on the complete dataset
    for model_name in MODELS_TO_TEST:
        print(f"\\n  >> Training {model_name.upper()} on Global Dataset:")
        
        # Create fresh model
        net = get_isolated_model(model_name, input_dim, output_dim=output_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        
        # Use consistent learning rate for all models
        adjusted_lr = LEARNING_RATE
        
        # Train with tracking (use appropriate function based on model type)
        if isinstance(net, SklearnWrapper):
            _, _, training_time = train_sklearn_model(
                net=net,
                trainloader=trainloader,
                testloader=testloader,
                epochs=LOCAL_EPOCHS,
                learning_rate=adjusted_lr,
                device=device
            )
            loss_history = None
            accuracy_history = None
        else:
            loss_history, accuracy_history, training_time = train_with_loss_and_accuracy_tracking(
                net=net,
                trainloader=trainloader,
                testloader=testloader,
                epochs=LOCAL_EPOCHS,
                learning_rate=adjusted_lr,
                device=device
            )
        
        # Evaluate with enhanced metrics
        test_loss, metrics = enhanced_test(net, testloader, device=device, icd_frequency_tiers=icd_frequency_tiers)
        
        print(f"    Final Test Loss: {test_loss:.4f}")
        print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"    Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
        print(f"    Top-20 Accuracy: {metrics['top20_accuracy']:.4f} ({metrics['top20_accuracy']*100:.2f}%)")
        print(f"    Top-50 Accuracy: {metrics['top50_accuracy']:.4f} ({metrics['top50_accuracy']*100:.2f}%)")
        if metrics['top500_accuracy'] > 0:  # Only show if applicable
            print(f"    Top-500 Accuracy: {metrics['top500_accuracy']:.4f} ({metrics['top500_accuracy']*100:.2f}%)")
        if metrics['top1000_accuracy'] > 0:  # Only show if applicable  
            print(f"    Top-1000 Accuracy: {metrics['top1000_accuracy']:.4f} ({metrics['top1000_accuracy']*100:.2f}%)")
        print(f"    F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"    F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        
        # Print rare vs common performance if available
        if metrics['rare_common_performance']:
            print(f"    Medical Domain Performance:")
            for tier, perf in metrics['rare_common_performance'].items():
                print(f"      {tier.title()}: Acc={perf['accuracy']:.3f}, F1={perf['f1']:.3f} (n={perf['count']})")

        # Store results
        result_dict = {
            'metrics': metrics,
            'test_loss': test_loss,
            'training_time': training_time,
            'sample_size': len(X_train),
            'input_dim': input_dim,
            'output_dim': output_dim
        }
        
        if loss_history is not None:
            result_dict['loss_history'] = loss_history
        if accuracy_history is not None:
            result_dict['accuracy_history'] = accuracy_history
            
        global_results[model_name] = result_dict
    
    # Generate training metrics visualization for global dataset
    print(f"\n  >> Generating training metrics visualization for Global Dataset...")
    plot_training_metrics(global_results, "Global_Dataset")
    
    return global_results

def plot_global_comparison(global_results):
    """Create visualization comparing all models on the global dataset."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(global_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 1. Training Loss Progression
    for i, (model_name, results) in enumerate(global_results.items()):
        if 'loss_history' in results:
            epochs = range(1, len(results['loss_history']) + 1)
            ax1.plot(epochs, results['loss_history'], 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    marker='o', 
                    markersize=4,
                    label=f'{model_name.title()} Model')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Global Dataset - Training Loss Progression', fontsize=14)
    ax1.set_ylim(0, 25.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Accuracy Progression  
    for i, (model_name, results) in enumerate(global_results.items()):
        if 'accuracy_history' in results:
            epochs = range(1, len(results['accuracy_history']) + 1)
            ax2.plot(epochs, [acc * 100 for acc in results['accuracy_history']], 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    marker='s', 
                    markersize=4,
                    label=f'{model_name.title()} Model')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Global Dataset - Accuracy Progression', fontsize=14)
    ax2.set_ylim(0, 105)  # Padded for visibility
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Final Performance Comparison - Bar Chart
    accuracies = [results['metrics']['accuracy'] for results in global_results.values()]
    f1_scores = [results['metrics']['f1'] for results in global_results.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Model Architecture')
    ax3.set_ylabel('Performance Score')
    ax3.set_title('Global Dataset - Final Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.title() for m in models], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 4. Loss Reduction Comparison
    loss_reductions = []
    for results in global_results.values():
        if 'loss_history' in results and len(results['loss_history']) > 1:
            reduction = ((results['loss_history'][0] - results['loss_history'][-1]) / results['loss_history'][0] * 100)
            loss_reductions.append(reduction)
        else:
            loss_reductions.append(0)
    
    bars4 = ax4.bar(models, loss_reductions, alpha=0.8, color='lightgreen')
    ax4.set_xlabel('Model Architecture')
    ax4.set_ylabel('Loss Reduction (%)')
    ax4.set_title('Global Dataset - Training Loss Reduction')
    ax4.set_xticklabels([m.title() for m in models], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        os.makedirs('isolated_plots', exist_ok=True)
        filename = 'isolated_plots/global_dataset_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Global dataset comparison plot saved: {filename}")
    
    plt.close()

def plot_enhanced_global_comparison(global_results):
    """Create enhanced visualization with medical-specific metrics for global dataset (matches partition format)."""
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)
    
    models = list(global_results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 1. Training Loss Progression (Neural models only)
    ax1 = fig.add_subplot(gs[0, 0])
    sklearn_models = ['random_forest', 'lightgbm', 'xgboost']
    for i, (model_name, results) in enumerate(global_results.items()):
        if 'loss_history' in results and model_name.lower() not in sklearn_models:
            epochs = range(1, len(results['loss_history']) + 1)
            ax1.plot(epochs, results['loss_history'], 
                    color=colors[i % len(colors)], 
                    linewidth=2, marker='o', markersize=3,
                    label=f'{model_name.title()}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Progression')
    ax1.set_ylim(0, 25.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # 2. Accuracy Progression (Neural models only)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (model_name, results) in enumerate(global_results.items()):
        if 'accuracy_history' in results and model_name.lower() not in sklearn_models:
            epochs = range(1, len(results['accuracy_history']) + 1)
            ax2.plot(epochs, [acc * 100 for acc in results['accuracy_history']], 
                    color=colors[i % len(colors)], 
                    linewidth=2, marker='s', markersize=3,
                    label=f'{model_name.title()}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Accuracy Progression')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # 3. Top-K Accuracy Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    top5_acc = [results['metrics']['top5_accuracy'] for results in global_results.values()]
    top20_acc = [results['metrics']['top20_accuracy'] for results in global_results.values()]
    top50_acc = [results['metrics']['top50_accuracy'] for results in global_results.values()]
    top500_acc = [results['metrics']['top500_accuracy'] for results in global_results.values()]
    top1000_acc = [results['metrics']['top1000_accuracy'] for results in global_results.values()]
    
    x = np.arange(len(models))
    width = 0.15  # Narrower bars to fit all
    ax3.bar(x - 2*width, [acc*100 for acc in top5_acc], width, label='Top-5', alpha=0.8, color='skyblue')
    ax3.bar(x - width, [acc*100 for acc in top20_acc], width, label='Top-20', alpha=0.8, color='lightgreen')
    ax3.bar(x, [acc*100 for acc in top50_acc], width, label='Top-50', alpha=0.8, color='lightcoral')
    ax3.bar(x + width, [acc*100 for acc in top500_acc], width, label='Top-500', alpha=0.8, color='orange')
    ax3.bar(x + 2*width, [acc*100 for acc in top1000_acc], width, label='Top-1000', alpha=0.8, color='purple')
    
    ax3.set_xlabel('Model Architecture')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Top-K Accuracy Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
    ax3.set_ylim(0, 100)  # No padding for bar charts
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score Comparison (Macro vs Weighted)
    ax4 = fig.add_subplot(gs[1, 0])
    f1_macro = [results['metrics']['f1_macro'] for results in global_results.values()]
    f1_weighted = [results['metrics']['f1_weighted'] for results in global_results.values()]
    
    x = np.arange(len(models))
    width = 0.35
    ax4.bar(x - width/2, f1_macro, width, label='F1 Macro', alpha=0.8, color='gold')
    ax4.bar(x + width/2, f1_weighted, width, label='F1 Weighted', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Model Architecture')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score: Macro vs Weighted')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
    ax4.set_ylim(0, 1.0)  # Standardized F1 scale: 0-1.0
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Rare vs Common Disease Performance
    ax5 = fig.add_subplot(gs[1, 1])
    # Check if rare/common data is available
    has_rare_common = any(results['metrics']['rare_common_performance'] is not None 
                         for results in global_results.values())
    
    if has_rare_common:
        tier_names = ['Common', 'Moderate', 'Rare', 'Other']
        model_names = list(models)
        tier_data = {tier: [] for tier in tier_names}
        
        for model_name in model_names:
            rare_common = global_results[model_name]['metrics']['rare_common_performance']
            if rare_common:
                for tier in tier_names:
                    tier_key = tier.lower()
                    if tier_key in rare_common:
                        tier_data[tier].append(rare_common[tier_key]['accuracy'])
                    else:
                        tier_data[tier].append(0)
            else:
                for tier in tier_names:
                    tier_data[tier].append(0)
        
        x = np.arange(len(model_names))
        width = 0.2
        for i, (tier, accuracies) in enumerate(tier_data.items()):
            ax5.bar(x + i*width - 1.5*width, [acc*100 for acc in accuracies], 
                   width, label=tier, alpha=0.8)
        
        ax5.set_xlabel('Model Architecture')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title('Performance by Disease Frequency')
        ax5.set_xticks(x)
        ax5.set_xticklabels([m.title() for m in model_names], rotation=45, ha='right', fontsize=8)
        ax5.set_ylim(0, 100)  # Standardized percentage scale: 0-100%
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Rare vs Common\\nData Not Available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Performance by Disease Frequency')
    
    # 6. Training Time Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    if all('training_time' in result for result in global_results.values()):
        training_times = [result['training_time'] for result in global_results.values()]
        
        x = np.arange(len(models))
        bars = ax6.bar(x, training_times, alpha=0.8, 
                      color=[colors[i % len(colors)] for i in range(len(models))])
        
        ax6.set_xlabel('Model')
        ax6.set_ylabel('Training Time (seconds)')
        ax6.set_title('Training Time Comparison - Global Dataset')
        ax6.set_xticks(x)
        ax6.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars, training_times)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # Removed numerical headers for cleaner visualization
    
    # 7. Final Performance Summary
    ax7 = fig.add_subplot(gs[2, 0])
    if all('metrics' in result for result in global_results.values()):
        accuracies = [result['metrics']['accuracy'] for result in global_results.values()]
        f1_scores = [result['metrics']['f1_weighted'] for result in global_results.values()]
        
        x = np.arange(len(models))
        width = 0.35
        bars1 = ax7.bar(x - width/2, [acc*100 for acc in accuracies], width, 
                       label='Accuracy', alpha=0.8, color='skyblue')
        bars2 = ax7.bar(x + width/2, [f1*100 for f1 in f1_scores], width, 
                       label='F1 Score', alpha=0.8, color='lightcoral')
        
        ax7.set_xlabel('Model')
        ax7.set_ylabel('Performance (%)')
        ax7.set_title('Final Performance Summary - Global Dataset')
        ax7.set_xticks(x)
        ax7.set_xticklabels([m.title() for m in models], rotation=45, ha='right', fontsize=8)
        ax7.set_ylim(0, 100)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Global Dataset', 
                 fontsize=16, y=0.95)
    
    if SAVE_PLOTS:
        os.makedirs('isolated_plots', exist_ok=True)
        filename = 'isolated_plots/enhanced_global_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Enhanced global comparison plot saved: {filename}")
    
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main function implementing Goal #4: Establish Performance Baseline
    
    Trains isolated models on each medical specialty partition to establish 
    crucial baseline metrics for measuring federated learning improvements.
    """
    print("="*80)
    print("ISOLATED MODEL TRAINING - PERFORMANCE BASELINE (Goal #4)")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Model Architectures: {MODELS_TO_TEST}")
    print(f"  - Epochs: {LOCAL_EPOCHS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Top-K ICD Codes: {'ALL' if TOP_K_CODES in ['none', 'None', None] else TOP_K_CODES}")
    print(f"  - Include Medical Features: {USE_ICD_FEATURES}")
    print("="*80)

    start_time = time.time()

    print(f"\nLoading data with {PARTITION_SCHEME} partitioning...")
    
    actual_top_k = None if TOP_K_CODES in ['none', 'None', None] else TOP_K_CODES
    if actual_top_k is None:
        print("Using ALL ICD codes (no limit)")
    
    partitions = load_and_partition_data(
        data_dir=MIMIC_DATA_DIR, 
        min_partition_size=MIN_PARTITION_SIZE,
        partition_scheme=PARTITION_SCHEME,
        top_k_codes=actual_top_k
    )
    categorical_features = [
        'admission_type', 'admission_location', 'insurance', 
        'language', 'marital_status', 'race', 'gender'
    ]
    
    global_feature_space = {}
    for feature in categorical_features:
        all_values = set()
        for chapter, df in partitions.items():
            if feature in df.columns:
                values = df[feature].fillna('Unknown').unique()
                all_values.update(values)
        
        global_feature_space[feature] = sorted(list(all_values))
        print(f"Global feature '{feature}': {len(global_feature_space[feature])} unique values")
    
    all_labels = []
    for partition_name, partition_data in partitions.items():
        _, labels = create_features_and_labels(
            partition_data, 
            partition_name, 
            global_feature_space, 
            include_icd_features=USE_ICD_FEATURES
        )
        all_labels.extend(labels.tolist())
    
    global ACTUAL_NUM_CLASSES
    unique_labels = set(all_labels)
    ACTUAL_NUM_CLASSES = max(unique_labels) + 1  # Use max label + 1 for proper indexing
    print(f"Detected {len(unique_labels)} unique classes in the data")
    print(f"Label range: {min(unique_labels)} to {max(unique_labels)}")
    print(f"Setting ACTUAL_NUM_CLASSES to {ACTUAL_NUM_CLASSES} (max_label + 1)")
    
    print(f"Found {len(partitions)} valid partitions")
    
    # Create ICD frequency tiers for medical domain analysis
    icd_frequency_tiers = create_icd_frequency_tiers()
    
    all_partition_results = {}

    for partition_idx, (partition_name, partition_data) in enumerate(partitions.items()):
        print(f"\n{'-'*60}")
        print(f"PARTITION {partition_idx+1}/{len(partitions)}: {partition_name.upper()}")
        print(f"Samples: {len(partition_data)}")
        print(f"{'-'*60}")
        
        all_partition_results[partition_name] = {}

        features, labels = create_features_and_labels(
            partition_data, partition_name, global_feature_space, 
            include_icd_features=USE_ICD_FEATURES
        )
        
        print(f"  Feature dimensions: {features.shape}")
        print(f"  Unique labels: {len(np.unique(labels))}")
        
        print(f"    Original label range: {labels.min()} to {labels.max()}")
        
        if TOP_K_CODES is None:
            valid_labels = np.array(labels, dtype=np.int64)  # Convert to numpy array
            print(f"    Using all ICD codes - no label remapping needed")
        else:
            valid_labels = []
            for label in labels:
                if label < TOP_K_CODES:
                    valid_labels.append(label)
                else:
                    valid_labels.append(TOP_K_CODES)
            
            valid_labels = np.array(valid_labels, dtype=np.int64)
        print(f"    Remapped label range: {valid_labels.min()} to {valid_labels.max()}")
        
        from collections import Counter
        label_counts = Counter(valid_labels)
        min_count = min(label_counts.values())
        
        use_stratification = len(np.unique(valid_labels)) > 1 and min_count >= 2
        
        if use_stratification:
            print(f"    Using stratified split (min class count: {min_count})")
            stratify_param = valid_labels
        else:
            print(f"    Using random split (some classes have only {min_count} sample(s))")
            stratify_param = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            features.values.astype(np.float32),
            valid_labels,
            test_size=0.2,
            random_state=42,
            stratify=stratify_param
        )

        train_dataset = MimicDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_dataset = MimicDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        input_dim = X_train.shape[1]
        
        if TOP_K_CODES is not None and TOP_K_CODES != 'none':
            output_dim = TOP_K_CODES + 1
        else:
            output_dim = ACTUAL_NUM_CLASSES

        # 3. Train and evaluate each model architecture
        for model_name in MODELS_TO_TEST:
            print(f"\n  >> Training {model_name.upper()} Model:")
            
            # Create fresh model for each run
            net = get_isolated_model(model_name, input_dim, output_dim=output_dim)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net.to(device)
            
            # Train with loss and accuracy tracking (use appropriate function based on model type)
            if isinstance(net, SklearnWrapper):
                _, _, training_time = train_sklearn_model(
                    net=net,
                    trainloader=trainloader,
                    testloader=testloader,
                    epochs=LOCAL_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    device=device
                )
                loss_history = None
                accuracy_history = None
            else:
                loss_history, accuracy_history, training_time = train_with_loss_and_accuracy_tracking(
                    net=net,
                    trainloader=trainloader,
                    testloader=testloader,
                    epochs=LOCAL_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    device=device
                )
            
            # Evaluate the trained model with enhanced metrics
            test_loss, metrics = enhanced_test(net, testloader, device=device, icd_frequency_tiers=icd_frequency_tiers)
            
            print(f"    Final Test Loss: {test_loss:.4f}")
            print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"    Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
            print(f"    Top-20 Accuracy: {metrics['top20_accuracy']:.4f} ({metrics['top20_accuracy']*100:.2f}%)")
            print(f"    Top-50 Accuracy: {metrics['top50_accuracy']:.4f} ({metrics['top50_accuracy']*100:.2f}%)")
            if metrics['top500_accuracy'] > 0:  # Only show if applicable
                print(f"    Top-500 Accuracy: {metrics['top500_accuracy']:.4f} ({metrics['top500_accuracy']*100:.2f}%)")
            if metrics['top1000_accuracy'] > 0:  # Only show if applicable  
                print(f"    Top-1000 Accuracy: {metrics['top1000_accuracy']:.4f} ({metrics['top1000_accuracy']*100:.2f}%)")
            print(f"    F1 Weighted: {metrics['f1_weighted']:.4f}")
            print(f"    F1 Macro: {metrics['f1_macro']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            
            # Print rare vs common performance if available
            if metrics['rare_common_performance']:
                print(f"    Medical Domain Performance:")
                for tier, perf in metrics['rare_common_performance'].items():
                    print(f"      {tier.title()}: Acc={perf['accuracy']:.3f}, F1={perf['f1']:.3f} (n={perf['count']})")

            # Store comprehensive results
            result_dict = {
                'metrics': metrics,
                'test_loss': test_loss,
                'training_time': training_time,
                'sample_size': len(X_train),
                'input_dim': input_dim,
                'output_dim': output_dim
            }
            
            if loss_history is not None:
                result_dict['loss_history'] = loss_history
            if accuracy_history is not None:
                result_dict['accuracy_history'] = accuracy_history
                
            all_partition_results[partition_name][model_name] = result_dict

        # Plot training metrics (loss and accuracy) for this partition
        print(f"\n  >> Generating training metrics visualization for {partition_name}...")
        plot_training_metrics(all_partition_results[partition_name], partition_name)

    # 4. Generate comprehensive comparison and final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("ISOLATED TRAINING COMPLETE - BASELINE ESTABLISHED")
    print(f"{'='*80}")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Partitions processed: {len(all_partition_results)}")
    print(f"Models per partition: {len(MODELS_TO_TEST)}")
    print(f"Total model runs: {len(all_partition_results) * len(MODELS_TO_TEST)}")

    # Print detailed results summary
    print(f"\n{'-'*80}")
    print("DETAILED PERFORMANCE SUMMARY")
    print(f"{'-'*80}")
    
    for partition_name, partition_results in all_partition_results.items():
        print(f"\n{partition_name.upper()} Partition:")
        for model_name, results in partition_results.items():
            metrics = results['metrics']
            
            if 'loss_history' in results and len(results['loss_history']) > 1:
                loss_reduction = ((results['loss_history'][0] - results['loss_history'][-1]) / results['loss_history'][0] * 100)
            else:
                loss_reduction = 0
                
            if 'accuracy_history' in results and len(results['accuracy_history']) > 1 and results['accuracy_history'][0] > 0:
                accuracy_improvement = ((results['accuracy_history'][-1] - results['accuracy_history'][0]) / results['accuracy_history'][0] * 100)
            else:
                accuracy_improvement = 0
            print(f"  {model_name.title()} Model:")
            print(f"    Final Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"    F1 Score: {metrics['f1']:.4f}")
            print(f"    Training Samples: {results['sample_size']:,}")
            print(f"    Loss Reduction: {loss_reduction:.1f}%")
            print(f"    Accuracy Improvement: {accuracy_improvement:+.1f}%")

        # Generate comprehensive comparison plots
        print(f"\n{'-'*80}")
        print("GENERATING COMPREHENSIVE COMPARISON CHARTS...")
        print(f"{'-'*80}")
        create_summary_comparison(all_partition_results)

    print(f"\n{'='*80}")
    print("PARTITION ANALYSIS COMPLETE!")
    print("Next Steps:")
    print("  1. Run 'python global_evaluation_only.py' for centralized baseline comparison")
    print("  2. Compare partition results with federated learning performance")
    print("  3. Identify which medical specialties benefit most from collaboration")
    print("  4. Analyze generalization capabilities across different partitions")
    print(f"{'='*80}")
    
    return all_partition_results

if __name__ == "__main__":
    main()