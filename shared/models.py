import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================================
# DATASETS
# ============================================================================
# PyTorch dataset wrapper for MIMIC-IV features and labels
class MimicDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================================================
# ULCD MODELS
# ============================================================================
# LoRA Adapter Module for ULCD
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        self.down = nn.Linear(in_features, r, bias=False)
        self.up = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))

# ULCD Tabular Transformer Module
class TabularTransformer(nn.Module):
    def __init__(self, input_dim=20, d_model=64, num_heads=4, num_layers=2, use_lora=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # Add layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            norm_first=True  # Pre-norm for better stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_lora = use_lora
        if use_lora:
            self.lora = LoRALinear(d_model, d_model)

    def forward(self, x):
        # Clamp inputs to prevent extreme values
        x = torch.clamp(x, -10, 10)
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, d_model]
        x = self.layer_norm(x)               # Normalize after projection
        
        x = self.encoder(x).squeeze(1)       # [B, d_model]
        
        if self.use_lora:
            lora_out = self.lora(x)
            # Clamp LoRA output to prevent explosion
            lora_out = torch.clamp(lora_out, -5, 5)
            x = x + lora_out
            
        # Final clamp before return
        x = torch.clamp(x, -10, 10)
        return x

# ULCD Network
class ULCDNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 64):
        super(ULCDNet, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Enhanced tabular encoder with transformer architecture
        self.tabular_encoder = TabularTransformer(
            input_dim=input_dim, 
            d_model=latent_dim,
            num_heads=min(4, latent_dim // 16),  # Adaptive heads based on latent dim
            num_layers=2,
            use_lora=False  # TODO: Re-enable LoRA after FL issues resolved
        )
        
        # Consensus mechanism for latent space alignment
        self.consensus_weights = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.Softmax(dim=1)
        )
        
        # Task-specific prediction head
        self.task_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, output_dim)
        )
        
        # Distillation head for knowledge transfer
        self.distillation_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Encode to latent space
        latent = self.tabular_encoder(x)
        
        # Check latent for stability
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            print(f"Warning: NaN/Inf detected in latent, using fallback")
            latent = torch.zeros_like(latent)
        
        # Apply consensus weighting with stability checks
        consensus = self.consensus_weights(latent)
        consensus = torch.clamp(consensus, 1e-8, 1.0)  # Prevent extreme values
        weighted_latent = latent * consensus
        
        # Task-specific prediction
        main_output = self.task_head(weighted_latent)
        
        # Check outputs for stability
        if torch.isnan(main_output).any() or torch.isinf(main_output).any():
            main_output = torch.zeros_like(main_output)
            
        return main_output

# ============================================================================
# LSTM MODELS
# ============================================================================
class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super(LSTMNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Input projection to prepare for LSTM
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(0.3),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input and add sequence dimension
        x = self.input_proj(x)  # [batch_size, hidden_dim]
        x = x.unsqueeze(1)      # [batch_size, 1, hidden_dim] - single timestep
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output (squeeze sequence dimension)
        output = lstm_out.squeeze(1)  # [batch_size, hidden_dim * 2]
        
        # Final classification
        return self.classifier(output)

# ============================================================================
# MIXTURE OF EXPERTS MODELS
# ============================================================================
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4, expert_dim: int = 128):
        super(MixtureOfExperts, self).__init__()
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(expert_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(expert_dim, output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # Compute gating weights
        gate_weights = self.gate(x)  # [batch_size, num_experts]
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # [batch_size, output_dim]
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(2)  # [batch_size, num_experts, 1]
        output = torch.sum(gate_weights * expert_outputs, dim=1)  # [batch_size, output_dim]
        
        return output

# ============================================================================
# LOGISTIC REGRESSION MODELS
# ============================================================================
class LogisticRegressionNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegressionNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# ============================================================================
# MLP MODELS
# ============================================================================
class MLPNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [512, 256, 128]):
        super(MLPNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers dynamically
        layers = []
        current_dim = input_dim
        
        # Input normalization
        layers.append(nn.BatchNorm1d(input_dim))
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================================
# SKLEARN WRAPPER FOR FEDERATED LEARNING
# ============================================================================
class SklearnWrapper(nn.Module):
    def __init__(self, sklearn_model, input_dim: int, output_dim: int):
        super(SklearnWrapper, self).__init__()
        self.sklearn_model = sklearn_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler = StandardScaler()
        self.is_trained = False

    def forward(self, x):
        # Convert to numpy for sklearn
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        if not self.is_trained:
            # Return random predictions if not trained
            batch_size = x_np.shape[0]
            random_probs = np.random.rand(batch_size, self.output_dim)
            random_probs = random_probs / random_probs.sum(axis=1, keepdims=True)
            return torch.tensor(random_probs, dtype=torch.float32, device=x.device if isinstance(x, torch.Tensor) else 'cpu')
        
        # Scale features
        x_scaled = self.scaler.transform(x_np)
        
        # Get predictions
        if hasattr(self.sklearn_model, 'predict_proba'):
            probabilities = self.sklearn_model.predict_proba(x_scaled)
            
            # Handle dimension mismatch for sklearn models
            if probabilities.shape[1] != self.output_dim:
                # Get the classes the model learned
                learned_classes = self.sklearn_model.classes_
                
                # Create full probability matrix
                full_probs = np.zeros((probabilities.shape[0], self.output_dim))
                for i, class_idx in enumerate(learned_classes):
                    if class_idx < self.output_dim:
                        full_probs[:, class_idx] = probabilities[:, i]
                
                probabilities = full_probs
        else:
            # For models without predict_proba, use decision function or predictions
            predictions = self.sklearn_model.predict(x_scaled)
            probabilities = np.eye(self.output_dim)[predictions]
        
        return torch.tensor(probabilities, dtype=torch.float32, device=x.device if isinstance(x, torch.Tensor) else 'cpu')

    def fit_sklearn(self, X, y):
        """Fit the sklearn model"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
            
        # Fit scaler and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.sklearn_model.fit(X_scaled, y)
        self.is_trained = True

# ============================================================================
# NEURAL NETWORKS
# ============================================================================
# Basic neural network with attention mechanism
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feature_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.feature_norm(x)
        x1 = torch.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)
        
        x2 = torch.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        
        x3 = torch.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        
        x4 = torch.relu(self.bn4(self.fc4(x3 + x2)))
        attention_weights = self.attention(x4)
        x4_attended = x4 * attention_weights
        
        output = self.classifier(x4_attended)
        
        return output

# Advanced neural network with configurable layers
class AdvancedNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(AdvancedNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feature_norm = nn.BatchNorm1d(input_dim)
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dims[-1]),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = self.feature_norm(x)
        x = self.hidden_layers(x)
        
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        output = self.classifier(x_attended)
        return output

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
# Focal loss for addressing class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# MODEL FACTORY
# ============================================================================
# Factory function to create neural network models
def get_model(model_name: str, input_dim: int, output_dim: int, **kwargs):
    if model_name == "basic":
        return Net(input_dim, output_dim)
    elif model_name == "advanced":
        return AdvancedNet(input_dim, output_dim)
    elif model_name == "ulcd":
        latent_dim = kwargs.get('latent_dim', 64)
        return ULCDNet(input_dim, output_dim, latent_dim=latent_dim)
    elif model_name == "ulcd_multimodal":
        from .ulcd_components import MultimodalULCDClient
        latent_dim = kwargs.get('latent_dim', 64)
        tabular_dim = kwargs.get('tabular_dim', input_dim)
        return MultimodalULCDClient(tabular_dim=tabular_dim, latent_dim=latent_dim, task_out=output_dim)
    elif model_name == "lstm":
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 2)
        return LSTMNet(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "moe":
        num_experts = kwargs.get('num_experts', 4)
        expert_dim = kwargs.get('expert_dim', 128)
        return MixtureOfExperts(input_dim, output_dim, num_experts=num_experts, expert_dim=expert_dim)
    elif model_name == "mlp":
        hidden_dims = kwargs.get('hidden_dims', [512, 256, 128])
        return MLPNet(input_dim, output_dim, hidden_dims=hidden_dims)
    elif model_name == "logistic":
        return LogisticRegressionNet(input_dim, output_dim)
    elif model_name == "random_forest":
        sklearn_model = RandomForestClassifier(n_estimators=100, random_state=42)
        return SklearnWrapper(sklearn_model, input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ============================================================================
# MODEL UTILITIES
# ============================================================================
def is_sklearn_model(model):
    """Check if a model is a sklearn wrapper"""
    return isinstance(model, SklearnWrapper)

def get_model_type(model):
    """Get the type of model for special handling"""
    # Check for multimodal ULCD first (more specific)
    from .ulcd_components import MultimodalULCDClient
    if isinstance(model, MultimodalULCDClient):
        return "ulcd_multimodal"
    elif isinstance(model, ULCDNet):
        return "ulcd"
    elif isinstance(model, LSTMNet):
        return "lstm"
    elif isinstance(model, MixtureOfExperts):
        return "moe"
    elif isinstance(model, MLPNet):
        return "mlp"
    elif isinstance(model, LogisticRegressionNet):
        return "logistic"
    elif isinstance(model, SklearnWrapper):
        return "sklearn"
    elif isinstance(model, (Net, AdvancedNet)):
        return "neural"
    else:
        return "unknown"
