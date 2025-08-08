import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F

# ============================================================================
# DATASETS
# ============================================================================
class MimicDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================================================
# NEURAL NETWORKS
# ============================================================================
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
def get_model(model_name: str, input_dim: int, output_dim: int):
    if model_name == "basic":
        return Net(input_dim, output_dim)
    elif model_name == "advanced":
        return AdvancedNet(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
