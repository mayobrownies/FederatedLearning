# ULCD (Unified Latent Consensus Distillation) Components
# Based on advisor's implementation for true latent consensus distillation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Optional
import numpy as np

# ============================================================================
# LORA MODULE (for ULCD)
# ============================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        self.down = nn.Linear(in_features, r, bias=False)
        self.up = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))

# ============================================================================
# TABULAR TRANSFORMER (for ULCD)
# ============================================================================
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

# ============================================================================
# MULTIMODAL ULCD CLIENT
# ============================================================================
class MultimodalULCDClient(nn.Module):
    def __init__(self, tabular_dim=20, text_model_name='bert-base-uncased', latent_dim=64, task_out=1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Text encoder (BERT)
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, latent_dim)
        
        # Tabular encoder (Transformer with LoRA)
        self.tabular_encoder = TabularTransformer(
            input_dim=tabular_dim, 
            d_model=latent_dim,
            use_lora=True
        )
        
        # Fusion layer
        self.fusion = nn.Linear(2 * latent_dim, latent_dim)
        
        # Task-specific head
        self.head = nn.Linear(latent_dim, task_out)

    def forward(self, tabular_x, text_input):
        """Forward pass for multimodal data"""
        # Encode tabular data
        tab_feat = self.tabular_encoder(tabular_x)         # [B, latent_dim]
        
        # Encode text data
        bert_out = self.bert(**text_input).pooler_output   # [B, 768]
        text_feat = self.text_proj(bert_out)               # [B, latent_dim]
        
        # Fuse modalities
        fused = torch.cat([tab_feat, text_feat], dim=-1)   # [B, 2*latent_dim]
        z = self.fusion(fused)                             # [B, latent_dim]
        
        # Task prediction
        y = self.head(z)                                   # [B, task_out]
        
        return z, y

    def get_latent_summary(self, loader):
        """Extract latent summary for ULCD communication"""
        self.eval()
        with torch.no_grad():
            all_z = []
            for batch in loader:
                if len(batch) == 4:  # (tabular, input_ids, attention_mask, labels)
                    tab_x, input_ids, attention_mask, _ = batch
                    text_input = {'input_ids': input_ids, 'attention_mask': attention_mask}
                elif len(batch) == 3:  # (tabular, text_dict, labels) or (tabular, input_ids, attention_mask)
                    tab_x, second, third = batch
                    if isinstance(second, dict):
                        text_input = second
                    else:
                        text_input = {'input_ids': second, 'attention_mask': third}
                else:  # Handle other formats
                    tab_x = batch[0]
                    text_input = {'input_ids': batch[1], 'attention_mask': batch[2]} if len(batch) > 2 else None
                
                if text_input is not None:
                    z, _ = self.forward(tab_x, text_input)
                    all_z.append(z.mean(dim=0))  # Average over batch
            
            if all_z:
                return torch.stack(all_z).mean(dim=0)  # Average over batches
            else:
                return torch.zeros(self.latent_dim)

    def fine_tune_with_prototype(self, prototype, loader, epochs=2, lr=1e-4):
        """Fine-tune model to align with server prototype"""
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in loader:
                if len(batch) == 3:
                    tab_x, text_input, y = batch
                else:
                    tab_x, text_input = batch[:2]
                    y = torch.ones(tab_x.size(0), 1)  # Dummy labels
                
                z, y_hat = self.forward(tab_x, text_input)
                
                # Alignment loss with prototype
                align_loss = F.mse_loss(z, prototype.expand_as(z))
                
                # Task loss
                task_loss = F.mse_loss(y_hat, y.float())
                
                # Combined loss
                loss = align_loss + task_loss
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Fine-tune epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")

# ============================================================================
# ULCD SERVER
# ============================================================================
class ULCDServer(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.prototype = nn.Parameter(torch.randn(latent_dim))

    def aggregate_latents(self, latents: List[torch.Tensor]):
        """Aggregate latent representations into prototype"""
        if latents:
            self.prototype.data = torch.stack(latents).mean(dim=0)
        else:
            print("[WARNING] No latents to aggregate")

    def detect_anomalies(self, latents: List[torch.Tensor], threshold=0.3):
        """Detect anomalous clients based on cosine similarity to prototype"""
        trusted, flagged = [], []
        
        for i, z in enumerate(latents):
            sim = F.cosine_similarity(z, self.prototype, dim=0).item()
            if sim >= threshold:
                trusted.append((i, z))
            else:
                flagged.append(i)
        
        return trusted, flagged

    def get_prototype(self):
        """Get current prototype for sharing with clients"""
        return self.prototype.detach().clone()

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================
def visualize_latents(latents: List[Tuple[int, torch.Tensor]], prototype: torch.Tensor, name: str, save_dir: str = "fl_plots"):
    """Visualize latent representations and prototype using PCA"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Combine all latents and prototype
        all_z = torch.stack([z for (_, z) in latents] + [prototype.detach()])
        coords = PCA(n_components=2).fit_transform(all_z.numpy())
        
        plt.figure(figsize=(10, 8))
        
        # Plot client latents
        for i, pt in enumerate(coords[:-1]):
            plt.scatter(pt[0], pt[1], label=f'Client {latents[i][0]}', s=100, alpha=0.7)
        
        # Plot server prototype
        plt.scatter(coords[-1][0], coords[-1][1], marker='x', c='black', s=200, linewidth=3, label='Server Prototype')
        
        plt.legend()
        plt.title(f'ULCD Latent Space: {name}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = f"{save_dir}/ulcd_latents_{name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] ULCD latent visualization: {filename}")
        
    except Exception as e:
        print(f"[WARNING] Could not create ULCD visualization: {e}")

# ============================================================================
# DATA PREPROCESSING FOR MULTIMODAL ULCD
# ============================================================================
def create_dummy_multimodal_data(tabular_features, num_samples=1000):
    """Create dummy multimodal data for testing when real MIMIC notes aren't available"""
    
    # Create dummy text data
    dummy_texts = [
        f"Patient {i}: Admitted with symptoms. Vital signs stable. Treatment ongoing." 
        for i in range(num_samples)
    ]
    
    # Tokenize text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(
        dummy_texts, 
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    
    return {
        'tabular': tabular_features,
        'text_input_ids': encodings['input_ids'],
        'text_attention_mask': encodings['attention_mask']
    }