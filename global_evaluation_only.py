# ============================================================================
# GLOBAL EVALUATION
# ============================================================================

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import pickle
from pathlib import Path
from train_isolated import create_icd_frequency_tiers, evaluate_on_global_dataset, plot_enhanced_global_comparison, PARTITION_SCHEME, MIMIC_DATA_DIR, MIN_PARTITION_SIZE, TOP_K_CODES
from shared.task import load_and_partition_data

# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================
from train_isolated import SklearnWrapper

_original_predict = SklearnWrapper.predict

def memory_efficient_predict(self, X):
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
    
    if hasattr(self, 'scaler') and self.scaler is not None:
        X_np = self.scaler.transform(X_np)
    
    if X_np.shape[0] > 10000:
        print(f"    Large dataset ({X_np.shape[0]} samples) - using batch prediction...")
        predictions = []
        batch_size = 5000  # Process in smaller chunks
        for i in range(0, X_np.shape[0], batch_size):
            batch = X_np[i:i+batch_size]
            batch_pred = self.sklearn_model.predict(batch)
            predictions.append(batch_pred)
        return np.concatenate(predictions)
    else:
        return self.sklearn_model.predict(X_np)

SklearnWrapper.predict = memory_efficient_predict

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*80)
    print("GLOBAL DATASET EVALUATION ONLY")
    print("="*80)
    
    print("Loading partitions...")
    actual_top_k = None if TOP_K_CODES in ['none', 'None', None] else TOP_K_CODES
    partitions = load_and_partition_data(
        data_dir=MIMIC_DATA_DIR, 
        min_partition_size=MIN_PARTITION_SIZE,
        partition_scheme=PARTITION_SCHEME,
        top_k_codes=actual_top_k
    )
    
    print(f"Loaded {len(partitions)} partitions")
    
    print("Creating global feature space...")
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
    
    print("Loading ICD frequency tiers...")
    icd_frequency_tiers = create_icd_frequency_tiers()
    
    import train_isolated
    # Set ACTUAL_NUM_CLASSES based on TOP_K_CODES (100 top codes + 1 'other' = 101 classes)
    train_isolated.ACTUAL_NUM_CLASSES = TOP_K_CODES + 1
    print(f"Using ACTUAL_NUM_CLASSES = {train_isolated.ACTUAL_NUM_CLASSES}")
    
    print("\nStarting global dataset evaluation...")
    global_results = evaluate_on_global_dataset(partitions, global_feature_space, icd_frequency_tiers)
    
    print("\nGenerating enhanced global comparison plots...")
    plot_enhanced_global_comparison(global_results)
    
    print(f"\n{'='*80}")
    print("GLOBAL EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print("\nGLOBAL RESULTS SUMMARY:")
    print("-" * 60)
    
    for model_name, results in global_results.items():
        print(f"{model_name.upper()}:")
        print(f"  Accuracy: {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
        print(f"  Top-5 Accuracy: {results['metrics']['top5_accuracy']:.4f} ({results['metrics']['top5_accuracy']*100:.2f}%)")
        print(f"  F1 Weighted: {results['metrics']['f1_weighted']:.4f}")
        print(f"  Training Time: {results['training_time']:.1f}s")

if __name__ == "__main__":
    main()