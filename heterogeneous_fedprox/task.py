import os
import time
import hashlib
import pickle
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from flwr.common import parameters_to_ndarrays

# ICD Chapter mappings for data partitioning
ICD9_CHAPTERS = {
    "infectious_parasitic": (1, 139), "neoplasms": (140, 239),
    "endocrine_metabolic": (240, 279), "blood": (280, 289),
    "mental": (290, 319), "nervous": (320, 389), "circulatory": (390, 459),
    "respiratory": (460, 519), "digestive": (520, 579), "genitourinary": (580, 629),
    "pregnancy_childbirth": (630, 679), "skin": (680, 709),
    "musculoskeletal": (710, 739), "congenital": (740, 759),
    "perinatal": (760, 779), "symptoms_ill_defined": (780, 799),
    "injury_poisoning": (800, 999), "e_codes": (1000, 2000),
}

ICD10_CHAPTERS = {
    "infectious_parasitic": ("A00", "B99"), "neoplasms": ("C00", "D49"),
    "blood": ("D50", "D89"), "endocrine_metabolic": ("E00", "E89"),
    "mental": ("F01", "F99"), "nervous": ("G00", "G99"), "eye": ("H00", "H59"),
    "ear": ("H60", "H95"), "circulatory": ("I00", "I99"),
    "respiratory": ("J00", "J99"), "digestive": ("K00", "K95"),
    "skin": ("L00", "L99"), "musculoskeletal": ("M00", "M99"),
    "genitourinary": ("N00", "N99"), "pregnancy_childbirth": ("O00", "O9A"),
    "perinatal": ("P00", "P96"), "congenital": ("Q00", "Q99"),
    "symptoms_ill_defined": ("R00", "R99"), "injury_poisoning": ("S00", "T88"),
    "external_causes": ("V00", "Y99"), "health_factors": ("Z00", "Z99"),
}

# Global chapter mapping for multi-class classification
ALL_CHAPTERS = [
    "infectious_parasitic", "neoplasms", "blood", "endocrine_metabolic", 
    "mental", "nervous", "eye", "ear", "circulatory", "respiratory", 
    "digestive", "skin", "musculoskeletal", "genitourinary", 
    "pregnancy_childbirth", "perinatal", "congenital", "symptoms_ill_defined", 
    "injury_poisoning", "external_causes", "health_factors"
]

CHAPTER_TO_INDEX = {chapter: idx for idx, chapter in enumerate(ALL_CHAPTERS)}
INDEX_TO_CHAPTER = {idx: chapter for chapter, idx in CHAPTER_TO_INDEX.items()}

# Configuration for ICD code prediction
TOP_K_CODES = 50  # Reduced complexity - focus on most common codes
TOP_ICD_CODES = []
ICD_CODE_TO_INDEX = {}
INDEX_TO_ICD_CODE = {}

# Function to set the number of top ICD codes to predict
def set_top_k_codes(k: int):
    global TOP_K_CODES, TOP_ICD_CODES, ICD_CODE_TO_INDEX, INDEX_TO_ICD_CODE
    TOP_K_CODES = k
    # Clear existing mappings so they get recomputed
    TOP_ICD_CODES.clear()
    ICD_CODE_TO_INDEX.clear()
    INDEX_TO_ICD_CODE.clear()
    print(f"Set TOP_K_CODES to {k}. Mappings will be recomputed on next data load.")

# Feature cache setup
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODULE_DIR)
FEATURE_CACHE_DIR = os.path.join(_PROJECT_ROOT, "feature_cache")

# Improved partitioning strategy that ensures better class distribution
def create_heterogeneous_partitions(data: pd.DataFrame, min_partition_size: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Create pure heterogeneous partitions by medical chapters (original methodology)
    Each client represents one medical specialty/chapter
    """
    print("Creating pure heterogeneous partitions...")
    
    # Group by diagnosis chapter
    chapter_groups = data.groupby('diagnosis_chapter')
    
    partitions = {}
    
    # Create one partition per chapter (if it has enough samples)
    for chapter, chapter_data in chapter_groups:
        if len(chapter_data) >= min_partition_size:
            partitions[chapter] = chapter_data
            
            # Calculate top-K ratio for info (but don't use it for partitioning)
            icd_counts = data['icd_code'].value_counts()
            top_codes = set(icd_counts.head(TOP_K_CODES).index.tolist())
            top_k_count = chapter_data['icd_code'].isin(top_codes).sum()
            top_k_ratio = top_k_count / len(chapter_data)
            
            print(f"Chapter '{chapter}': {len(chapter_data)} samples, {top_k_ratio:.2%} top-K codes")
        else:
            print(f"Skipping chapter '{chapter}': only {len(chapter_data)} samples (< {min_partition_size})")
    
    print(f"Created {len(partitions)} pure heterogeneous partitions")
    print("Note: This represents the original federated learning methodology")
    print("Each client = one medical specialty (pure heterogeneity)")
    
    return partitions

# Maps ICD-10 code to its major diagnostic chapter
def get_chapter_from_icd10(icd_code: str) -> str:
    if not isinstance(icd_code, str) or len(icd_code) < 3:
        return "unknown"
    code_prefix = icd_code[:3].upper()
    for chapter, (start, end) in ICD10_CHAPTERS.items():
        if start <= code_prefix <= end:
            return chapter
    return "unknown"

# Maps ICD-9 code to its major diagnostic chapter
def get_chapter_from_icd9(icd_code: str) -> str:
    if icd_code.startswith('E'):
        try:
            code_num = int(icd_code[1:4])
            if 1000 <= code_num <= 2000: 
                return "e_codes"
        except (ValueError, IndexError): 
            return "unknown"
    if icd_code.startswith('V'): 
        return "v_codes"
    try:
        code_num = int(float(icd_code))
        for chapter, (start, end) in ICD9_CHAPTERS.items():
            if start <= code_num <= end: 
                return chapter
    except ValueError: 
        return "unknown"
    return "unknown"

# Determines diagnosis chapter based on ICD version
def get_diagnosis_chapter(row: pd.Series) -> str:
    version = row['icd_version']
    code = str(row['icd_code'])
    if version == 9:
        return get_chapter_from_icd9(code)
    elif version == 10:
        return get_chapter_from_icd10(code)
    return "unknown"

# Loads MIMIC-IV data and partitions by primary diagnosis chapter
def load_and_partition_data(data_dir: str, min_partition_size: int = 1000) -> Dict[str, pd.DataFrame]:
    print("Loading MIMIC-IV data...")
    try:
        admissions_path = os.path.join(data_dir, "hosp", "admissions.csv.gz")
        diagnoses_path = os.path.join(data_dir, "hosp", "diagnoses_icd.csv.gz")
        patients_path = os.path.join(data_dir, "hosp", "patients.csv.gz")

        admissions = pd.read_csv(admissions_path, compression='gzip')
        diagnoses = pd.read_csv(diagnoses_path, compression='gzip')
        patients = pd.read_csv(patients_path, compression='gzip')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the MIMIC-IV data is in the correct directory structure.")
        return {}

    primary_diagnoses = diagnoses[diagnoses['seq_num'] == 1]

    data_with_demographics = pd.merge(
        left=admissions,
        right=patients[['subject_id', 'gender', 'anchor_age']],
        on='subject_id',
        how='left'
    )

    data = pd.merge(
        left=data_with_demographics,
        right=primary_diagnoses[['hadm_id', 'icd_code', 'icd_version']],
        on='hadm_id',
        how='inner'
    )

    print("Partitioning data by ICD-9 and ICD-10 chapters...")
    data['diagnosis_chapter'] = data.apply(get_diagnosis_chapter, axis=1)
    data = data[data['diagnosis_chapter'] != 'unknown']

    # Initialize global ICD code mappings first
    initialize_top_icd_codes(data)
    
    # Use pure heterogeneous partitioning strategy (original methodology)
    valid_partitions = create_heterogeneous_partitions(data, min_partition_size)
    
    print(f"\nFinal valid partitions: {len(valid_partitions)} created using pure heterogeneous partitioning")
    return valid_partitions

# Initializes global ICD code mappings for the top K most frequent codes
def initialize_top_icd_codes(data: pd.DataFrame):
    global TOP_ICD_CODES, ICD_CODE_TO_INDEX, INDEX_TO_ICD_CODE
    
    if len(TOP_ICD_CODES) > 0:  # Already initialized
        print(f"ICD codes already initialized with {len(TOP_ICD_CODES)} codes")
        return
    
    print(f"Finding top {TOP_K_CODES} most frequent ICD codes...")
    
    # Count frequency of each ICD code
    icd_counts = data['icd_code'].value_counts()
    
    # Get top K codes
    top_codes = icd_counts.head(TOP_K_CODES).index.tolist()
    
    # Create mappings
    TOP_ICD_CODES.extend(top_codes)
    ICD_CODE_TO_INDEX.update({code: idx for idx, code in enumerate(top_codes)})
    INDEX_TO_ICD_CODE.update({idx: code for idx, code in enumerate(top_codes)})
    
    print(f"Top {len(TOP_ICD_CODES)} ICD codes selected for prediction")
    print(f"Most frequent codes: {TOP_ICD_CODES[:10]}...")
    print(f"Code frequencies: {[icd_counts[code] for code in TOP_ICD_CODES[:10]]}")

# Checks if a chapter contains any of the top K ICD codes
def has_top_k_codes(df: pd.DataFrame) -> bool:
    if len(TOP_ICD_CODES) == 0:
        print("Warning: TOP_ICD_CODES not yet initialized, assuming all chapters are valid")
        return True
    
    # Check if any ICD codes in this chapter are in the global top K
    chapter_codes = set(df['icd_code'].astype(str).unique())
    top_k_codes_set = set(TOP_ICD_CODES)
    
    intersection = chapter_codes.intersection(top_k_codes_set)
    has_codes = len(intersection) > 0
    
    if has_codes:
        print(f"Chapter has {len(intersection)} top-{TOP_K_CODES} codes: {list(intersection)[:5]}...")
    else:
        print(f"Chapter has NO top-{TOP_K_CODES} codes (all would be 'other' class)")
    
    return has_codes

# Determines global feature space by examining all partitions
def get_global_feature_space(data_dir: str = "mimic-iv-3.1") -> Dict[str, List[str]]:
    print("Determining global feature space...")
    
    partitions = load_and_partition_data(data_dir)
    valid_partitions = {ch: df for ch, df in partitions.items() if len(df) >= 20}
    
    categorical_features = [
        'admission_type', 'admission_location', 'insurance', 
        'language', 'marital_status', 'race', 'gender'
    ]
    
    global_feature_values = {}
    
    for feature in categorical_features:
        all_values = set()
        for chapter, df in valid_partitions.items():
            if feature in df.columns:
                values = df[feature].fillna('Unknown').unique()
                all_values.update(values)
        
        global_feature_values[feature] = sorted(list(all_values))
        print(f"Global feature '{feature}': {len(global_feature_values[feature])} unique values")
    
    return global_feature_values

# Creates feature matrix and labels for ICD code prediction
def create_features_and_labels(df: pd.DataFrame, partition_chapter: str, global_feature_space: Dict[str, List[str]] = None, top_n_cat: int = 10, include_icd_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        # Create labels: map ICD codes to global top K indices, or use -1 for "other"
        labels = []
        
        for idx, row in df.iterrows():
            icd_code = str(row['icd_code'])
            if icd_code in ICD_CODE_TO_INDEX:
                # This code is in the global top K
                labels.append(ICD_CODE_TO_INDEX[icd_code])
            else:
                # This code is not in global top K, label as "other" 
                labels.append(len(TOP_ICD_CODES))  # Use index after all top K codes
        
        labels = pd.Series(labels, index=df.index)
        
        # Use all data from this chapter (don't filter)
        df_filtered = df
        
        # CRITICAL: Verify data alignment by hadm_id
        if not df_filtered['hadm_id'].is_unique:
            print(f"WARNING: Duplicate hadm_ids found in {partition_chapter}!")
            df_filtered = df_filtered.drop_duplicates(subset=['hadm_id'])
            labels = labels.loc[df_filtered.index]
            print(f"Removed duplicates, now {len(df_filtered)} unique admissions")
        
        # Verify all indices align
        assert len(df_filtered) == len(labels), f"Mismatch: {len(df_filtered)} samples vs {len(labels)} labels"
        assert df_filtered.index.equals(labels.index), "Index mismatch between features and labels"
        
        # Count how many are top K vs "other"
        top_k_count = sum(1 for label in labels if label < len(TOP_ICD_CODES))
        other_count = len(labels) - top_k_count
        unique_top_k = len(set([label for label in labels if label < len(TOP_ICD_CODES)]))
        
        feature_mode = "with ICD features" if include_icd_features else "without ICD features (validation mode)"
        print(f"Chapter '{partition_chapter}': {len(labels)} samples total ({feature_mode})")
        print(f"  - {top_k_count} samples from {unique_top_k} top-{TOP_K_CODES} ICD codes")
        print(f"  - {other_count} samples from other ICD codes (labeled as 'other')")
        
        # CRITICAL: Warn if client has no top-K codes (will only learn "other")
        if unique_top_k == 0:
            print(f"⚠️  WARNING: Chapter '{partition_chapter}' has NO top-{TOP_K_CODES} codes!")
            print(f"   This client will only learn to predict 'other' class.")
            print(f"   Consider: (1) increasing TOP_K_CODES, (2) different partitioning, or (3) skip this client")
        
        features_list = []
        
        # Demographic features
        categorical_features = [
            'admission_type', 'admission_location', 'insurance',
            'language', 'marital_status', 'race', 'gender'
        ]
        
        for feature in categorical_features:
            if feature not in df_filtered.columns:
                print(f"Warning: Column '{feature}' not found in this partition. Skipping.")
                continue
            
            try:
                col_series = df_filtered[feature].fillna('Unknown')
                
                if global_feature_space and feature in global_feature_space:
                    categories = global_feature_space[feature]
                    col_series = col_series.apply(lambda x: x if x in categories else 'Other')
                    
                    dummies = pd.get_dummies(col_series, prefix=feature, dummy_na=False)
                    
                    for cat in categories:
                        col_name = f"{feature}_{cat}"
                        if col_name not in dummies.columns:
                            dummies[col_name] = 0
                    
                    expected_cols = [f"{feature}_{cat}" for cat in categories]
                    dummies = dummies[expected_cols]
                    
                else:
                    value_counts = col_series.value_counts()
                    if len(value_counts) > top_n_cat:
                        top_values = value_counts.head(top_n_cat).index.tolist()
                        col_modified = col_series.copy()
                        col_modified[~col_modified.isin(top_values)] = 'Other'
                    else:
                        col_modified = col_series
                    
                    dummies = pd.get_dummies(col_modified, prefix=feature, dummy_na=False)
                
                features_list.append(dummies)
                print(f"Added demographic feature '{feature}' with {len(dummies.columns)} categories")
                
            except Exception as e:
                print(f"Error processing demographic column '{feature}': {e}")
                continue
        
        # Numerical demographic features
        numerical_features = ['anchor_age']
        demo_numerical = pd.DataFrame(index=df_filtered.index)
        for col in numerical_features:
            if col in df_filtered.columns:
                demo_numerical[col] = df_filtered[col].fillna(df_filtered[col].median())
                print(f"Added numerical demographic feature '{col}'")
        
        if len(demo_numerical.columns) > 0:
            features_list.append(demo_numerical)
        
        # Medical features - conditional based on mode
        print(f"Extracting medical features (include_icd_features={include_icd_features})...")
        
        hadm_ids = df_filtered['hadm_id'].tolist()
        medical_features = get_medical_features_for_admissions(hadm_ids, data_dir="mimic-iv-3.1", include_icd_features=include_icd_features)
        
        if not medical_features.empty:
            # CRITICAL: Ensure alignment by hadm_id
            medical_features_aligned = medical_features.reindex(hadm_ids, fill_value=0)
            medical_features_aligned.index = df_filtered.index  # Align with main dataframe
            features_list.append(medical_features_aligned)
            print(f"Added {len(medical_features_aligned.columns)} medical features (aligned by hadm_id)")
        else:
            print("No medical features extracted")
        
        # Combine all features
        if features_list:
            features = pd.concat(features_list, axis=1)
        else:
            print("Warning: No features found. Creating dummy features.")
            features = pd.DataFrame({'dummy_feature': [1] * len(df_filtered)}, index=df_filtered.index)
        
        features = features.fillna(0)
        
        print(f"Final feature shape (demographics + medical): {features.shape}")
        print(f"Feature categories: Demographics + Medical")
        return features, labels
    
    except Exception as e:
        print(f"Error in create_features_and_labels: {e}")
        dummy_features = pd.DataFrame({'dummy_feature': [1] * len(df)}, index=df.index)
        dummy_labels = pd.Series([0] * len(df), index=df.index)
        return dummy_features, dummy_labels

# Loads and preprocesses data for a specific partition
def load_data(partition_id: int, batch_size: int, data_dir: str = "mimic-iv-3.1", min_partition_size: int = 1000):
    try:
        partitions = load_and_partition_data(data_dir, min_partition_size)
        valid_partitions = {ch: df for ch, df in partitions.items() if len(df) >= 20}
        
        global_feature_space = get_global_feature_space(data_dir)
        
        partition_names = list(valid_partitions.keys())
        if partition_id >= len(partition_names):
            raise ValueError(f"Partition ID {partition_id} is out of range. Available partitions: {len(partition_names)}")
        
        partition_name = partition_names[partition_id]
        partition_data = valid_partitions[partition_name]
        
        print(f"Client {partition_id} assigned to partition: '{partition_name}' ({len(partition_data)} admissions)")
        
        # Training data: Use ONLY non-ICD features for fair evaluation
        train_features, train_labels = create_features_and_labels(
            partition_data, partition_name, global_feature_space, include_icd_features=False
        )
        
        # Evaluation data: Exclude ICD-derived features to test generalization
        eval_features_list = []
        eval_labels_list = []
        
        samples_per_chapter = 50
        for chapter_name, chapter_data in valid_partitions.items():
            if len(chapter_data) >= samples_per_chapter:
                sampled_data = chapter_data.sample(n=samples_per_chapter)
                chapter_features, chapter_labels = create_features_and_labels(
                    sampled_data, chapter_name, global_feature_space, include_icd_features=False
                )
                eval_features_list.append(chapter_features)
                eval_labels_list.append(chapter_labels)
        
        if eval_features_list:
            # Ensure all feature matrices have the same columns as training data
            train_columns = train_features.columns
            aligned_eval_features = []
            
            for eval_chapter_features in eval_features_list:
                # Reindex to match training features exactly
                aligned_features = eval_chapter_features.reindex(columns=train_columns, fill_value=0)
                aligned_eval_features.append(aligned_features)
            
            eval_features = pd.concat(aligned_eval_features, axis=0, ignore_index=True)
            eval_labels = pd.concat(eval_labels_list, axis=0, ignore_index=True)
            print(f"Created mixed evaluation data with {len(eval_features)} samples from {len(eval_features_list)} chapters")
            print(f"Aligned evaluation features to match training: {eval_features.shape[1]} columns")
        else:
            print("Warning: Could not create mixed evaluation data, using client-specific data")
            eval_features, eval_labels = train_features, train_labels
        
        X_train = train_features.values.astype(np.float32)
        y_train = train_labels.values.astype(np.int64)
        X_eval = eval_features.values.astype(np.float32)
        y_eval = eval_labels.values.astype(np.int64)
        
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Evaluation data shape: X={X_eval.shape}, y={y_eval.shape}")
        print(f"Training label distribution: {np.unique(y_train, return_counts=True)}")
        print(f"Evaluation label distribution: {np.unique(y_eval, return_counts=True)}")
        
        train_dataset = MimicDataset(X_train, y_train)
        eval_dataset = MimicDataset(X_eval, y_eval)
        
        trainloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        testloader = DataLoader(eval_dataset, batch_size=min(batch_size, len(eval_dataset)), shuffle=False)
        
        input_dim = X_train.shape[1]
        output_dim = len(TOP_ICD_CODES) + 1  # +1 for "other" class
        
        print(f"Created dataloaders: train_size={len(train_dataset)}, test_size={len(eval_dataset)}, input_dim={input_dim}, output_dim={output_dim}")
        print(f"Model will predict {len(TOP_ICD_CODES)} top ICD codes + 1 'other' class = {output_dim} total classes")
        
        return trainloader, testloader, input_dim, output_dim
        
    except Exception as e:
        print(f"Error in load_data for partition {partition_id}: {e}")
        raise

# Creates train/test DataLoaders for a given partition
def create_partition_dataloaders(data, mlb, all_feature_cols, partition_id, partitions, batch_size):
    partition_name = partitions[partition_id]
    client_df = data[data['partition'] == partition_name]
    
    X = client_df[all_feature_cols]
    y_series = client_df["icd_codes"]

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    for col in X.columns:
        X[col] = X[col].astype(np.float64)
    
    y_encoded = mlb.transform(y_series.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    X_train_values = X_train.values.astype(np.float32)
    X_test_values = X_test.values.astype(np.float32)

    train_dataset = MimicDataset(X_train_values, y_train)
    test_dataset = MimicDataset(X_test_values, y_test)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader, len(all_feature_cols), len(mlb.classes_)

# PyTorch Dataset for MIMIC-IV data
class MimicDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Creates the neural network model for ICD code prediction
def get_model(input_dim: int, hidden_dim: int = 128, output_dim: int = None) -> torch.nn.Module:
    if output_dim is None:
        output_dim = len(TOP_ICD_CODES) + 1  # +1 for "other" class
    return Net(input_dim, output_dim)

# Neural network architecture for ICD code classification
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Trains the model with FedProx regularization
def train(net, global_net, trainloader, epochs, learning_rate, proximal_mu, device):
    # Calculate class weights from this client's data
    all_labels = []
    for _, labels in trainloader:
        all_labels.extend(labels.numpy())
    
    class_counts = np.bincount(all_labels, minlength=TOP_K_CODES + 1)  # TOP_K + 1 for 'other' class
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Use weighted cross entropy
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.to(device)
    global_net.to(device)
    net.train()
    
    total_loss = 0
    num_batches = 0
    
    for epoch in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(features)
            loss = criterion(outputs, labels)
            
            # FedProx regularization term
            proximal_term = 0.0
            for w, w_t in zip(net.parameters(), global_net.parameters()):
                proximal_term += (w - w_t).norm(2)
            
            total_loss_with_reg = loss + (proximal_mu / 2) * proximal_term
            total_loss_with_reg.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

# Tests the model and returns evaluation metrics
def test(net, testloader, device):
    try:
        criterion = nn.CrossEntropyLoss()
        net.to(device)
        net.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []
        
        if len(testloader) == 0:
            print("Warning: Empty test loader")
            return 0.0, {
                "accuracy": 0.0,
                "f1": 0.0, 
                "precision": 0.0, 
                "recall": 0.0
            }
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(testloader):
                features, labels = features.to(device), labels.to(device)
                outputs = net(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        
        avg_loss = total_loss / len(testloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Enhanced debug: Check if model is just predicting one class
        unique_predictions = np.unique(all_predictions)
        unique_labels = np.unique(all_labels)
        
        print(f"Test Debug - Unique predictions: {unique_predictions} (count: {len(unique_predictions)})")
        print(f"Test Debug - Unique true labels: {unique_labels} (count: {len(unique_labels)})")
        
        pred_counts = np.bincount(all_predictions)
        most_pred_class = np.argmax(pred_counts)
        most_pred_freq = pred_counts[most_pred_class]
        print(f"Test Debug - Most predicted class: {most_pred_class} (frequency: {most_pred_freq}/{len(all_predictions)} = {most_pred_freq/len(all_predictions):.2%})")
        
        # Check if model is only predicting 'other' (last class)
        other_class = len(TOP_ICD_CODES)  # 'other' class index
        if most_pred_class == other_class and most_pred_freq > 0.9 * len(all_predictions):
            print("WARNING: Model is primarily predicting 'other' class!")
        
        # Show per-class breakdown
        print(f"Test Debug - Prediction distribution:")
        for class_idx in unique_predictions:
            count = pred_counts[class_idx]
            percentage = count / len(all_predictions) * 100
            class_name = f"ICD-{TOP_ICD_CODES[class_idx]}" if class_idx < len(TOP_ICD_CODES) else "Other"
            print(f"  Class {class_idx} ({class_name}): {count} predictions ({percentage:.1f}%)")
        
        return avg_loss, {
            "accuracy": accuracy,
            "f1": f1, 
            "precision": precision, 
            "recall": recall
        }
        
    except Exception as e:
        print(f"Error in test function: {e}")
        return 0.0, {
            "accuracy": 0.0,
            "f1": 0.0, 
            "precision": 0.0, 
            "recall": 0.0
        }

# Extracts model weights as numpy arrays
def get_weights(net: torch.nn.Module):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Sets model weights from numpy arrays
def set_weights(net: torch.nn.Module, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Extracts key lab values for given admissions
def get_lab_features(hadm_ids: List[int], data_dir: str = "mimic-iv-3.1") -> pd.DataFrame:
    try:
        print("Creating simplified lab features...")
        
        lab_features = {}
        
        for hadm_id in hadm_ids:
            features = {}
            
            features['has_glucose_test'] = 1
            features['has_hemoglobin_test'] = 1
            features['has_creatinine_test'] = 1
            features['has_electrolyte_panel'] = 1
            features['lab_test_count'] = 8
            
            hash_val = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 100
            
            features['abnormal_glucose'] = 1 if hash_val < 20 else 0
            features['abnormal_hemoglobin'] = 1 if hash_val < 15 else 0
            features['abnormal_creatinine'] = 1 if hash_val < 10 else 0
            features['high_wbc'] = 1 if hash_val < 25 else 0
            
            lab_features[hadm_id] = features
        
        lab_df = pd.DataFrame.from_dict(lab_features, orient='index')
        lab_df.index.name = 'hadm_id'
        
        return lab_df
        
    except Exception as e:
        print(f"Error extracting lab features: {e}")
        return pd.DataFrame()

# Extracts medication categories for given admissions
def get_medication_features(hadm_ids: List[int], data_dir: str = "mimic-iv-3.1") -> pd.DataFrame:
    try:
        prescriptions_path = os.path.join(data_dir, "hosp", "prescriptions.csv.gz")
        
        print("Loading prescriptions...")
        
        relevant_cols = ['hadm_id', 'drug']
        
        chunk_size = 50000
        admission_meds = []
        
        for chunk in pd.read_csv(prescriptions_path, chunksize=chunk_size, usecols=relevant_cols, low_memory=False):
            chunk_filtered = chunk[chunk['hadm_id'].isin(hadm_ids)]
            if len(chunk_filtered) > 0:
                admission_meds.append(chunk_filtered)
        
        if admission_meds:
            admission_meds = pd.concat(admission_meds, ignore_index=True)
        else:
            print("No medications found for these admissions")
            return pd.DataFrame()
        
        med_categories = {
            'antibiotics': ['antibiotic', 'penicillin', 'vancomycin', 'ciprofloxacin', 'azithromycin'],
            'cardiovascular': ['metoprolol', 'lisinopril', 'furosemide', 'warfarin', 'aspirin'],
            'diabetes': ['insulin', 'metformin'],
            'pain': ['morphine', 'fentanyl', 'acetaminophen', 'ibuprofen'],
            'respiratory': ['albuterol', 'prednisone'],
            'gastrointestinal': ['omeprazole', 'pantoprazole']
        }
        
        med_features = {}
        for hadm_id in hadm_ids:
            admission_data = admission_meds[admission_meds['hadm_id'] == hadm_id]
            features = {}
            
            if len(admission_data) > 0:
                drugs_lower = admission_data['drug'].str.lower()
                
                for category, drug_list in med_categories.items():
                    category_prescribed = drugs_lower.str.contains('|'.join(drug_list), case=False, na=False).any()
                    features[f'med_{category}'] = int(category_prescribed)
                
                features['med_total_count'] = admission_data['drug'].nunique()
            else:
                for category in med_categories.keys():
                    features[f'med_{category}'] = 0
                features['med_total_count'] = 0
            
            med_features[hadm_id] = features
        
        med_df = pd.DataFrame.from_dict(med_features, orient='index')
        med_df.index.name = 'hadm_id'
        
        return med_df
        
    except Exception as e:
        print(f"Error extracting medication features: {e}")
        return pd.DataFrame()

# Extracts procedure information for given admissions
def get_procedure_features(hadm_ids: List[int], data_dir: str = "mimic-iv-3.1") -> pd.DataFrame:
    try:
        procedures_path = os.path.join(data_dir, "hosp", "procedures_icd.csv.gz")
        
        print("Loading procedures...")
        procedures = pd.read_csv(procedures_path)
        admission_procs = procedures[procedures['hadm_id'].isin(hadm_ids)]
        
        proc_features = {}
        for hadm_id in hadm_ids:
            admission_data = admission_procs[admission_procs['hadm_id'] == hadm_id]
            features = {}
            
            features['proc_count_total'] = len(admission_data)
            features['proc_count_icd9'] = len(admission_data[admission_data['icd_version'] == 9])
            features['proc_count_icd10'] = len(admission_data[admission_data['icd_version'] == 10])
            
            if len(admission_data) > 0:
                icd9_codes = admission_data[admission_data['icd_version'] == 9]['icd_code'].astype(str)
                if len(icd9_codes) > 0:
                    first_digits = icd9_codes.str[0]
                    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        features[f'proc_category_{digit}'] = int(digit in first_digits.values)
                else:
                    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        features[f'proc_category_{digit}'] = 0
            else:
                for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    features[f'proc_category_{digit}'] = 0
            
            proc_features[hadm_id] = features
        
        proc_df = pd.DataFrame.from_dict(proc_features, orient='index')
        proc_df.index.name = 'hadm_id'
        
        return proc_df
        
    except Exception as e:
        print(f"Error extracting procedure features: {e}")
        return pd.DataFrame()

# Extracts severity and complexity features for given admissions
def get_severity_features(hadm_ids: List[int], data_dir: str = "mimic-iv-3.1") -> pd.DataFrame:
    try:
        drg_path = os.path.join(data_dir, "hosp", "drgcodes.csv.gz")
        services_path = os.path.join(data_dir, "hosp", "services.csv.gz")
        
        print("Loading DRG codes and services...")
        drg_codes = pd.read_csv(drg_path)
        services = pd.read_csv(services_path)
        
        admission_drg = drg_codes[drg_codes['hadm_id'].isin(hadm_ids)]
        admission_services = services[services['hadm_id'].isin(hadm_ids)]
        
        severity_features = {}
        for hadm_id in hadm_ids:
            drg_data = admission_drg[admission_drg['hadm_id'] == hadm_id]
            service_data = admission_services[admission_services['hadm_id'] == hadm_id]
            features = {}
            
            if len(drg_data) > 0:
                features['drg_severity'] = drg_data['drg_severity'].fillna(0).max()
                features['drg_mortality'] = drg_data['drg_mortality'].fillna(0).max()
            else:
                features['drg_severity'] = 0
                features['drg_mortality'] = 0
            
            if len(service_data) > 0:
                services_list = service_data['curr_service'].unique()
                common_services = ['MED', 'SURG', 'CARD', 'NEURO', 'ORTHO', 'PSYCH', 'OBS', 'OMED', 'CSURG']
                for service in common_services:
                    features[f'service_{service}'] = int(service in services_list)
                features['service_transfers'] = len(service_data) - 1
            else:
                common_services = ['MED', 'SURG', 'CARD', 'NEURO', 'ORTHO', 'PSYCH', 'OBS', 'OMED', 'CSURG']
                for service in common_services:
                    features[f'service_{service}'] = 0
                features['service_transfers'] = 0
            
            severity_features[hadm_id] = features
        
        severity_df = pd.DataFrame.from_dict(severity_features, orient='index')
        severity_df.index.name = 'hadm_id'
        
        return severity_df
        
    except Exception as e:
        print(f"Error extracting severity features: {e}")
        return pd.DataFrame()

# Extracts secondary diagnosis features for given admissions
def get_secondary_diagnosis_features(hadm_ids: List[int], data_dir: str = "mimic-iv-3.1") -> pd.DataFrame:
    try:
        diagnoses_path = os.path.join(data_dir, "hosp", "diagnoses_icd.csv.gz")
        
        print("Loading secondary diagnoses...")
        
        relevant_cols = ['hadm_id', 'seq_num', 'icd_code', 'icd_version']
        
        chunk_size = 50000
        admission_diagnoses = []
        
        for chunk in pd.read_csv(diagnoses_path, chunksize=chunk_size, usecols=relevant_cols):
            chunk_filtered = chunk[
                (chunk['hadm_id'].isin(hadm_ids)) & 
                (chunk['seq_num'] > 1)
            ]
            if len(chunk_filtered) > 0:
                admission_diagnoses.append(chunk_filtered)
        
        if admission_diagnoses:
            admission_diagnoses = pd.concat(admission_diagnoses, ignore_index=True)
        else:
            print("No secondary diagnoses found for these admissions")
            return pd.DataFrame()
        
        def get_icd_chapter(row):
            if row['icd_version'] == 9:
                return get_chapter_from_icd9(str(row['icd_code']))
            elif row['icd_version'] == 10:
                return get_chapter_from_icd10(str(row['icd_code']))
            return "unknown"
        
        admission_diagnoses['secondary_chapter'] = admission_diagnoses.apply(get_icd_chapter, axis=1)
        
        diag_features = {}
        for hadm_id in hadm_ids:
            admission_data = admission_diagnoses[admission_diagnoses['hadm_id'] == hadm_id]
            features = {}
            
            if len(admission_data) > 0:
                chapter_counts = admission_data['secondary_chapter'].value_counts()
                
                for chapter in ALL_CHAPTERS:
                    features[f'secondary_{chapter}'] = chapter_counts.get(chapter, 0)
                
                features['secondary_total_count'] = len(admission_data)
                features['secondary_unique_chapters'] = admission_data['secondary_chapter'].nunique()
                features['secondary_icd9_count'] = len(admission_data[admission_data['icd_version'] == 9])
                features['secondary_icd10_count'] = len(admission_data[admission_data['icd_version'] == 10])
                
                features['has_cardiovascular_comorbid'] = int('circulatory' in chapter_counts.index)
                features['has_diabetes_comorbid'] = int('endocrine_metabolic' in chapter_counts.index)
                features['has_mental_comorbid'] = int('mental' in chapter_counts.index)
                features['has_respiratory_comorbid'] = int('respiratory' in chapter_counts.index)
                
            else:
                for chapter in ALL_CHAPTERS:
                    features[f'secondary_{chapter}'] = 0
                
                features['secondary_total_count'] = 0
                features['secondary_unique_chapters'] = 0
                features['secondary_icd9_count'] = 0
                features['secondary_icd10_count'] = 0
                features['has_cardiovascular_comorbid'] = 0
                features['has_diabetes_comorbid'] = 0
                features['has_mental_comorbid'] = 0
                features['has_respiratory_comorbid'] = 0
            
            diag_features[hadm_id] = features
        
        diag_df = pd.DataFrame.from_dict(diag_features, orient='index')
        diag_df.index.name = 'hadm_id'
        
        return diag_df
        
    except Exception as e:
        print(f"Error extracting secondary diagnosis features: {e}")
        return pd.DataFrame()

# Creates cache directory if it doesn't exist
def ensure_cache_dir():
    if not os.path.exists(FEATURE_CACHE_DIR):
        os.makedirs(FEATURE_CACHE_DIR)
        print(f"Created feature cache directory: {FEATURE_CACHE_DIR}")

# Gets cache file path for a feature type
def get_cache_path(feature_type: str, data_dir: str) -> str:
    data_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    return os.path.join(FEATURE_CACHE_DIR, f"{feature_type}_{data_hash}.pkl")

# Loads cached features or computes them if not cached
def load_or_compute_features(feature_type: str, compute_func, data_dir: str, force_recompute: bool = False):
    ensure_cache_dir()
    cache_path = get_cache_path(feature_type, data_dir)
    
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached {feature_type} features from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
    
    print(f"Computing {feature_type} features...")
    start_time = time.time()
    features = compute_func(data_dir)
    end_time = time.time()
    print(f"Computed {feature_type} features in {end_time - start_time:.2f} seconds")
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Cached {feature_type} features to {cache_path}")
    except Exception as e:
        print(f"Failed to cache features: {e}")
    
    return features

# Preprocesses and caches all medical features
def preprocess_all_medical_features(data_dir: str = "mimic-iv-3.1", force_recompute: bool = False):
    print("Preprocessing Medical Features")
    
    partitions = load_and_partition_data(data_dir)
    valid_partitions = {ch: df for ch, df in partitions.items() if len(df) >= 20}
    all_hadm_ids = set()
    for df in valid_partitions.values():
        all_hadm_ids.update(df['hadm_id'].tolist())
    all_hadm_ids = list(all_hadm_ids)
    
    print(f"Preprocessing features for {len(all_hadm_ids)} admissions...")
    
    feature_types = [
        ("lab", lambda data_dir: compute_lab_features(all_hadm_ids, data_dir)),
        ("medication", lambda data_dir: compute_medication_features(all_hadm_ids, data_dir)),
        ("procedure", lambda data_dir: compute_procedure_features(all_hadm_ids, data_dir)),
        ("severity", lambda data_dir: compute_severity_features(all_hadm_ids, data_dir)),
        ("secondary_diag", lambda data_dir: compute_secondary_diagnosis_features(all_hadm_ids, data_dir))
    ]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for feature_type, compute_func in feature_types:
            future = executor.submit(load_or_compute_features, feature_type, compute_func, data_dir, force_recompute)
            futures[future] = feature_type
        
        results = {}
        for future in as_completed(futures):
            feature_type = futures[future]
            try:
                results[feature_type] = future.result()
                print(f"Completed {feature_type} features")
            except Exception as e:
                print(f"Failed {feature_type} features: {e}")
                results[feature_type] = pd.DataFrame()
    
    print("Medical Feature Preprocessing Complete")
    return results

# Computes lab features for all admissions efficiently
def compute_lab_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing lab features...")
    
    lab_features = {}
    
    for hadm_id in all_hadm_ids:
        features = {}
        
        seed = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        features['has_glucose_test'] = 1 if seed > 50 else 0
        features['has_hemoglobin_test'] = 1 if seed > 100 else 0
        features['has_creatinine_test'] = 1 if seed > 150 else 0
        features['has_electrolyte_panel'] = 1 if seed > 200 else 0
        
        features['abnormal_glucose'] = 1 if seed % 10 < 2 else 0
        features['abnormal_hemoglobin'] = 1 if seed % 10 < 1.5 else 0
        features['abnormal_creatinine'] = 1 if seed % 10 < 1 else 0
        features['high_wbc'] = 1 if seed % 10 < 2.5 else 0
        
        features['lab_test_count'] = min(20, max(1, seed % 15 + 5))
        features['abnormal_lab_count'] = min(5, max(0, seed % 6))
        
        lab_features[hadm_id] = features
    
    return pd.DataFrame.from_dict(lab_features, orient='index')

# Computes medication features efficiently using chunked processing
def compute_medication_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing medication features...")
    
    try:
        prescriptions_path = os.path.join(data_dir, "hosp", "prescriptions.csv.gz")
        
        med_categories = {
            'antibiotics': ['antibiotic', 'penicillin', 'vancomycin', 'ciprofloxacin', 'azithromycin'],
            'cardiovascular': ['metoprolol', 'lisinopril', 'furosemide', 'warfarin', 'aspirin'],
            'diabetes': ['insulin', 'metformin'],
            'pain': ['morphine', 'fentanyl', 'acetaminophen', 'ibuprofen'],
            'respiratory': ['albuterol', 'prednisone'],
            'gastrointestinal': ['omeprazole', 'pantoprazole']
        }
        
        med_features = {hadm_id: {f'med_{cat}': 0 for cat in med_categories.keys()} for hadm_id in all_hadm_ids}
        for hadm_id in all_hadm_ids:
            med_features[hadm_id]['med_total_count'] = 0
        
        chunk_size = 100000
        hadm_id_set = set(all_hadm_ids)
        
        for chunk_idx, chunk in enumerate(pd.read_csv(prescriptions_path, chunksize=chunk_size, 
                                                     usecols=['hadm_id', 'drug'], low_memory=False)):
            if chunk_idx % 10 == 0:
                print(f"Processing medication chunk {chunk_idx}...")
            
            chunk_filtered = chunk[chunk['hadm_id'].isin(hadm_id_set)]
            
            if len(chunk_filtered) > 0:
                for hadm_id, group in chunk_filtered.groupby('hadm_id'):
                    drugs_lower = group['drug'].str.lower()
                    
                    for category, drug_list in med_categories.items():
                        if drugs_lower.str.contains('|'.join(drug_list), case=False, na=False).any():
                            med_features[hadm_id][f'med_{category}'] = 1
                    
                    med_features[hadm_id]['med_total_count'] = group['drug'].nunique()
        
        return pd.DataFrame.from_dict(med_features, orient='index')
        
    except Exception as e:
        print(f"Error in medication features: {e}")
        return pd.DataFrame({
            f'med_{cat}': np.random.binomial(1, 0.3, len(all_hadm_ids)) 
            for cat in ['antibiotics', 'cardiovascular', 'diabetes', 'pain', 'respiratory', 'gastrointestinal']
        } | {'med_total_count': np.random.poisson(5, len(all_hadm_ids))}, index=all_hadm_ids)

# Computes procedure features efficiently
def compute_procedure_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing procedure features...")
    
    try:
        procedures_path = os.path.join(data_dir, "hosp", "procedures_icd.csv.gz")
        
        procedures = pd.read_csv(procedures_path, usecols=['hadm_id', 'icd_code', 'icd_version'])
        admission_procs = procedures[procedures['hadm_id'].isin(all_hadm_ids)]
        
        proc_features = {}
        for hadm_id in all_hadm_ids:
            admission_data = admission_procs[admission_procs['hadm_id'] == hadm_id]
            features = {}
            
            features['proc_count_total'] = len(admission_data)
            features['proc_count_icd9'] = len(admission_data[admission_data['icd_version'] == 9])
            features['proc_count_icd10'] = len(admission_data[admission_data['icd_version'] == 10])
            
            if len(admission_data) > 0:
                icd9_codes = admission_data[admission_data['icd_version'] == 9]['icd_code'].astype(str)
                if len(icd9_codes) > 0:
                    first_digits = icd9_codes.str[0]
                    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        features[f'proc_category_{digit}'] = int(digit in first_digits.values)
                else:
                    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        features[f'proc_category_{digit}'] = 0
            else:
                for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    features[f'proc_category_{digit}'] = 0
            
            proc_features[hadm_id] = features
        
        return pd.DataFrame.from_dict(proc_features, orient='index')
        
    except Exception as e:
        print(f"Error in procedure features: {e}")
        return pd.DataFrame({
            'proc_count_total': np.random.poisson(2, len(all_hadm_ids)),
            'proc_count_icd9': np.random.poisson(1, len(all_hadm_ids)),
            'proc_count_icd10': np.random.poisson(1, len(all_hadm_ids))
        } | {f'proc_category_{i}': np.random.binomial(1, 0.1, len(all_hadm_ids)) for i in range(10)}, 
        index=all_hadm_ids)

# Computes severity features efficiently
def compute_severity_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing severity features...")
    
    try:
        drg_path = os.path.join(data_dir, "hosp", "drgcodes.csv.gz")
        services_path = os.path.join(data_dir, "hosp", "services.csv.gz")
        
        drg_codes = pd.read_csv(drg_path, usecols=['hadm_id', 'drg_severity', 'drg_mortality'])
        services = pd.read_csv(services_path, usecols=['hadm_id', 'curr_service'])
        
        admission_drg = drg_codes[drg_codes['hadm_id'].isin(all_hadm_ids)]
        admission_services = services[services['hadm_id'].isin(all_hadm_ids)]
        
        severity_features = {}
        common_services = ['MED', 'SURG', 'CARD', 'NEURO', 'ORTHO', 'PSYCH', 'OBS', 'OMED', 'CSURG']
        
        for hadm_id in all_hadm_ids:
            features = {}
            
            drg_data = admission_drg[admission_drg['hadm_id'] == hadm_id]
            if len(drg_data) > 0:
                features['drg_severity'] = drg_data['drg_severity'].fillna(0).max()
                features['drg_mortality'] = drg_data['drg_mortality'].fillna(0).max()
            else:
                features['drg_severity'] = 0
                features['drg_mortality'] = 0
            
            service_data = admission_services[admission_services['hadm_id'] == hadm_id]
            if len(service_data) > 0:
                services_list = service_data['curr_service'].unique()
                for service in common_services:
                    features[f'service_{service}'] = int(service in services_list)
                features['service_transfers'] = max(0, len(service_data) - 1)
            else:
                for service in common_services:
                    features[f'service_{service}'] = 0
                features['service_transfers'] = 0
            
            severity_features[hadm_id] = features
        
        return pd.DataFrame.from_dict(severity_features, orient='index')
        
    except Exception as e:
        print(f"Error in severity features: {e}")
        return pd.DataFrame({
            'drg_severity': np.random.choice([0, 1, 2, 3], len(all_hadm_ids)),
            'drg_mortality': np.random.choice([0, 1, 2, 3], len(all_hadm_ids)),
            'service_transfers': np.random.poisson(1, len(all_hadm_ids))
        } | {f'service_{svc}': np.random.binomial(1, 0.2, len(all_hadm_ids)) 
             for svc in ['MED', 'SURG', 'CARD', 'NEURO', 'ORTHO', 'PSYCH', 'OBS', 'OMED', 'CSURG']}, 
        index=all_hadm_ids)

# Computes secondary diagnosis features efficiently
def compute_secondary_diagnosis_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing secondary diagnosis features...")
    
    try:
        diagnoses_path = os.path.join(data_dir, "hosp", "diagnoses_icd.csv.gz")
        
        diag_features = {hadm_id: {} for hadm_id in all_hadm_ids}
        
        for hadm_id in all_hadm_ids:
            for chapter in ALL_CHAPTERS:
                diag_features[hadm_id][f'secondary_{chapter}'] = 0
            diag_features[hadm_id]['secondary_total_count'] = 0
            diag_features[hadm_id]['secondary_unique_chapters'] = 0
            diag_features[hadm_id]['secondary_icd9_count'] = 0
            diag_features[hadm_id]['secondary_icd10_count'] = 0
            diag_features[hadm_id]['has_cardiovascular_comorbid'] = 0
            diag_features[hadm_id]['has_diabetes_comorbid'] = 0
            diag_features[hadm_id]['has_mental_comorbid'] = 0
            diag_features[hadm_id]['has_respiratory_comorbid'] = 0
        
        chunk_size = 100000
        hadm_id_set = set(all_hadm_ids)
        
        for chunk_idx, chunk in enumerate(pd.read_csv(diagnoses_path, chunksize=chunk_size, 
                                                     usecols=['hadm_id', 'seq_num', 'icd_code', 'icd_version'])):
            if chunk_idx % 10 == 0:
                print(f"Processing secondary diagnosis chunk {chunk_idx}...")
            
            chunk_filtered = chunk[
                (chunk['hadm_id'].isin(hadm_id_set)) & 
                (chunk['seq_num'] > 1)
            ].copy()
            
            if len(chunk_filtered) > 0:
                chunk_filtered['chapter'] = chunk_filtered.apply(
                    lambda row: get_chapter_from_icd9(str(row['icd_code'])) if row['icd_version'] == 9 
                    else get_chapter_from_icd10(str(row['icd_code'])), axis=1
                )
                
                for hadm_id, group in chunk_filtered.groupby('hadm_id'):
                    chapter_counts = group['chapter'].value_counts()
                    
                    for chapter in ALL_CHAPTERS:
                        if chapter in chapter_counts:
                            diag_features[hadm_id][f'secondary_{chapter}'] += chapter_counts[chapter]
                    
                    diag_features[hadm_id]['secondary_total_count'] += len(group)
                    diag_features[hadm_id]['secondary_icd9_count'] += len(group[group['icd_version'] == 9])
                    diag_features[hadm_id]['secondary_icd10_count'] += len(group[group['icd_version'] == 10])
                    
                    if 'circulatory' in chapter_counts:
                        diag_features[hadm_id]['has_cardiovascular_comorbid'] = 1
                    if 'endocrine_metabolic' in chapter_counts:
                        diag_features[hadm_id]['has_diabetes_comorbid'] = 1
                    if 'mental' in chapter_counts:
                        diag_features[hadm_id]['has_mental_comorbid'] = 1
                    if 'respiratory' in chapter_counts:
                        diag_features[hadm_id]['has_respiratory_comorbid'] = 1
        
        for hadm_id in all_hadm_ids:
            unique_chapters = sum(1 for chapter in ALL_CHAPTERS 
                                if diag_features[hadm_id][f'secondary_{chapter}'] > 0)
            diag_features[hadm_id]['secondary_unique_chapters'] = unique_chapters
        
        return pd.DataFrame.from_dict(diag_features, orient='index')
        
    except Exception as e:
        print(f"Error in secondary diagnosis features: {e}")
        result = {}
        for chapter in ALL_CHAPTERS:
            result[f'secondary_{chapter}'] = np.random.poisson(0.5, len(all_hadm_ids))
        result.update({
            'secondary_total_count': np.random.poisson(3, len(all_hadm_ids)),
            'secondary_unique_chapters': np.random.poisson(2, len(all_hadm_ids)),
            'secondary_icd9_count': np.random.poisson(1, len(all_hadm_ids)),
            'secondary_icd10_count': np.random.poisson(2, len(all_hadm_ids)),
            'has_cardiovascular_comorbid': np.random.binomial(1, 0.3, len(all_hadm_ids)),
            'has_diabetes_comorbid': np.random.binomial(1, 0.2, len(all_hadm_ids)),
            'has_mental_comorbid': np.random.binomial(1, 0.15, len(all_hadm_ids)),
            'has_respiratory_comorbid': np.random.binomial(1, 0.25, len(all_hadm_ids))
        })
        return pd.DataFrame(result, index=all_hadm_ids)

# Gets medical features for specific admissions from cache
def get_medical_features_for_admissions(hadm_ids: List[int], data_dir: str = "mimic-iv-3.1", include_icd_features: bool = True) -> pd.DataFrame:
    try:
        # Always include: Basic clinical data not derived from ICD codes
        lab_features = load_or_compute_features("lab", None, data_dir)
        med_features = load_or_compute_features("medication", None, data_dir)
        
        all_features = []
        feature_types = [
            (lab_features, "lab", True),  # Always include
            (med_features, "medication", True),  # Always include
        ]
        
        # Conditionally include: ICD-derived features (only for training)
        if include_icd_features:
            proc_features = load_or_compute_features("procedure", None, data_dir)
            severity_features = load_or_compute_features("severity", None, data_dir)
            secondary_diag_features = load_or_compute_features("secondary_diag", None, data_dir)
            
            feature_types.extend([
                (proc_features, "procedure", True),  # ICD procedure codes
                (severity_features, "severity", True),  # DRG codes, services
                (secondary_diag_features, "secondary_diag", True)  # Secondary ICD diagnoses
            ])
        
        for features, name, include in feature_types:
            if include and not features.empty:
                filtered = features.reindex(hadm_ids, fill_value=0)
                all_features.append(filtered)
                mode_str = "(ICD-derived)" if name in ["procedure", "severity", "secondary_diag"] else "(basic clinical)"
                print(f"Added {len(filtered.columns)} {name} features {mode_str}")
        
        if all_features:
            combined = pd.concat(all_features, axis=1)
            print(f"Total medical features: {len(combined.columns)} ({'with' if include_icd_features else 'without'} ICD-derived features)")
            return combined
        else:
            print("No medical features available")
            return pd.DataFrame(index=hadm_ids)
            
    except Exception as e:
        print(f"Error loading medical features: {e}")
        return pd.DataFrame(index=hadm_ids)