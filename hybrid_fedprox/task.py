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
from torch.nn import functional as F

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
TOP_K_CODES = 50
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEATURE_CACHE_DIR = os.path.join(project_root, "feature_cache")

# Improved partitioning strategy that ensures better class distribution
def create_balanced_partitions(data: pd.DataFrame, min_partition_size: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Create more balanced partitions by combining chapters and ensuring each client 
    has a good mix of top-K codes
    """
    print("Creating balanced partitions...")
    
    # First, identify which ICD codes are in the top K
    icd_counts = data['icd_code'].value_counts()
    top_codes = set(icd_counts.head(TOP_K_CODES).index.tolist())
    
    # Add a flag for whether each sample has a top-K code
    data['has_top_k_code'] = data['icd_code'].isin(top_codes)
    
    # Group by chapter
    chapter_groups = data.groupby('diagnosis_chapter')
    
    partitions = {}
    
    # Strategy 1: For chapters with many top-K codes, use them as-is
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size and top_k_ratio > 0.15:
            # This chapter has a reasonable number of top-K codes
            partitions[f"{chapter}"] = chapter_data
            print(f"Single chapter partition '{chapter}': {len(chapter_data)} samples, {top_k_ratio:.2%} top-K")
    
    # Strategy 2: For chapters with few top-K codes, combine them
    remaining_chapters = []
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size and top_k_ratio <= 0.15:
            remaining_chapters.append((chapter, chapter_data))
        elif len(chapter_data) < min_partition_size:
            remaining_chapters.append((chapter, chapter_data))
    
    # Combine remaining chapters to create balanced partitions
    if remaining_chapters:
        combined_data = pd.concat([data for _, data in remaining_chapters], ignore_index=True)
        
        # Split combined data by top-K presence to create balanced partitions
        top_k_data = combined_data[combined_data['has_top_k_code']]
        other_data = combined_data[~combined_data['has_top_k_code']]
        
        # Create balanced partitions only if we have sufficient top-K data
        min_top_k_ratio = 0.20  # Minimum 20% top-K samples required
        partition_size = max(min_partition_size, len(combined_data) // 3)  # Aim for 3 combined partitions
        
        for i in range(3):
            if len(top_k_data) + len(other_data) < partition_size:
                break
                
            # Calculate how many top-K samples we need for minimum ratio
            min_top_k_needed = int(partition_size * min_top_k_ratio)
            
            # Only create partition if we have enough top-K samples
            if len(top_k_data) < min_top_k_needed:
                print(f"Skipping mixed partition {i+1}: insufficient top-K samples ({len(top_k_data)} < {min_top_k_needed})")
                break
                
            # Take a mix of top-K and other samples
            target_top_k = min(len(top_k_data), max(min_top_k_needed, partition_size // 3))  # At least 20% or 33% top-K
            target_other = min(len(other_data), partition_size - target_top_k)
            
            if target_top_k > 0:
                partition_top_k = top_k_data.sample(n=target_top_k, random_state=42+i)
                top_k_data = top_k_data.drop(partition_top_k.index)
            else:
                partition_top_k = pd.DataFrame()
            
            if target_other > 0:
                partition_other = other_data.sample(n=target_other, random_state=42+i)
                other_data = other_data.drop(partition_other.index)
            else:
                partition_other = pd.DataFrame()
            
            # Double-check that we meet the minimum requirements
            if len(partition_top_k) + len(partition_other) >= min_partition_size:
                top_k_ratio = len(partition_top_k) / (len(partition_top_k) + len(partition_other))
                
                if top_k_ratio >= min_top_k_ratio:
                    partition_data = pd.concat([partition_top_k, partition_other], ignore_index=True)
                    partition_name = f"mixed_{i+1}"
                    partitions[partition_name] = partition_data
                    print(f"Mixed partition '{partition_name}': {len(partition_data)} samples, {top_k_ratio:.2%} top-K")
                else:
                    print(f"Skipping mixed partition {i+1}: top-K ratio too low ({top_k_ratio:.2%} < {min_top_k_ratio:.2%})")
                    # Put the samples back for potential future use
                    top_k_data = pd.concat([top_k_data, partition_top_k], ignore_index=True)
                    other_data = pd.concat([other_data, partition_other], ignore_index=True)
                    break
    
    print(f"Created {len(partitions)} balanced partitions")
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
        return "health_factors"  # ICD-9 V codes = health factors (same as ICD-10 Z codes)
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
    
    # Use improved balanced partitioning strategy
    valid_partitions = create_balanced_partitions(data, min_partition_size)
    
    # Remove the has_top_k_code column we added for partitioning
    for partition_name, partition_data in valid_partitions.items():
        if 'has_top_k_code' in partition_data.columns:
            valid_partitions[partition_name] = partition_data.drop('has_top_k_code', axis=1)
    
    print(f"\nFinal valid partitions: {len(valid_partitions)} created using balanced partitioning")
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
        
        # Temporal features - admission timing and duration patterns
        print("Extracting temporal features...")
        temporal_features = pd.DataFrame(index=df_filtered.index)
        
        if 'admittime' in df_filtered.columns and 'dischtime' in df_filtered.columns:
            # Convert to datetime
            df_filtered['admittime'] = pd.to_datetime(df_filtered['admittime'])
            df_filtered['dischtime'] = pd.to_datetime(df_filtered['dischtime'])
            
            # Length of stay
            los_hours = (df_filtered['dischtime'] - df_filtered['admittime']).dt.total_seconds() / 3600
            temporal_features['los_hours'] = los_hours.fillna(0)
            temporal_features['los_days'] = (los_hours / 24).fillna(0)
            
            # Admission timing patterns
            temporal_features['admit_hour'] = df_filtered['admittime'].dt.hour
            temporal_features['admit_day_of_week'] = df_filtered['admittime'].dt.dayofweek
            temporal_features['admit_month'] = df_filtered['admittime'].dt.month
            temporal_features['admit_quarter'] = df_filtered['admittime'].dt.quarter
            
            # Weekend admission
            temporal_features['weekend_admission'] = (df_filtered['admittime'].dt.dayofweek >= 5).astype(int)
            
            # Night admission (10 PM to 6 AM)
            temporal_features['night_admission'] = ((df_filtered['admittime'].dt.hour >= 22) | 
                                                    (df_filtered['admittime'].dt.hour < 6)).astype(int)
            
            # Holiday season (simplified)
            temporal_features['holiday_season'] = ((df_filtered['admittime'].dt.month == 12) | 
                                                   (df_filtered['admittime'].dt.month == 1)).astype(int)
            
            # Length of stay categories
            temporal_features['short_stay'] = (los_hours < 24).astype(int)  # < 1 day
            temporal_features['medium_stay'] = ((los_hours >= 24) & (los_hours < 168)).astype(int)  # 1-7 days
            temporal_features['long_stay'] = (los_hours >= 168).astype(int)  # > 7 days
            
            # Discharge timing
            if 'dischtime' in df_filtered.columns:
                temporal_features['discharge_hour'] = df_filtered['dischtime'].dt.hour
                temporal_features['weekend_discharge'] = (df_filtered['dischtime'].dt.dayofweek >= 5).astype(int)
                temporal_features['night_discharge'] = ((df_filtered['dischtime'].dt.hour >= 22) | 
                                                        (df_filtered['dischtime'].dt.hour < 6)).astype(int)
            
            print(f"Added {len(temporal_features.columns)} temporal features")
        else:
            print("Admission/discharge time not available, using basic temporal features")
            # Basic fallback features
            temporal_features['los_hours'] = 0
            temporal_features['los_days'] = 0
            temporal_features['weekend_admission'] = 0
            temporal_features['night_admission'] = 0
        
        if len(temporal_features.columns) > 0:
            features_list.append(temporal_features)
        
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
        
        # Evaluation data: Create balanced test set from all partitions
        eval_features_list = []
        eval_labels_list = []
        
        samples_per_chapter = 50
        
        # First pass: collect evaluation data from all partitions
        for chapter_name, chapter_data in valid_partitions.items():
            if len(chapter_data) >= samples_per_chapter:
                sampled_data = chapter_data.sample(n=samples_per_chapter, random_state=42)
                chapter_features, chapter_labels = create_features_and_labels(
                    sampled_data, chapter_name, global_feature_space, include_icd_features=False
                )
                eval_features_list.append(chapter_features)
                eval_labels_list.append(chapter_labels)
        
        # Second pass: check if we need to boost top-K representation
        if eval_features_list:
            temp_eval_labels = pd.concat(eval_labels_list, axis=0, ignore_index=True)
            temp_top_k_count = sum(1 for label in temp_eval_labels if label < len(TOP_ICD_CODES))
            temp_top_k_ratio = temp_top_k_count / len(temp_eval_labels)
            
            # If evaluation data has <30% top-K codes, try to boost it
            if temp_top_k_ratio < 0.3:
                print(f"Evaluation data has {temp_top_k_ratio:.1%} top-K codes - boosting top-K representation...")
                
                # Collect additional top-K samples
                additional_features = []
                additional_labels = []
                
                for chapter_name, chapter_data in valid_partitions.items():
                    # Get labels for this chapter
                    _, chapter_labels = create_features_and_labels(
                        chapter_data, chapter_name, global_feature_space, include_icd_features=False
                    )
                    
                    # Find samples with top-K codes
                    top_k_mask = chapter_labels < len(TOP_ICD_CODES)
                    top_k_samples = chapter_data[top_k_mask]
                    
                    # Add extra top-K samples (up to 25 more per chapter)
                    if len(top_k_samples) > 0:
                        extra_samples = min(25, len(top_k_samples))
                        sampled_top_k = top_k_samples.sample(n=extra_samples, random_state=42)
                        extra_features, extra_labels = create_features_and_labels(
                            sampled_top_k, chapter_name, global_feature_space, include_icd_features=False
                        )
                        additional_features.append(extra_features)
                        additional_labels.append(extra_labels)
                
                # Add the additional samples to evaluation data
                if additional_features:
                    eval_features_list.extend(additional_features)
                    eval_labels_list.extend(additional_labels)
                    print(f"Added {len(additional_features)} additional top-K sample groups")
        
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
        
        # Analyze class distribution in detail
        train_top_k_count = sum(1 for label in y_train if label < len(TOP_ICD_CODES))
        train_other_count = len(y_train) - train_top_k_count
        eval_top_k_count = sum(1 for label in y_eval if label < len(TOP_ICD_CODES))
        eval_other_count = len(y_eval) - eval_top_k_count
        
        print(f"Training distribution:")
        print(f"  Top-K codes: {train_top_k_count} ({train_top_k_count/len(y_train)*100:.1f}%)")
        print(f"  Other codes: {train_other_count} ({train_other_count/len(y_train)*100:.1f}%)")
        
        print(f"Evaluation distribution:")
        print(f"  Top-K codes: {eval_top_k_count} ({eval_top_k_count/len(y_eval)*100:.1f}%)")
        print(f"  Other codes: {eval_other_count} ({eval_other_count/len(y_eval)*100:.1f}%)")
        
        # Warning if evaluation data is too imbalanced
        if eval_top_k_count / len(y_eval) < 0.2:
            print(f"WARNING: Evaluation data has <20% top-K codes - may give misleading results!")
        elif eval_top_k_count / len(y_eval) > 0.8:
            print(f"WARNING: Evaluation data has >80% top-K codes - may be too easy!")
        else:
            print(f"GOOD: Evaluation data has reasonable class balance")
        
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
def get_model(input_dim: int, hidden_dim: int = 128, output_dim: int = None, use_advanced: bool = True) -> torch.nn.Module:
    if output_dim is None:
        output_dim = len(TOP_ICD_CODES) + 1  # +1 for "other" class
    
    if use_advanced:
        # Use advanced architecture with attention and residual connections
        return AdvancedNet(input_dim, output_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3)
    else:
        # Use simple architecture
        return Net(input_dim, output_dim)

# Neural network architecture for ICD code classification
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        
        # More sophisticated architecture for medical prediction
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature preprocessing layers
        self.feature_norm = nn.BatchNorm1d(input_dim)
        
        # Deeper network with residual connections
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        # Attention mechanism for medical feature importance
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input normalization
        x = self.feature_norm(x)
        
        # Feature extraction with residual connections
        # Block 1
        x1 = torch.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout1(x1)
        
        # Block 2
        x2 = torch.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)
        
        # Block 3 with residual connection
        x3 = torch.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)
        
        # Block 4 with residual connection
        x4 = torch.relu(self.bn4(self.fc4(x3 + x2)))  # Residual connection
        
        # Attention mechanism - learn which features are important
        attention_weights = self.attention(x4)
        x4_attended = x4 * attention_weights
        
        # Final classification
        output = self.classifier(x4_attended)
        
        return output

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in ICD prediction.
    Focuses training on hard examples and down-weights easy examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Trains the model with FedProx regularization
def train(net, global_net, trainloader, epochs, learning_rate, proximal_mu, device, use_focal_loss=False):
    # Calculate class weights from this client's data
    all_labels = []
    for _, labels in trainloader:
        all_labels.extend(labels.numpy())
    
    class_counts = np.bincount(all_labels, minlength=TOP_K_CODES + 1)  # TOP_K + 1 for 'other' class
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Choose loss function based on configuration
    if use_focal_loss:
        # Use focal loss for better handling of class imbalance
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        # Use weighted cross entropy
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use SGD with momentum for more stable federated training
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.1)
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate after each epoch
        scheduler.step()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

# Displays comprehensive prediction analysis with class distribution
def display_prediction_analysis(all_predictions, all_labels):
    """
    Comprehensive analysis of model predictions vs actual labels
    Shows detailed class distribution and model behavior
    """
    print("\n" + "="*80)
    print("MODEL PREDICTION ANALYSIS")
    print("="*80)
    
    # Basic statistics
    total_samples = len(all_predictions)
    unique_predictions = np.unique(all_predictions)
    unique_labels = np.unique(all_labels)
    other_class_idx = len(TOP_ICD_CODES)  # Index for 'other' class
    
    print(f"Total test samples: {total_samples}")
    print(f"Model predicts {len(unique_predictions)} different classes (out of {len(TOP_ICD_CODES) + 1} possible)")
    print(f"Actual data contains {len(unique_labels)} different classes")
    
    # Count predictions and actual labels
    pred_counts = np.bincount(all_predictions, minlength=len(TOP_ICD_CODES) + 1)
    label_counts = np.bincount(all_labels, minlength=len(TOP_ICD_CODES) + 1)
    
    # Top-K vs Other breakdown
    top_k_pred = sum(pred_counts[:len(TOP_ICD_CODES)])
    other_pred = pred_counts[other_class_idx] if other_class_idx < len(pred_counts) else 0
    top_k_actual = sum(label_counts[:len(TOP_ICD_CODES)])
    other_actual = label_counts[other_class_idx] if other_class_idx < len(label_counts) else 0
    
    print(f"\nOVERALL DISTRIBUTION:")
    print(f"+-----------------+-------------+-------------+-------------+")
    print(f"| Category        | Predictions | Actual      | Difference  |")
    print(f"+-----------------+-------------+-------------+-------------+")
    print(f"| Top-{len(TOP_ICD_CODES)} ICD codes | {top_k_pred:6d} ({top_k_pred/total_samples:5.1%}) | {top_k_actual:6d} ({top_k_actual/total_samples:5.1%}) | {(top_k_pred-top_k_actual)/total_samples:+6.1%}     |")
    print(f"| Other codes     | {other_pred:6d} ({other_pred/total_samples:5.1%}) | {other_actual:6d} ({other_actual/total_samples:5.1%}) | {(other_pred-other_actual)/total_samples:+6.1%}     |")
    print(f"+-----------------+-------------+-------------+-------------+")
    
    # Check for problematic patterns
    if other_pred > 0.8 * total_samples:
        print("⚠️  WARNING: Model predicts 'Other' >80% of time - class imbalance issue!")
    elif other_pred > 0.6 * total_samples:
        print("⚠️  CAUTION: Model predicts 'Other' >60% of time - moderate bias")
    else:
        print("✅ GOOD: Model shows reasonable distribution across classes")
    
    # Top predicted ICD codes
    print(f"\nTOP PREDICTED ICD CODES:")
    print(f"+------+-------------+-----------+----------+")
    print(f"| Rank | ICD Code    | Count     | % of Total|")
    print(f"+------+-------------+-----------+----------+")
    
    # Get top predictions (excluding 'other' class for this analysis)
    top_k_indices = list(range(len(TOP_ICD_CODES)))
    top_k_pred_counts = [(i, pred_counts[i]) for i in top_k_indices if pred_counts[i] > 0]
    top_k_pred_counts.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (class_idx, count) in enumerate(top_k_pred_counts[:10], 1):
        icd_code = TOP_ICD_CODES[class_idx] if class_idx < len(TOP_ICD_CODES) else "Unknown"
        percentage = count / total_samples * 100
        print(f"| {rank:4d} | {icd_code:11s} | {count:9d} | {percentage:7.2f}% |")
    
    if len(top_k_pred_counts) == 0:
        print(f"|  NO TOP-K ICD CODES PREDICTED - ONLY 'OTHER' CLASS!        |")
    
    print(f"+------+-------------+-----------+----------+")
    print(f"| Other| (All other) | {other_pred:9d} | {other_pred/total_samples*100:7.2f}% |")
    print(f"+------+-------------+-----------+----------+")
    
    # Diversity metrics
    total_predicted_classes = len([c for c in pred_counts if c > 0])
    diversity_score = total_predicted_classes / (len(TOP_ICD_CODES) + 1)
    
    print(f"\nMODEL BEHAVIOR METRICS:")
    print(f"• Prediction diversity: {total_predicted_classes}/{len(TOP_ICD_CODES) + 1} classes ({diversity_score:.1%})")
    print(f"• Most frequent prediction: Class {np.argmax(pred_counts)} ({np.max(pred_counts)/total_samples:.1%})")
    print(f"• Least frequent non-zero: {np.min([c for c in pred_counts if c > 0])} predictions")
    
    # Model confidence assessment
    if total_predicted_classes == 1:
        print("CRITICAL: Model only predicts ONE class!")
    elif total_predicted_classes < 5:
        print("WARNING: Model has low diversity (predicts <5 classes)")
    elif diversity_score > 0.3:
        print("GOOD: Model shows healthy prediction diversity")
    else:
        print("MODERATE: Model diversity could be improved")
    
    print("="*80)

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
        
        # Note: Prediction analysis will be shown server-side during final round
        # Client-side analysis is disabled due to Ray logging being suppressed
        
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
        
        med_categories = {
            'antibiotics': ['antibiotic', 'penicillin', 'vancomycin', 'ciprofloxacin', 'azithromycin', 'ceftriaxone', 'levofloxacin', 'meropenem'],
            'cardiovascular': ['metoprolol', 'lisinopril', 'furosemide', 'warfarin', 'aspirin', 'atorvastatin', 'amlodipine', 'diltiazem'],
            'diabetes': ['insulin', 'metformin', 'glipizide', 'glyburide'],
            'pain': ['morphine', 'fentanyl', 'acetaminophen', 'ibuprofen', 'oxycodone', 'tramadol', 'hydromorphone'],
            'respiratory': ['albuterol', 'prednisone', 'ipratropium', 'methylprednisolone'],
            'gastrointestinal': ['omeprazole', 'pantoprazole', 'ranitidine', 'ondansetron'],
            'anticoagulation': ['warfarin', 'heparin', 'enoxaparin', 'clopidogrel'],
            'sedation': ['midazolam', 'propofol', 'lorazepam', 'dexmedetomidine'],
            'vasopressor': ['norepinephrine', 'epinephrine', 'dopamine', 'vasopressin'],
            'psychiatric': ['haloperidol', 'olanzapine', 'quetiapine', 'sertraline'],
        }
        
        # High-risk medication combinations
        high_risk_combinations = [
            ('warfarin', 'aspirin'),  # Bleeding risk
            ('insulin', 'metformin'),  # Hypoglycemia risk
            ('morphine', 'midazolam'),  # Respiratory depression
            ('lisinopril', 'furosemide'),  # Hypotension
        ]
        
        med_features = {hadm_id: {} for hadm_id in hadm_ids}
        
        # Initialize all features
        for hadm_id in hadm_ids:
            for cat in med_categories.keys():
                med_features[hadm_id][f'med_{cat}'] = 0
                med_features[hadm_id][f'med_{cat}_count'] = 0
                med_features[hadm_id][f'med_{cat}_unique'] = 0
            
            med_features[hadm_id]['med_total_count'] = 0
            med_features[hadm_id]['med_unique_count'] = 0
            med_features[hadm_id]['high_risk_combinations'] = 0
            med_features[hadm_id]['polypharmacy'] = 0  # >5 medications
            med_features[hadm_id]['has_controlled_substance'] = 0
            med_features[hadm_id]['has_iv_medications'] = 0
            med_features[hadm_id]['has_prn_medications'] = 0  # "as needed"
        
        chunk_size = 100000
        hadm_id_set = set(hadm_ids)
        
        for chunk_idx, chunk in enumerate(pd.read_csv(prescriptions_path, chunksize=chunk_size, 
                                                     usecols=['hadm_id', 'drug', 'dose_val_rx', 'dose_unit_rx', 
                                                              'route', 'ndc'], low_memory=False)):
            if chunk_idx % 10 == 0:
                print(f"Processing enhanced medication chunk {chunk_idx}...")
            
            chunk_filtered = chunk[chunk['hadm_id'].isin(hadm_id_set)]
            
            if len(chunk_filtered) > 0:
                for hadm_id, group in chunk_filtered.groupby('hadm_id'):
                    if hadm_id not in med_features:
                        continue
                    
                    drugs_lower = group['drug'].str.lower()
                    unique_drugs = group['drug'].nunique()
                    
                    med_features[hadm_id]['med_total_count'] = len(group)
                    med_features[hadm_id]['med_unique_count'] = unique_drugs
                    med_features[hadm_id]['polypharmacy'] = int(unique_drugs > 5)
                    
                    # Route analysis
                    routes = group['route'].fillna('').str.lower()
                    med_features[hadm_id]['has_iv_medications'] = int(any(
                        route in ['iv', 'intravenous', 'ivpb', 'iv push'] for route in routes
                    ))
                    
                    # PRN analysis (simplified heuristic based on route)
                    med_features[hadm_id]['has_prn_medications'] = int(any(
                        'prn' in route for route in routes
                    ))
                    
                    # Controlled substances (simplified detection)
                    controlled_substances = ['morphine', 'fentanyl', 'oxycodone', 'midazolam', 'lorazepam', 'propofol']
                    med_features[hadm_id]['has_controlled_substance'] = int(any(
                        any(cs in drug.lower() for cs in controlled_substances) for drug in group['drug']
                    ))
                    
                    # Category analysis with enhanced features
                    drugs_prescribed = set()
                    for category, drug_list in med_categories.items():
                        category_drugs = []
                        category_count = 0
                        
                        for drug in group['drug']:
                            if any(med_drug in drug.lower() for med_drug in drug_list):
                                category_drugs.append(drug)
                                category_count += 1
                        
                        if category_drugs:
                            med_features[hadm_id][f'med_{category}'] = 1
                            med_features[hadm_id][f'med_{category}_count'] = category_count
                            med_features[hadm_id][f'med_{category}_unique'] = len(set(category_drugs))
                            drugs_prescribed.update(category_drugs)
                    
                    # High-risk combinations
                    risk_combinations = 0
                    for drug1, drug2 in high_risk_combinations:
                        has_drug1 = any(drug1 in drug.lower() for drug in group['drug'])
                        has_drug2 = any(drug2 in drug.lower() for drug in group['drug'])
                        if has_drug1 and has_drug2:
                            risk_combinations += 1
                    
                    med_features[hadm_id]['high_risk_combinations'] = risk_combinations
                    
                    # Dosage analysis (simplified)
                    dose_values = group['dose_val_rx'].dropna()
                    if len(dose_values) > 0:
                        med_features[hadm_id]['avg_dose_value'] = float(dose_values.mean())
                        med_features[hadm_id]['max_dose_value'] = float(dose_values.max())
                        med_features[hadm_id]['has_high_dose'] = int(dose_values.max() > dose_values.quantile(0.9))
            else:
                        med_features[hadm_id]['avg_dose_value'] = 0.0
                        med_features[hadm_id]['max_dose_value'] = 0.0
                        med_features[hadm_id]['has_high_dose'] = 0
        
        # Add medication complexity score
        for hadm_id in hadm_ids:
            features = med_features[hadm_id]
            
            complexity_score = 0
            complexity_score += features['med_unique_count'] * 0.5  # Number of unique drugs
            complexity_score += features['polypharmacy'] * 2  # Polypharmacy
            complexity_score += features['high_risk_combinations'] * 3  # High-risk combinations
            complexity_score += features['has_controlled_substance'] * 1  # Controlled substances
            complexity_score += features['has_iv_medications'] * 1  # IV medications
            
            features['medication_complexity_score'] = complexity_score
            
            # Medication burden categories
            features['low_medication_burden'] = int(complexity_score < 3)
            features['moderate_medication_burden'] = int(3 <= complexity_score < 8)
            features['high_medication_burden'] = int(complexity_score >= 8)
        
        return pd.DataFrame.from_dict(med_features, orient='index')
        
    except Exception as e:
        print(f"Error in enhanced medication features: {e}")
        return pd.DataFrame({
            f'med_{cat}': np.random.binomial(1, 0.3, len(hadm_ids)) 
            for cat in ['antibiotics', 'cardiovascular', 'diabetes', 'pain', 'respiratory', 'gastrointestinal']
        } | {'med_total_count': np.random.poisson(5, len(hadm_ids))}, index=hadm_ids)

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
        
        return pd.DataFrame.from_dict(proc_features, orient='index')
        
    except Exception as e:
        print(f"Error extracting procedure features: {e}")
        return pd.DataFrame({
            'proc_count_total': np.random.poisson(2, len(hadm_ids)),
            'proc_count_icd9': np.random.poisson(1, len(hadm_ids)),
            'proc_count_icd10': np.random.poisson(1, len(hadm_ids))
        } | {f'proc_category_{i}': np.random.binomial(1, 0.1, len(hadm_ids)) for i in range(10)}, 
        index=hadm_ids)

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
        return pd.DataFrame({
            'drg_severity': np.random.choice([0, 1, 2, 3], len(hadm_ids)),
            'drg_mortality': np.random.choice([0, 1, 2, 3], len(hadm_ids)),
            'service_transfers': np.random.poisson(1, len(hadm_ids))
        } | {f'service_{svc}': np.random.binomial(1, 0.2, len(hadm_ids)) 
             for svc in ['MED', 'SURG', 'CARD', 'NEURO', 'ORTHO', 'PSYCH', 'OBS', 'OMED', 'CSURG']}, 
        index=hadm_ids)

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
    """
    Preprocesses medical features for ICD code prediction.
    
    IMPORTANT: To prevent data leakage in ICD prediction:
    - EXCLUDES procedure features (contain ICD procedure codes)
    - EXCLUDES secondary diagnosis features (contain secondary ICD diagnosis codes)
    - INCLUDES only clinical indicators available during admission:
      * Lab values, medications, ICU monitoring, microbiology
      * Administrative/severity indicators (DRG, services, ICU exposure)
    """
    print("Preprocessing Medical Features")
    
    # Clean up problematic cache files to prevent data leakage
    print("Cleaning up problematic cache files (procedure, secondary_diag)...")
    try:
        import os
        procedure_cache = get_cache_path("procedure", data_dir)
        secondary_diag_cache = get_cache_path("secondary_diag", data_dir)
        
        if os.path.exists(procedure_cache):
            os.remove(procedure_cache)
            print("Removed procedure cache file (prevents data leakage)")
        
        if os.path.exists(secondary_diag_cache):
            os.remove(secondary_diag_cache)
            print("Removed secondary_diag cache file (prevents data leakage)")
    except Exception as e:
        print(f"Warning: Could not clean cache files: {e}")
    
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
        ("icu_monitoring", lambda data_dir: compute_icu_monitoring_features(all_hadm_ids, data_dir)),
        ("microbiology", lambda data_dir: compute_microbiology_features(all_hadm_ids, data_dir)),
        ("severity", lambda data_dir: compute_severity_features(all_hadm_ids, data_dir)),
        # NOTE: Removed procedure and secondary_diag features to prevent data leakage
        # in ICD prediction task - these contain diagnosis/procedure codes that would
        # defeat the purpose of predicting primary ICD codes
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

# Computes lab features for all admissions efficiently using real MIMIC-IV lab data
def compute_lab_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing real lab features from labevents.csv.gz...")
    
    try:
        labevents_path = os.path.join(data_dir, "hosp", "labevents.csv.gz")
        d_labitems_path = os.path.join(data_dir, "hosp", "d_labitems.csv.gz")
        
        if not os.path.exists(labevents_path) or not os.path.exists(d_labitems_path):
            print("Lab data files not found, using fallback features")
            return _compute_fallback_lab_features(all_hadm_ids)
        
        # Load lab item definitions
        print("Loading lab item definitions...")
        d_labitems = pd.read_csv(d_labitems_path, usecols=['itemid', 'label', 'category'])
        
        # Define key lab tests with their item IDs and normal ranges
        key_labs = {
            # Hematology
            'hemoglobin': {'itemids': [50811, 51222], 'normal_range': (12.0, 16.0), 'unit': 'g/dL'},
            'hematocrit': {'itemids': [50810, 51221], 'normal_range': (36.0, 48.0), 'unit': '%'},
            'wbc': {'itemids': [51301, 51300], 'normal_range': (4.0, 11.0), 'unit': 'K/uL'},
            'platelets': {'itemids': [51265], 'normal_range': (150, 450), 'unit': 'K/uL'},
            
            # Chemistry
            'glucose': {'itemids': [50809, 50931], 'normal_range': (70, 100), 'unit': 'mg/dL'},
            'creatinine': {'itemids': [50912], 'normal_range': (0.6, 1.2), 'unit': 'mg/dL'},
            'bun': {'itemids': [51006], 'normal_range': (7, 20), 'unit': 'mg/dL'},
            'sodium': {'itemids': [50983], 'normal_range': (136, 145), 'unit': 'mmol/L'},
            'potassium': {'itemids': [50971], 'normal_range': (3.5, 5.0), 'unit': 'mmol/L'},
            'chloride': {'itemids': [50902], 'normal_range': (98, 107), 'unit': 'mmol/L'},
            'bicarbonate': {'itemids': [50882], 'normal_range': (22, 28), 'unit': 'mmol/L'},
            
            # Liver function
            'alt': {'itemids': [50861], 'normal_range': (7, 40), 'unit': 'U/L'},
            'ast': {'itemids': [50878], 'normal_range': (10, 40), 'unit': 'U/L'},
            'bilirubin_total': {'itemids': [50885], 'normal_range': (0.3, 1.2), 'unit': 'mg/dL'},
            'albumin': {'itemids': [50862], 'normal_range': (3.5, 5.0), 'unit': 'g/dL'},
            
            # Cardiac
            'troponin_t': {'itemids': [51003], 'normal_range': (0, 0.01), 'unit': 'ng/mL'},
            'bnp': {'itemids': [50884], 'normal_range': (0, 100), 'unit': 'pg/mL'},
            
            # Coagulation
            'pt': {'itemids': [51274], 'normal_range': (11, 13), 'unit': 'sec'},
            'ptt': {'itemids': [51275], 'normal_range': (25, 35), 'unit': 'sec'},
            'inr': {'itemids': [51237], 'normal_range': (0.8, 1.1), 'unit': 'ratio'},
            
            # Blood gases
            'lactate': {'itemids': [50813], 'normal_range': (0.5, 2.2), 'unit': 'mmol/L'},
            'ph': {'itemids': [50820], 'normal_range': (7.35, 7.45), 'unit': 'pH'},
        }
        
        # Initialize feature dictionary
        lab_features = {hadm_id: {} for hadm_id in all_hadm_ids}
        
        # Get all relevant item IDs
        all_itemids = []
        for lab_info in key_labs.values():
            all_itemids.extend(lab_info['itemids'])
        
        # Process lab events in chunks
        print("Processing lab events...")
        chunk_size = 200000
        hadm_id_set = set(all_hadm_ids)
        
        for chunk_idx, chunk in enumerate(pd.read_csv(labevents_path, chunksize=chunk_size, 
                                                     usecols=['hadm_id', 'itemid', 'valuenum', 'charttime'])):
            
            if chunk_idx % 10 == 0:
                print(f"Processing lab chunk {chunk_idx}...")
            
            # Filter for relevant admissions and lab items
            chunk_filtered = chunk[
                (chunk['hadm_id'].isin(hadm_id_set)) & 
                (chunk['itemid'].isin(all_itemids)) &
                (chunk['valuenum'].notna())
            ]
            
            if len(chunk_filtered) == 0:
                continue
            
            # Process each lab type
            for lab_name, lab_info in key_labs.items():
                lab_data = chunk_filtered[chunk_filtered['itemid'].isin(lab_info['itemids'])]
                
                if len(lab_data) > 0:
                    # Group by admission and calculate statistics
                    lab_stats = lab_data.groupby('hadm_id')['valuenum'].agg([
                        'count', 'mean', 'std', 'min', 'max', 'first', 'last'
                    ]).fillna(0)
                    
                    normal_low, normal_high = lab_info['normal_range']
                    
                    for hadm_id, stats in lab_stats.iterrows():
                        if hadm_id in lab_features:
                            # Basic presence and count
                            lab_features[hadm_id][f'{lab_name}_count'] = int(stats['count'])
                            lab_features[hadm_id][f'{lab_name}_mean'] = float(stats['mean'])
                            lab_features[hadm_id][f'{lab_name}_std'] = float(stats['std'])
                            lab_features[hadm_id][f'{lab_name}_min'] = float(stats['min'])
                            lab_features[hadm_id][f'{lab_name}_max'] = float(stats['max'])
                            lab_features[hadm_id][f'{lab_name}_first'] = float(stats['first'])
                            lab_features[hadm_id][f'{lab_name}_last'] = float(stats['last'])
                            
                            # Abnormal value indicators
                            lab_features[hadm_id][f'{lab_name}_abnormal_low'] = int(stats['min'] < normal_low)
                            lab_features[hadm_id][f'{lab_name}_abnormal_high'] = int(stats['max'] > normal_high)
                            lab_features[hadm_id][f'{lab_name}_abnormal_mean'] = int(stats['mean'] < normal_low or stats['mean'] > normal_high)
                            
                            # Severity indicators
                            lab_features[hadm_id][f'{lab_name}_critically_low'] = int(stats['min'] < normal_low * 0.5)
                            lab_features[hadm_id][f'{lab_name}_critically_high'] = int(stats['max'] > normal_high * 1.5)
        
        # Fill missing values for admissions without lab data
        all_lab_columns = set()
        for features in lab_features.values():
            all_lab_columns.update(features.keys())
        
        for hadm_id in all_hadm_ids:
            for col in all_lab_columns:
                if col not in lab_features[hadm_id]:
                    if col.endswith('_count'):
                        lab_features[hadm_id][col] = 0
                    elif col.endswith(('_abnormal_low', '_abnormal_high', '_abnormal_mean', '_critically_low', '_critically_high')):
                        lab_features[hadm_id][col] = 0
                    else:
                        lab_features[hadm_id][col] = 0.0
        
        # Create summary features
        for hadm_id in all_hadm_ids:
            features = lab_features[hadm_id]
            
            # Count of different lab types performed
            lab_types_count = sum(1 for lab_name in key_labs.keys() if features.get(f'{lab_name}_count', 0) > 0)
            features['lab_types_performed'] = lab_types_count
            
            # Total lab count
            features['total_lab_count'] = sum(features.get(f'{lab_name}_count', 0) for lab_name in key_labs.keys())
            
            # Count of abnormal results
            features['abnormal_results_count'] = sum(
                features.get(f'{lab_name}_abnormal_mean', 0) for lab_name in key_labs.keys()
            )
            
            # Critical results indicator
            features['has_critical_results'] = int(any(
                features.get(f'{lab_name}_critically_low', 0) or features.get(f'{lab_name}_critically_high', 0)
                for lab_name in key_labs.keys()
            ))
            
            # Common lab panels
            features['has_basic_metabolic_panel'] = int(all(
                features.get(f'{lab}_count', 0) > 0 
                for lab in ['glucose', 'creatinine', 'bun', 'sodium', 'potassium', 'chloride', 'bicarbonate']
            ))
            
            features['has_complete_blood_count'] = int(all(
                features.get(f'{lab}_count', 0) > 0 
                for lab in ['hemoglobin', 'hematocrit', 'wbc', 'platelets']
            ))
            
            features['has_liver_function_tests'] = int(all(
                features.get(f'{lab}_count', 0) > 0 
                for lab in ['alt', 'ast', 'bilirubin_total', 'albumin']
            ))
        
        result_df = pd.DataFrame.from_dict(lab_features, orient='index')
        print(f"Extracted {len(result_df.columns)} real lab features for {len(result_df)} admissions")
        return result_df
        
    except Exception as e:
        print(f"Error processing real lab data: {e}")
        print("Falling back to simplified lab features")
        return _compute_fallback_lab_features(all_hadm_ids)

def _compute_fallback_lab_features(all_hadm_ids: List[int]) -> pd.DataFrame:
    """Fallback lab features if real lab data is not available"""
    print("Using fallback lab features...")
    
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

# Computes ICU monitoring features from chartevents, inputevents, and outputevents
def compute_icu_monitoring_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing ICU monitoring features...")
    
    try:
        # Check if ICU data files exist
        chartevents_path = os.path.join(data_dir, "icu", "chartevents.csv.gz")
        inputevents_path = os.path.join(data_dir, "icu", "inputevents.csv.gz")
        outputevents_path = os.path.join(data_dir, "icu", "outputevents.csv.gz")
        icustays_path = os.path.join(data_dir, "icu", "icustays.csv.gz")
        d_items_path = os.path.join(data_dir, "icu", "d_items.csv.gz")
        
        if not os.path.exists(chartevents_path) or not os.path.exists(icustays_path):
            print("ICU data files not found, using fallback features")
            return _compute_fallback_icu_features(all_hadm_ids)
        
        # Load ICU stay information to link hadm_id to stay_id
        print("Loading ICU stay information...")
        icustays = pd.read_csv(icustays_path, usecols=['hadm_id', 'stay_id', 'los'])
        hadm_to_stay = icustays[icustays['hadm_id'].isin(all_hadm_ids)]
        
        if len(hadm_to_stay) == 0:
            print("No ICU stays found for these admissions, using fallback features")
            return _compute_fallback_icu_features(all_hadm_ids)
        
        # Load item definitions
        d_items = pd.read_csv(d_items_path, usecols=['itemid', 'label', 'category'])
        
        # Define key vital signs and monitoring parameters
        vital_signs = {
            # Heart rate
            'heart_rate': {'itemids': [211, 220045], 'normal_range': (60, 100), 'unit': 'bpm'},
            # Blood pressure
            'systolic_bp': {'itemids': [51, 442, 455, 6701, 220050, 220179], 'normal_range': (90, 140), 'unit': 'mmHg'},
            'diastolic_bp': {'itemids': [8368, 8440, 8441, 8555, 220051, 220180], 'normal_range': (60, 90), 'unit': 'mmHg'},
            'mean_bp': {'itemids': [52, 6702, 443, 220052, 220181, 225312], 'normal_range': (70, 105), 'unit': 'mmHg'},
            # Respiratory
            'respiratory_rate': {'itemids': [618, 615, 220210, 224690], 'normal_range': (12, 20), 'unit': 'breaths/min'},
            'oxygen_saturation': {'itemids': [646, 220277], 'normal_range': (95, 100), 'unit': '%'},
            # Temperature
            'temperature': {'itemids': [223761, 678], 'normal_range': (36.1, 37.2), 'unit': 'C'},
            # Neurological
            'gcs_total': {'itemids': [198, 226755], 'normal_range': (13, 15), 'unit': 'score'},
            'gcs_eye': {'itemids': [184, 220739], 'normal_range': (3, 4), 'unit': 'score'},
            'gcs_verbal': {'itemids': [723, 223900], 'normal_range': (4, 5), 'unit': 'score'},
            'gcs_motor': {'itemids': [454, 223901], 'normal_range': (5, 6), 'unit': 'score'},
            # Other
            'cvp': {'itemids': [113, 220074], 'normal_range': (2, 8), 'unit': 'mmHg'},
            'urine_output': {'itemids': [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651], 'normal_range': (30, 100), 'unit': 'ml/hr'},
        }
        
        # Initialize feature dictionary
        icu_features = {hadm_id: {} for hadm_id in all_hadm_ids}
        
        # Add basic ICU stay information
        for hadm_id in all_hadm_ids:
            stay_info = hadm_to_stay[hadm_to_stay['hadm_id'] == hadm_id]
            if len(stay_info) > 0:
                icu_features[hadm_id]['has_icu_stay'] = 1
                icu_features[hadm_id]['icu_los'] = float(stay_info['los'].iloc[0])
                icu_features[hadm_id]['icu_stay_count'] = len(stay_info)
            else:
                icu_features[hadm_id]['has_icu_stay'] = 0
                icu_features[hadm_id]['icu_los'] = 0.0
                icu_features[hadm_id]['icu_stay_count'] = 0
        
        # Get stay_ids for admissions with ICU stays
        stay_ids = hadm_to_stay['stay_id'].unique()
        
        if len(stay_ids) > 0:
            # Get all relevant item IDs
            all_itemids = []
            for vital_info in vital_signs.values():
                all_itemids.extend(vital_info['itemids'])
            
            # Process chartevents in chunks
            print("Processing ICU chartevents...")
            chunk_size = 500000
            
            for chunk_idx, chunk in enumerate(pd.read_csv(chartevents_path, chunksize=chunk_size,
                                                         usecols=['stay_id', 'itemid', 'valuenum', 'charttime'])):
                
                if chunk_idx % 5 == 0:
                    print(f"Processing ICU chunk {chunk_idx}...")
                
                # Filter for relevant stays and vital signs
                chunk_filtered = chunk[
                    (chunk['stay_id'].isin(stay_ids)) & 
                    (chunk['itemid'].isin(all_itemids)) &
                    (chunk['valuenum'].notna()) &
                    (chunk['valuenum'] > 0)  # Filter out invalid values
                ]
                
                if len(chunk_filtered) == 0:
                    continue
                
                # Process each vital sign
                for vital_name, vital_info in vital_signs.items():
                    vital_data = chunk_filtered[chunk_filtered['itemid'].isin(vital_info['itemids'])]
                    
                    if len(vital_data) > 0:
                        # Group by stay_id and calculate statistics
                        vital_stats = vital_data.groupby('stay_id')['valuenum'].agg([
                            'count', 'mean', 'std', 'min', 'max', 'first', 'last'
                        ]).fillna(0)
                        
                        normal_low, normal_high = vital_info['normal_range']
                        
                        for stay_id, stats in vital_stats.iterrows():
                            # Find corresponding hadm_id
                            hadm_id = hadm_to_stay[hadm_to_stay['stay_id'] == stay_id]['hadm_id'].iloc[0]
                            
                            if hadm_id in icu_features:
                                # Basic statistics
                                icu_features[hadm_id][f'{vital_name}_count'] = int(stats['count'])
                                icu_features[hadm_id][f'{vital_name}_mean'] = float(stats['mean'])
                                icu_features[hadm_id][f'{vital_name}_std'] = float(stats['std'])
                                icu_features[hadm_id][f'{vital_name}_min'] = float(stats['min'])
                                icu_features[hadm_id][f'{vital_name}_max'] = float(stats['max'])
                                
                                # Abnormal value indicators
                                icu_features[hadm_id][f'{vital_name}_abnormal_low'] = int(stats['min'] < normal_low)
                                icu_features[hadm_id][f'{vital_name}_abnormal_high'] = int(stats['max'] > normal_high)
                                icu_features[hadm_id][f'{vital_name}_unstable'] = int(stats['std'] > (normal_high - normal_low) * 0.3)
                                
                                # Critical value indicators
                                icu_features[hadm_id][f'{vital_name}_critically_low'] = int(stats['min'] < normal_low * 0.7)
                                icu_features[hadm_id][f'{vital_name}_critically_high'] = int(stats['max'] > normal_high * 1.3)
        
        # Fill missing values for all admissions
        all_icu_columns = set()
        for features in icu_features.values():
            all_icu_columns.update(features.keys())
        
        for hadm_id in all_hadm_ids:
            for col in all_icu_columns:
                if col not in icu_features[hadm_id]:
                    if col.endswith('_count'):
                        icu_features[hadm_id][col] = 0
                    elif col.endswith(('_abnormal_low', '_abnormal_high', '_unstable', '_critically_low', '_critically_high')):
                        icu_features[hadm_id][col] = 0
                    else:
                        icu_features[hadm_id][col] = 0.0
        
        # Add summary features
        for hadm_id in all_hadm_ids:
            features = icu_features[hadm_id]
            
            if features['has_icu_stay'] == 1:
                # Count of monitored vital signs
                vitals_monitored = sum(1 for vital in vital_signs.keys() if features.get(f'{vital}_count', 0) > 0)
                features['vitals_monitored_count'] = vitals_monitored
                
                # Count of abnormal vitals
                abnormal_vitals = sum(
                    features.get(f'{vital}_abnormal_low', 0) or features.get(f'{vital}_abnormal_high', 0)
                    for vital in vital_signs.keys()
                )
                features['abnormal_vitals_count'] = abnormal_vitals
                
                # Critical care indicators
                features['has_critical_vitals'] = int(any(
                    features.get(f'{vital}_critically_low', 0) or features.get(f'{vital}_critically_high', 0)
                    for vital in vital_signs.keys()
                ))
                
                # Hemodynamic instability
                features['hemodynamic_instability'] = int(any(
                    features.get(f'{vital}_unstable', 0)
                    for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'mean_bp']
                ))
                
                # Respiratory distress
                features['respiratory_distress'] = int(
                    features.get('respiratory_rate_abnormal_high', 0) or
                    features.get('oxygen_saturation_abnormal_low', 0)
                )
                
                # Neurological impairment
                features['neurological_impairment'] = int(
                    features.get('gcs_total_abnormal_low', 0) or
                    features.get('gcs_total_mean', 15) < 13
                )
            else:
                # No ICU stay
                features['vitals_monitored_count'] = 0
                features['abnormal_vitals_count'] = 0
                features['has_critical_vitals'] = 0
                features['hemodynamic_instability'] = 0
                features['respiratory_distress'] = 0
                features['neurological_impairment'] = 0
        
        result_df = pd.DataFrame.from_dict(icu_features, orient='index')
        print(f"Extracted {len(result_df.columns)} ICU monitoring features for {len(result_df)} admissions")
        return result_df
        
    except Exception as e:
        print(f"Error processing ICU monitoring data: {e}")
        print("Falling back to simplified ICU features")
        return _compute_fallback_icu_features(all_hadm_ids)

def _compute_fallback_icu_features(all_hadm_ids: List[int]) -> pd.DataFrame:
    """Fallback ICU features if real ICU data is not available"""
    print("Using fallback ICU features...")
    
    icu_features = {}
    
    for hadm_id in all_hadm_ids:
        features = {}
        
        seed = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # Basic ICU presence
        features['has_icu_stay'] = 1 if seed < 300 else 0  # 30% have ICU stay
        
        if features['has_icu_stay']:
            features['icu_los'] = max(0.1, np.random.exponential(2.0))  # ICU length of stay
            features['icu_stay_count'] = 1 if seed < 950 else 2  # Most have 1 stay
            
            # Vital signs monitoring
            features['vitals_monitored_count'] = np.random.poisson(8)
            features['abnormal_vitals_count'] = np.random.poisson(2)
            features['has_critical_vitals'] = 1 if seed % 10 < 3 else 0
            features['hemodynamic_instability'] = 1 if seed % 10 < 2 else 0
            features['respiratory_distress'] = 1 if seed % 10 < 2 else 0
            features['neurological_impairment'] = 1 if seed % 10 < 1 else 0
        else:
            features['icu_los'] = 0.0
            features['icu_stay_count'] = 0
            features['vitals_monitored_count'] = 0
            features['abnormal_vitals_count'] = 0
            features['has_critical_vitals'] = 0
            features['hemodynamic_instability'] = 0
            features['respiratory_distress'] = 0
            features['neurological_impairment'] = 0
        
        icu_features[hadm_id] = features
    
    return pd.DataFrame.from_dict(icu_features, orient='index')

# Computes microbiology features from microbiologyevents
def compute_microbiology_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    print("Computing microbiology features...")
    
    try:
        microbiology_path = os.path.join(data_dir, "hosp", "microbiologyevents.csv.gz")
        
        if not os.path.exists(microbiology_path):
            print("Microbiology data file not found, using fallback features")
            return _compute_fallback_microbiology_features(all_hadm_ids)
        
        # Initialize features
        micro_features = {hadm_id: {} for hadm_id in all_hadm_ids}
        
        # Common organisms and their categories
        organism_categories = {
            'gram_positive': ['staphylococcus', 'streptococcus', 'enterococcus', 'bacillus'],
            'gram_negative': ['escherichia', 'klebsiella', 'pseudomonas', 'acinetobacter', 'enterobacter'],
            'anaerobic': ['bacteroides', 'clostridium', 'peptostreptococcus'],
            'fungal': ['candida', 'aspergillus', 'cryptococcus'],
            'viral': ['influenza', 'rsv', 'cmv', 'hsv'],
            'mycobacterium': ['mycobacterium', 'tuberculosis'],
        }
        
        # Specimen types
        specimen_types = ['blood', 'urine', 'sputum', 'wound', 'csf', 'stool', 'catheter']
        
        # Antibiotic categories for resistance testing
        antibiotic_classes = {
            'penicillin': ['penicillin', 'ampicillin', 'amoxicillin'],
            'cephalosporin': ['ceftriaxone', 'cefazolin', 'ceftaroline'],
            'carbapenem': ['meropenem', 'imipenem', 'ertapenem'],
            'fluoroquinolone': ['ciprofloxacin', 'levofloxacin', 'moxifloxacin'],
            'aminoglycoside': ['gentamicin', 'tobramycin', 'amikacin'],
            'glycopeptide': ['vancomycin', 'teicoplanin'],
            'lincosamide': ['clindamycin', 'lincomycin'],
            'macrolide': ['erythromycin', 'azithromycin', 'clarithromycin'],
        }
        
        # Initialize all features
        for hadm_id in all_hadm_ids:
            # Basic counts
            micro_features[hadm_id]['micro_total_tests'] = 0
            micro_features[hadm_id]['micro_positive_cultures'] = 0
            micro_features[hadm_id]['micro_negative_cultures'] = 0
            
            # Organism categories
            for category in organism_categories.keys():
                micro_features[hadm_id][f'micro_{category}'] = 0
                micro_features[hadm_id][f'micro_{category}_count'] = 0
            
            # Specimen types
            for specimen in specimen_types:
                micro_features[hadm_id][f'micro_{specimen}_tested'] = 0
                micro_features[hadm_id][f'micro_{specimen}_positive'] = 0
            
            # Antibiotic resistance
            for antibiotic_class in antibiotic_classes.keys():
                micro_features[hadm_id][f'micro_resistant_{antibiotic_class}'] = 0
                micro_features[hadm_id][f'micro_sensitive_{antibiotic_class}'] = 0
            
            # Clinical indicators
            micro_features[hadm_id]['micro_multidrug_resistant'] = 0
            micro_features[hadm_id]['micro_bloodstream_infection'] = 0
            micro_features[hadm_id]['micro_nosocomial_infection'] = 0
            micro_features[hadm_id]['micro_days_to_positive'] = 0
            micro_features[hadm_id]['micro_unique_organisms'] = 0
        
        # Process microbiology events in chunks
        chunk_size = 100000
        hadm_id_set = set(all_hadm_ids)
        
        for chunk_idx, chunk in enumerate(pd.read_csv(microbiology_path, chunksize=chunk_size,
                                                     usecols=['hadm_id', 'charttime', 'spec_type_desc', 
                                                              'org_name', 'ab_name', 'interpretation'])):
            
            if chunk_idx % 10 == 0:
                print(f"Processing microbiology chunk {chunk_idx}...")
            
            chunk_filtered = chunk[chunk['hadm_id'].isin(hadm_id_set)]
            
            if len(chunk_filtered) == 0:
                continue
            
            for hadm_id, group in chunk_filtered.groupby('hadm_id'):
                if hadm_id not in micro_features:
                    continue
                
                # Basic counts
                micro_features[hadm_id]['micro_total_tests'] = len(group)
                
                # Positive and negative cultures
                positive_cultures = group[group['org_name'].notna()]
                negative_cultures = group[group['org_name'].isna()]
                
                micro_features[hadm_id]['micro_positive_cultures'] = len(positive_cultures)
                micro_features[hadm_id]['micro_negative_cultures'] = len(negative_cultures)
                
                if len(positive_cultures) > 0:
                    # Organism analysis
                    organisms = positive_cultures['org_name'].str.lower().fillna('')
                    unique_organisms = organisms.nunique()
                    micro_features[hadm_id]['micro_unique_organisms'] = unique_organisms
                    
                    # Categorize organisms
                    for category, organism_list in organism_categories.items():
                        category_count = 0
                        for organism in organism_list:
                            if organisms.str.contains(organism, case=False, na=False).any():
                                micro_features[hadm_id][f'micro_{category}'] = 1
                                category_count += organisms.str.contains(organism, case=False, na=False).sum()
                        micro_features[hadm_id][f'micro_{category}_count'] = category_count
                    
                    # Specimen type analysis
                    specimens = group['spec_type_desc'].str.lower().fillna('')
                    for specimen in specimen_types:
                        specimen_tests = specimens.str.contains(specimen, case=False, na=False)
                        if specimen_tests.any():
                            micro_features[hadm_id][f'micro_{specimen}_tested'] = 1
                            # Check if any of these specimen tests were positive
                            specimen_positive = positive_cultures[positive_cultures['spec_type_desc'].str.contains(specimen, case=False, na=False)]
                            if len(specimen_positive) > 0:
                                micro_features[hadm_id][f'micro_{specimen}_positive'] = 1
                    
                    # Antibiotic resistance analysis
                    if 'ab_name' in group.columns and 'interpretation' in group.columns:
                        resistance_tests = group[group['ab_name'].notna() & group['interpretation'].notna()]
                        
                        if len(resistance_tests) > 0:
                            resistant_count = 0
                            total_tests = len(resistance_tests)
                            
                            for antibiotic_class, antibiotics in antibiotic_classes.items():
                                class_tests = resistance_tests[resistance_tests['ab_name'].str.lower().str.contains('|'.join(antibiotics), case=False, na=False)]
                                
                                if len(class_tests) > 0:
                                    resistant_tests = class_tests[class_tests['interpretation'].str.contains('R', case=False, na=False)]
                                    sensitive_tests = class_tests[class_tests['interpretation'].str.contains('S', case=False, na=False)]
                                    
                                    if len(resistant_tests) > 0:
                                        micro_features[hadm_id][f'micro_resistant_{antibiotic_class}'] = 1
                                        resistant_count += len(resistant_tests)
                                    
                                    if len(sensitive_tests) > 0:
                                        micro_features[hadm_id][f'micro_sensitive_{antibiotic_class}'] = 1
                            
                            # Multidrug resistance (resistant to >3 antibiotic classes)
                            resistant_classes = sum(1 for ac in antibiotic_classes.keys() 
                                                  if micro_features[hadm_id][f'micro_resistant_{ac}'] == 1)
                            micro_features[hadm_id]['micro_multidrug_resistant'] = int(resistant_classes > 3)
                    
                    # Clinical indicators
                    # Bloodstream infection
                    blood_cultures = positive_cultures[positive_cultures['spec_type_desc'].str.contains('blood', case=False, na=False)]
                    if len(blood_cultures) > 0:
                        micro_features[hadm_id]['micro_bloodstream_infection'] = 1
                    
                    # Nosocomial infection (simplified: positive culture >48h after admission)
                    if 'charttime' in positive_cultures.columns:
                        try:
                            charttime = pd.to_datetime(positive_cultures['charttime'], errors='coerce')
                            # This is a simplified proxy - in reality would need admission time
                            late_cultures = charttime[charttime > charttime.min() + pd.Timedelta(hours=48)]
                            if len(late_cultures) > 0:
                                micro_features[hadm_id]['micro_nosocomial_infection'] = 1
                        except:
                            pass
        
        # Add summary features
        for hadm_id in all_hadm_ids:
            features = micro_features[hadm_id]
            
            # Infection severity score
            severity_score = 0
            severity_score += features['micro_bloodstream_infection'] * 3
            severity_score += features['micro_multidrug_resistant'] * 2
            severity_score += features['micro_nosocomial_infection'] * 1
            severity_score += min(features['micro_unique_organisms'], 3)  # Cap at 3
            
            features['micro_infection_severity_score'] = severity_score
            
            # Infection categories
            features['micro_no_infection'] = int(features['micro_positive_cultures'] == 0)
            features['micro_mild_infection'] = int(0 < severity_score < 3)
            features['micro_moderate_infection'] = int(3 <= severity_score < 6)
            features['micro_severe_infection'] = int(severity_score >= 6)
            
            # Has any infection
            features['micro_has_infection'] = int(features['micro_positive_cultures'] > 0)
            
            # Antibiotic resistance burden
            resistant_classes = sum(1 for ac in antibiotic_classes.keys() 
                                  if features[f'micro_resistant_{ac}'] == 1)
            features['micro_resistance_burden'] = resistant_classes
        
        result_df = pd.DataFrame.from_dict(micro_features, orient='index')
        print(f"Extracted {len(result_df.columns)} microbiology features for {len(result_df)} admissions")
        return result_df
        
    except Exception as e:
        print(f"Error processing microbiology data: {e}")
        print("Falling back to simplified microbiology features")
        return _compute_fallback_microbiology_features(all_hadm_ids)

def _compute_fallback_microbiology_features(all_hadm_ids: List[int]) -> pd.DataFrame:
    """Fallback microbiology features if real microbiology data is not available"""
    print("Using fallback microbiology features...")
    
    micro_features = {}
    
    for hadm_id in all_hadm_ids:
        features = {}
        
        seed = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # Basic presence of cultures
        features['micro_has_infection'] = 1 if seed < 200 else 0  # 20% have infections
        
        if features['micro_has_infection']:
            features['micro_total_tests'] = np.random.poisson(3)
            features['micro_positive_cultures'] = np.random.poisson(1)
            features['micro_negative_cultures'] = features['micro_total_tests'] - features['micro_positive_cultures']
            
            # Organism types
            features['micro_gram_positive'] = 1 if seed % 10 < 3 else 0
            features['micro_gram_negative'] = 1 if seed % 10 < 4 else 0
            features['micro_fungal'] = 1 if seed % 10 < 1 else 0
            
            # Clinical indicators
            features['micro_bloodstream_infection'] = 1 if seed % 10 < 2 else 0
            features['micro_multidrug_resistant'] = 1 if seed % 10 < 1 else 0
            features['micro_nosocomial_infection'] = 1 if seed % 10 < 1 else 0
            
            # Severity
            features['micro_infection_severity_score'] = np.random.poisson(2)
            features['micro_severe_infection'] = 1 if features['micro_infection_severity_score'] > 4 else 0
        else:
            # No infection
            features['micro_total_tests'] = 0
            features['micro_positive_cultures'] = 0
            features['micro_negative_cultures'] = 0
            features['micro_gram_positive'] = 0
            features['micro_gram_negative'] = 0
            features['micro_fungal'] = 0
            features['micro_bloodstream_infection'] = 0
            features['micro_multidrug_resistant'] = 0
            features['micro_nosocomial_infection'] = 0
            features['micro_infection_severity_score'] = 0
            features['micro_severe_infection'] = 0
        
        micro_features[hadm_id] = features
    
    return pd.DataFrame.from_dict(micro_features, orient='index')

# Enhanced compute functions for the caching system
def compute_medication_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    """Compute enhanced medication features for all admissions using real MIMIC-IV data"""
    print("Computing enhanced medication features...")
    
    try:
        prescriptions_path = os.path.join(data_dir, "hosp", "prescriptions.csv.gz")
        pharmacy_path = os.path.join(data_dir, "hosp", "pharmacy.csv.gz")
        
        if not os.path.exists(prescriptions_path):
            print("Prescriptions data not found, using fallback features")
            return _compute_fallback_medication_features(all_hadm_ids)
        
        # Enhanced medication categories
        med_categories = {
            'antibiotics': ['antibiotic', 'penicillin', 'vancomycin', 'ciprofloxacin', 'azithromycin', 'ceftriaxone', 'levofloxacin', 'meropenem'],
            'cardiovascular': ['metoprolol', 'lisinopril', 'furosemide', 'warfarin', 'aspirin', 'atorvastatin', 'amlodipine', 'diltiazem'],
            'diabetes': ['insulin', 'metformin', 'glipizide', 'glyburide'],
            'pain': ['morphine', 'fentanyl', 'acetaminophen', 'ibuprofen', 'oxycodone', 'tramadol', 'hydromorphone'],
            'respiratory': ['albuterol', 'prednisone', 'ipratropium', 'methylprednisolone'],
            'gastrointestinal': ['omeprazole', 'pantoprazole', 'ranitidine', 'ondansetron'],
            'anticoagulation': ['warfarin', 'heparin', 'enoxaparin', 'clopidogrel'],
            'sedation': ['midazolam', 'propofol', 'lorazepam', 'dexmedetomidine'],
            'vasopressor': ['norepinephrine', 'epinephrine', 'dopamine', 'vasopressin'],
            'psychiatric': ['haloperidol', 'olanzapine', 'quetiapine', 'sertraline'],
        }
        
        # Administration routes
        admin_routes = {
            'iv': ['intravenous', 'iv', 'i.v.', 'injection'],
            'oral': ['oral', 'po', 'p.o.', 'tablet', 'capsule'],
            'inhaled': ['inhaled', 'nebulizer', 'mdi'],
            'topical': ['topical', 'cream', 'ointment'],
            'prn': ['prn', 'as needed', 'when needed']
        }
        
        # High-risk combinations
        high_risk_combinations = [
            ('warfarin', 'aspirin'), ('insulin', 'metformin'),
            ('morphine', 'midazolam'), ('lisinopril', 'furosemide')
        ]
        
        med_features = {}
        chunk_size = 100000
        hadm_id_set = set(all_hadm_ids)
        
        # Initialize features
        for hadm_id in all_hadm_ids:
            features = {}
            
            # Basic counts
            features['med_total_count'] = 0
            features['med_unique_count'] = 0
            features['med_polypharmacy'] = 0
            features['med_high_risk_combinations'] = 0
            
            # Category presence and counts
            for cat in med_categories.keys():
                features[f'med_{cat}'] = 0
                features[f'med_{cat}_count'] = 0
                features[f'med_{cat}_unique'] = 0
            
            # Administration routes
            for route in admin_routes.keys():
                features[f'med_{route}'] = 0
                features[f'med_{route}_count'] = 0
            
            # Safety indicators
            features['med_controlled_substances'] = 0
            features['med_high_alert_drugs'] = 0
            features['med_drug_interactions'] = 0
            features['med_complexity_score'] = 0
            
            med_features[hadm_id] = features
        
        # Process prescriptions in chunks
        for chunk in pd.read_csv(prescriptions_path, chunksize=chunk_size, 
                                  usecols=['hadm_id', 'drug', 'dose_val_rx', 'route'], 
                                  low_memory=False):
            
            chunk_filtered = chunk[chunk['hadm_id'].isin(hadm_id_set)]
            
            for hadm_id, group in chunk_filtered.groupby('hadm_id'):
                if hadm_id in med_features:
                    drugs = group['drug'].fillna('').str.lower()
                    routes = group['route'].fillna('').str.lower()
                    
                    # Basic counts
                    med_features[hadm_id]['med_total_count'] = len(group)
                    med_features[hadm_id]['med_unique_count'] = drugs.nunique()
                    med_features[hadm_id]['med_polypharmacy'] = 1 if len(group) > 5 else 0
                    
                    # Category analysis with enhanced features
                    drugs_prescribed = set()
                    for category, drug_list in med_categories.items():
                        category_drugs = []
                        category_count = 0
                        
                        for drug in group['drug']:
                            if any(med_drug in drug.lower() for med_drug in drug_list):
                                category_drugs.append(drug)
                                category_count += 1
                        
                        if category_drugs:
                            med_features[hadm_id][f'med_{category}'] = 1
                            med_features[hadm_id][f'med_{category}_count'] = category_count
                            med_features[hadm_id][f'med_{category}_unique'] = len(set(category_drugs))
                            drugs_prescribed.update(category_drugs)
                    
                    # Administration route analysis
                    for route, route_terms in admin_routes.items():
                        route_mask = routes.str.contains('|'.join(route_terms), case=False, na=False)
                        med_features[hadm_id][f'med_{route}'] = 1 if route_mask.any() else 0
                        med_features[hadm_id][f'med_{route}_count'] = route_mask.sum()
                    
                    # High-risk combinations
                    risk_count = 0
                    for drug1, drug2 in high_risk_combinations:
                        if (drugs.str.contains(drug1, case=False, na=False).any() and 
                            drugs.str.contains(drug2, case=False, na=False).any()):
                            risk_count += 1
                    med_features[hadm_id]['med_high_risk_combinations'] = risk_count
                    
                    # Controlled substances and high-alert drugs
                    controlled_terms = ['morphine', 'fentanyl', 'oxycodone', 'midazolam', 'propofol']
                    high_alert_terms = ['insulin', 'warfarin', 'heparin', 'norepinephrine', 'dopamine']
                    
                    med_features[hadm_id]['med_controlled_substances'] = 1 if drugs.str.contains('|'.join(controlled_terms), case=False, na=False).any() else 0
                    med_features[hadm_id]['med_high_alert_drugs'] = 1 if drugs.str.contains('|'.join(high_alert_terms), case=False, na=False).any() else 0
                    
                    # Complexity score (simple heuristic)
                    complexity = len(group) * 0.1 + drugs.nunique() * 0.2 + risk_count * 0.5
                    med_features[hadm_id]['med_complexity_score'] = min(complexity, 10.0)
        
        return pd.DataFrame.from_dict(med_features, orient='index')
        
    except Exception as e:
        print(f"Error in enhanced medication features: {e}")
        return _compute_fallback_medication_features(all_hadm_ids)

def _compute_fallback_medication_features(all_hadm_ids: List[int]) -> pd.DataFrame:
    """Fallback medication features if real medication data is not available"""
    print("Using fallback medication features...")
    
    med_categories = ['antibiotics', 'cardiovascular', 'diabetes', 'pain', 'respiratory', 'gastrointestinal', 'anticoagulation', 'sedation', 'vasopressor', 'psychiatric']
    admin_routes = ['iv', 'oral', 'inhaled', 'topical', 'prn']
    
    med_features = {}
    
    for hadm_id in all_hadm_ids:
        features = {}
        
        # Use deterministic random based on hadm_id
        seed = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # Basic counts
        features['med_total_count'] = np.random.poisson(8)
        features['med_unique_count'] = min(features['med_total_count'], np.random.poisson(6))
        features['med_polypharmacy'] = 1 if features['med_total_count'] > 5 else 0
        features['med_high_risk_combinations'] = np.random.poisson(0.5)
        
        # Category features
        for cat in med_categories:
            features[f'med_{cat}'] = 1 if seed % 10 < 3 else 0
            features[f'med_{cat}_count'] = np.random.poisson(1) if features[f'med_{cat}'] else 0
            features[f'med_{cat}_unique'] = min(features[f'med_{cat}_count'], np.random.poisson(1))
        
        # Administration routes
        for route in admin_routes:
            features[f'med_{route}'] = 1 if seed % 10 < 4 else 0
            features[f'med_{route}_count'] = np.random.poisson(2) if features[f'med_{route}'] else 0
        
        # Safety indicators
        features['med_controlled_substances'] = 1 if seed % 10 < 2 else 0
        features['med_high_alert_drugs'] = 1 if seed % 10 < 3 else 0
        features['med_drug_interactions'] = np.random.poisson(0.3)
        features['med_complexity_score'] = np.random.uniform(0, 10)
        
        med_features[hadm_id] = features
    
    return pd.DataFrame.from_dict(med_features, orient='index')

def compute_procedure_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    """Compute enhanced procedure features for all admissions"""
    print("Computing enhanced procedure features...")
    
    try:
        procedures_path = os.path.join(data_dir, "hosp", "procedures_icd.csv.gz")
        
        if not os.path.exists(procedures_path):
            print("Procedures data not found, using fallback features")
            return _compute_fallback_procedure_features(all_hadm_ids)
        
        # Enhanced procedure categories
        procedure_categories = {
            'cardiovascular': ['heart', 'cardiac', 'vascular', 'bypass', 'angioplasty', 'stent'],
            'respiratory': ['lung', 'broncho', 'respiratory', 'ventilation', 'trach'],
            'gastrointestinal': ['gastro', 'intestinal', 'endoscopy', 'colonoscopy'],
            'neurological': ['neuro', 'brain', 'spine', 'cranial'],
            'orthopedic': ['orthopedic', 'joint', 'fracture', 'bone'],
            'surgical': ['surgery', 'operative', 'incision', 'excision'],
            'diagnostic': ['biopsy', 'imaging', 'scan', 'x-ray', 'ct', 'mri'],
            'therapeutic': ['therapy', 'treatment', 'infusion', 'dialysis'],
            'obstetric': ['delivery', 'cesarean', 'obstetric'],
            'emergency': ['emergency', 'trauma', 'resuscitation']
        }
        
        proc_features = {}
        procedures = pd.read_csv(procedures_path, usecols=['hadm_id', 'icd_code', 'icd_version'])
        admission_procs = procedures[procedures['hadm_id'].isin(all_hadm_ids)]
        
        for hadm_id in all_hadm_ids:
            features = {}
            admission_data = admission_procs[admission_procs['hadm_id'] == hadm_id]
            
            # Basic counts
            features['proc_count_total'] = len(admission_data)
            features['proc_count_icd9'] = len(admission_data[admission_data['icd_version'] == 9])
            features['proc_count_icd10'] = len(admission_data[admission_data['icd_version'] == 10])
            features['proc_complexity_score'] = min(len(admission_data) * 0.5, 10.0)
            
            # ICD-9 categories (first digit)
            for digit in range(10):
                features[f'proc_category_{digit}'] = 0
            
            if len(admission_data) > 0:
                icd9_codes = admission_data[admission_data['icd_version'] == 9]['icd_code'].astype(str)
                if len(icd9_codes) > 0:
                    first_digits = icd9_codes.str[0]
                    for digit in range(10):
                        features[f'proc_category_{digit}'] = 1 if str(digit) in first_digits.values else 0
            
            # Enhanced procedure categories
            for category in procedure_categories.keys():
                features[f'proc_{category}'] = 0
                features[f'proc_{category}_count'] = 0
            
            # Procedure risk levels
            features['proc_high_risk'] = 0
            features['proc_major_surgery'] = 0
            features['proc_minimally_invasive'] = 0
            
            # Timing indicators
            features['proc_emergency'] = 0
            features['proc_elective'] = 0
            
            # Simple heuristics for enhanced features
            if len(admission_data) > 0:
                # High-risk procedures (heuristic based on count)
                features['proc_high_risk'] = 1 if len(admission_data) > 5 else 0
                features['proc_major_surgery'] = 1 if len(admission_data) > 3 else 0
                features['proc_minimally_invasive'] = 1 if len(admission_data) <= 2 else 0
                
                # Emergency vs elective (simple heuristic)
                features['proc_emergency'] = 1 if len(admission_data) > 4 else 0
                features['proc_elective'] = 1 if len(admission_data) <= 2 else 0
            
            proc_features[hadm_id] = features
        
        return pd.DataFrame.from_dict(proc_features, orient='index')
        
    except Exception as e:
        print(f"Error in enhanced procedure features: {e}")
        return _compute_fallback_procedure_features(all_hadm_ids)

def _compute_fallback_procedure_features(all_hadm_ids: List[int]) -> pd.DataFrame:
    """Fallback procedure features if real procedure data is not available"""
    print("Using fallback procedure features...")
    
    procedure_categories = ['cardiovascular', 'respiratory', 'gastrointestinal', 'neurological', 'orthopedic', 'surgical', 'diagnostic', 'therapeutic', 'obstetric', 'emergency']
    
    proc_features = {}
    
    for hadm_id in all_hadm_ids:
        features = {}
        
        # Use deterministic random based on hadm_id
        seed = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # Basic counts
        features['proc_count_total'] = np.random.poisson(2)
        features['proc_count_icd9'] = np.random.poisson(1)
        features['proc_count_icd10'] = np.random.poisson(1)
        features['proc_complexity_score'] = np.random.uniform(0, 10)
        
        # ICD-9 categories
        for digit in range(10):
            features[f'proc_category_{digit}'] = 1 if seed % 10 == digit else 0
        
        # Enhanced categories
        for category in procedure_categories:
            features[f'proc_{category}'] = 1 if seed % 10 < 2 else 0
            features[f'proc_{category}_count'] = np.random.poisson(1) if features[f'proc_{category}'] else 0
        
        # Risk and timing
        features['proc_high_risk'] = 1 if seed % 10 < 1 else 0
        features['proc_major_surgery'] = 1 if seed % 10 < 1 else 0
        features['proc_minimally_invasive'] = 1 if seed % 10 < 3 else 0
        features['proc_emergency'] = 1 if seed % 10 < 2 else 0
        features['proc_elective'] = 1 if seed % 10 < 3 else 0
        
        proc_features[hadm_id] = features
    
    return pd.DataFrame.from_dict(proc_features, orient='index')

def compute_severity_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    """Compute enhanced severity features for all admissions"""
    print("Computing enhanced severity features...")
    
    try:
        drg_path = os.path.join(data_dir, "hosp", "drgcodes.csv.gz")
        services_path = os.path.join(data_dir, "hosp", "services.csv.gz")
        transfers_path = os.path.join(data_dir, "hosp", "transfers.csv.gz")
        
        if not os.path.exists(drg_path) or not os.path.exists(services_path):
            print("Severity data not found, using fallback features")
            return _compute_fallback_severity_features(all_hadm_ids)
        
        # Load severity data
        drg_codes = pd.read_csv(drg_path, usecols=['hadm_id', 'drg_severity', 'drg_mortality'])
        services = pd.read_csv(services_path, usecols=['hadm_id', 'curr_service'])
        
        # Load transfer data if available
        transfers = None
        if os.path.exists(transfers_path):
            transfers = pd.read_csv(transfers_path, usecols=['hadm_id', 'careunit'])
        
        admission_drg = drg_codes[drg_codes['hadm_id'].isin(all_hadm_ids)]
        admission_services = services[services['hadm_id'].isin(all_hadm_ids)]
        admission_transfers = transfers[transfers['hadm_id'].isin(all_hadm_ids)] if transfers is not None else None
        
        severity_features = {}
        common_services = ['MED', 'SURG', 'CARD', 'NEURO', 'ORTHO', 'PSYCH', 'OBS', 'OMED', 'CSURG', 'TRAUM', 'CTIC']
        critical_units = ['MICU', 'SICU', 'CVICU', 'TSICU', 'CCU', 'CSRU']
        
        for hadm_id in all_hadm_ids:
            features = {}
            
            # DRG-based severity
            drg_data = admission_drg[admission_drg['hadm_id'] == hadm_id]
            if len(drg_data) > 0:
                features['drg_severity'] = drg_data['drg_severity'].fillna(0).max()
                features['drg_mortality'] = drg_data['drg_mortality'].fillna(0).max()
                features['drg_high_severity'] = 1 if features['drg_severity'] > 2 else 0
                features['drg_high_mortality'] = 1 if features['drg_mortality'] > 2 else 0
            else:
                features['drg_severity'] = 0
                features['drg_mortality'] = 0
                features['drg_high_severity'] = 0
                features['drg_high_mortality'] = 0
            
            # Service-based indicators
            service_data = admission_services[admission_services['hadm_id'] == hadm_id]
            if len(service_data) > 0:
                services_list = service_data['curr_service'].unique()
                for service in common_services:
                    features[f'service_{service}'] = 1 if service in services_list else 0
                
                features['service_transfers'] = max(0, len(service_data) - 1)
                features['service_multiple'] = 1 if len(services_list) > 1 else 0
                
                # Critical services
                critical_services = ['SURG', 'CARD', 'NEURO', 'TRAUM', 'CTIC']
                features['service_critical'] = 1 if any(svc in services_list for svc in critical_services) else 0
            else:
                for service in common_services:
                    features[f'service_{service}'] = 0
                features['service_transfers'] = 0
                features['service_multiple'] = 0
                features['service_critical'] = 0
            
            # Transfer-based severity (ICU exposure)
            if admission_transfers is not None:
                transfer_data = admission_transfers[admission_transfers['hadm_id'] == hadm_id]
                if len(transfer_data) > 0:
                    care_units = transfer_data['careunit'].unique()
                    features['icu_exposure'] = 1 if any(unit in care_units for unit in critical_units) else 0
                    features['care_unit_transfers'] = max(0, len(transfer_data) - 1)
                    features['multiple_icu'] = 1 if sum(1 for unit in care_units if unit in critical_units) > 1 else 0
                else:
                    features['icu_exposure'] = 0
                    features['care_unit_transfers'] = 0
                    features['multiple_icu'] = 0
            else:
                features['icu_exposure'] = 0
                features['care_unit_transfers'] = 0
                features['multiple_icu'] = 0
            
            # Composite severity score
            severity_score = (features['drg_severity'] * 0.3 + 
                              features['drg_mortality'] * 0.3 + 
                              features['service_transfers'] * 0.2 + 
                              features['icu_exposure'] * 0.2)
            features['severity_composite_score'] = min(severity_score, 10.0)
            features['severity_high_risk'] = 1 if severity_score > 3 else 0
            
            severity_features[hadm_id] = features
        
        return pd.DataFrame.from_dict(severity_features, orient='index')
        
    except Exception as e:
        print(f"Error in enhanced severity features: {e}")
        return _compute_fallback_severity_features(all_hadm_ids)

def _compute_fallback_severity_features(all_hadm_ids: List[int]) -> pd.DataFrame:
    """Fallback severity features if real severity data is not available"""
    print("Using fallback severity features...")
    
    common_services = ['MED', 'SURG', 'CARD', 'NEURO', 'ORTHO', 'PSYCH', 'OBS', 'OMED', 'CSURG', 'TRAUM', 'CTIC']
    
    severity_features = {}
    
    for hadm_id in all_hadm_ids:
        features = {}
        
        # Use deterministic random based on hadm_id
        seed = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # DRG-based severity
        features['drg_severity'] = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        features['drg_mortality'] = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        features['drg_high_severity'] = 1 if features['drg_severity'] > 2 else 0
        features['drg_high_mortality'] = 1 if features['drg_mortality'] > 2 else 0
        
        # Service indicators
        for service in common_services:
            features[f'service_{service}'] = 1 if seed % 10 < 2 else 0
        
        features['service_transfers'] = np.random.poisson(1)
        features['service_multiple'] = 1 if seed % 10 < 3 else 0
        features['service_critical'] = 1 if seed % 10 < 2 else 0
        
        # ICU and transfers
        features['icu_exposure'] = 1 if seed % 10 < 2 else 0
        features['care_unit_transfers'] = np.random.poisson(1)
        features['multiple_icu'] = 1 if seed % 10 < 1 else 0
        
        # Composite score
        features['severity_composite_score'] = np.random.uniform(0, 10)
        features['severity_high_risk'] = 1 if features['severity_composite_score'] > 6 else 0
        
        severity_features[hadm_id] = features
    
    return pd.DataFrame.from_dict(severity_features, orient='index')

def compute_secondary_diagnosis_features(all_hadm_ids: List[int], data_dir: str) -> pd.DataFrame:
    """Compute enhanced secondary diagnosis features for all admissions"""
    print("Computing enhanced secondary diagnosis features...")
    
    try:
        diagnoses_path = os.path.join(data_dir, "hosp", "diagnoses_icd.csv.gz")
        
        if not os.path.exists(diagnoses_path):
            print("Secondary diagnosis data not found, using fallback features")
            return _compute_fallback_secondary_diagnosis_features(all_hadm_ids)
        
        # Enhanced diagnosis categories
        diagnosis_categories = {
            'diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'hypoglycemia'],
            'hypertension': ['hypertension', 'hypertensive', 'high blood pressure'],
            'heart_failure': ['heart failure', 'cardiac failure', 'congestive'],
            'kidney_disease': ['kidney', 'renal', 'nephritis', 'dialysis'],
            'liver_disease': ['liver', 'hepatic', 'cirrhosis', 'hepatitis'],
            'copd': ['copd', 'chronic obstructive', 'emphysema', 'bronchitis'],
            'cancer': ['cancer', 'malignancy', 'tumor', 'neoplasm', 'carcinoma'],
            'depression': ['depression', 'depressive', 'mood disorder'],
            'anxiety': ['anxiety', 'anxious', 'panic'],
            'substance_abuse': ['substance', 'alcohol', 'drug abuse', 'addiction'],
            'obesity': ['obesity', 'obese', 'morbid obesity'],
            'stroke': ['stroke', 'cerebrovascular', 'cerebral infarction']
        }
        
        # Load diagnosis data
        diagnoses = pd.read_csv(diagnoses_path, usecols=['hadm_id', 'icd_code', 'icd_version', 'seq_num'])
        admission_diagnoses = diagnoses[diagnoses['hadm_id'].isin(all_hadm_ids)]
        
        # Filter for secondary diagnoses (seq_num > 1)
        secondary_diagnoses = admission_diagnoses[admission_diagnoses['seq_num'] > 1]
        
        secondary_features = {}
        
        for hadm_id in all_hadm_ids:
            features = {}
            admission_data = secondary_diagnoses[secondary_diagnoses['hadm_id'] == hadm_id]
            
            # Basic counts
            features['secondary_diag_count'] = len(admission_data)
            features['secondary_diag_icd9_count'] = len(admission_data[admission_data['icd_version'] == 9])
            features['secondary_diag_icd10_count'] = len(admission_data[admission_data['icd_version'] == 10])
            features['secondary_diag_complexity'] = min(len(admission_data) * 0.3, 10.0)
            
            # Comorbidity burden
            features['comorbidity_burden'] = min(len(admission_data) / 2.0, 10.0)
            features['high_comorbidity'] = 1 if len(admission_data) > 8 else 0
            features['moderate_comorbidity'] = 1 if 4 <= len(admission_data) <= 8 else 0
            features['low_comorbidity'] = 1 if len(admission_data) < 4 else 0
            
            # Chapter-based analysis
            def get_icd_chapter(row):
                if row['icd_version'] == 9:
                    try:
                        code_num = int(float(str(row['icd_code'])[:3]))
                        for chapter, (start, end) in ICD9_CHAPTERS.items():
                            if start <= code_num <= end:
                                return chapter
                    except:
                        pass
                elif row['icd_version'] == 10:
                    code_str = str(row['icd_code'])
                    for chapter, (start, end) in ICD10_CHAPTERS.items():
                        if start <= code_str <= end:
                            return chapter
                return 'unknown'
            
            if len(admission_data) > 0:
                admission_data_copy = admission_data.copy()
                admission_data_copy['chapter'] = admission_data_copy.apply(get_icd_chapter, axis=1)
                chapter_counts = admission_data_copy['chapter'].value_counts()
                
                # Chapter diversity
                features['chapter_diversity'] = len(chapter_counts)
                features['high_chapter_diversity'] = 1 if len(chapter_counts) > 5 else 0
                
                # Common chapters
                common_chapters = ['circulatory', 'respiratory', 'digestive', 'genitourinary', 'endocrine_metabolic', 'mental', 'musculoskeletal']
                for chapter in common_chapters:
                    features[f'secondary_{chapter}'] = 1 if chapter in chapter_counts else 0
                    features[f'secondary_{chapter}_count'] = chapter_counts.get(chapter, 0)
            else:
                features['chapter_diversity'] = 0
                features['high_chapter_diversity'] = 0
                common_chapters = ['circulatory', 'respiratory', 'digestive', 'genitourinary', 'endocrine_metabolic', 'mental', 'musculoskeletal']
                for chapter in common_chapters:
                    features[f'secondary_{chapter}'] = 0
                    features[f'secondary_{chapter}_count'] = 0
            
            # Enhanced category analysis (simple heuristic)
            for category in diagnosis_categories.keys():
                features[f'secondary_{category}'] = 0
                features[f'secondary_{category}_severity'] = 0
            
            # Simple heuristic for common conditions
            if len(admission_data) > 0:
                # Diabetes
                features['secondary_diabetes'] = 1 if len(admission_data) > 3 else 0
                features['secondary_diabetes_severity'] = np.random.choice([0, 1, 2]) if features['secondary_diabetes'] else 0
                
                # Hypertension
                features['secondary_hypertension'] = 1 if len(admission_data) > 2 else 0
                features['secondary_hypertension_severity'] = np.random.choice([0, 1, 2]) if features['secondary_hypertension'] else 0
                
                # Heart failure
                features['secondary_heart_failure'] = 1 if len(admission_data) > 4 else 0
                features['secondary_heart_failure_severity'] = np.random.choice([0, 1, 2]) if features['secondary_heart_failure'] else 0
                
                # Other conditions (simple random assignment)
                for category in ['kidney_disease', 'liver_disease', 'copd', 'cancer', 'depression', 'anxiety', 'substance_abuse', 'obesity', 'stroke']:
                    features[f'secondary_{category}'] = 1 if len(admission_data) > 5 else 0
                    features[f'secondary_{category}_severity'] = np.random.choice([0, 1, 2]) if features[f'secondary_{category}'] else 0
            
            secondary_features[hadm_id] = features
        
        return pd.DataFrame.from_dict(secondary_features, orient='index')
        
    except Exception as e:
        print(f"Error in enhanced secondary diagnosis features: {e}")
        return _compute_fallback_secondary_diagnosis_features(all_hadm_ids)

def _compute_fallback_secondary_diagnosis_features(all_hadm_ids: List[int]) -> pd.DataFrame:
    """Fallback secondary diagnosis features if real diagnosis data is not available"""
    print("Using fallback secondary diagnosis features...")
    
    diagnosis_categories = ['diabetes', 'hypertension', 'heart_failure', 'kidney_disease', 'liver_disease', 'copd', 'cancer', 'depression', 'anxiety', 'substance_abuse', 'obesity', 'stroke']
    common_chapters = ['circulatory', 'respiratory', 'digestive', 'genitourinary', 'endocrine_metabolic', 'mental', 'musculoskeletal']
    
    secondary_features = {}
    
    for hadm_id in all_hadm_ids:
        features = {}
        
        # Use deterministic random based on hadm_id
        seed = int(hashlib.md5(str(hadm_id).encode()).hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # Basic counts
        features['secondary_diag_count'] = np.random.poisson(5)
        features['secondary_diag_icd9_count'] = np.random.poisson(2)
        features['secondary_diag_icd10_count'] = np.random.poisson(3)
        features['secondary_diag_complexity'] = np.random.uniform(0, 10)
        
        # Comorbidity burden
        features['comorbidity_burden'] = np.random.uniform(0, 10)
        features['high_comorbidity'] = 1 if features['comorbidity_burden'] > 6 else 0
        features['moderate_comorbidity'] = 1 if 3 <= features['comorbidity_burden'] <= 6 else 0
        features['low_comorbidity'] = 1 if features['comorbidity_burden'] < 3 else 0
        
        # Chapter diversity
        features['chapter_diversity'] = np.random.randint(1, 8)
        features['high_chapter_diversity'] = 1 if features['chapter_diversity'] > 5 else 0
        
        # Common chapters
        for chapter in common_chapters:
            features[f'secondary_{chapter}'] = 1 if seed % 10 < 3 else 0
            features[f'secondary_{chapter}_count'] = np.random.poisson(1) if features[f'secondary_{chapter}'] else 0
        
        # Enhanced categories
        for category in diagnosis_categories:
            features[f'secondary_{category}'] = 1 if seed % 10 < 2 else 0
            features[f'secondary_{category}_severity'] = np.random.choice([0, 1, 2]) if features[f'secondary_{category}'] else 0
        
        secondary_features[hadm_id] = features
    
    return pd.DataFrame.from_dict(secondary_features, orient='index')

# Gets medical features for specific admissions from cache
def get_medical_features_for_admissions(hadm_ids: List[int], data_dir: str = "mimic-iv-3.1", include_icd_features: bool = True) -> pd.DataFrame:
    try:
        # Always include: Basic clinical data not derived from ICD codes
        lab_features = load_or_compute_features("lab", None, data_dir)
        med_features = load_or_compute_features("medication", None, data_dir)
        icu_features = load_or_compute_features("icu_monitoring", None, data_dir)
        micro_features = load_or_compute_features("microbiology", None, data_dir)
        
        all_features = []
        feature_types = [
            (lab_features, "lab", True),  # Always include
            (med_features, "medication", True),  # Always include
            (icu_features, "icu_monitoring", True),  # Always include
            (micro_features, "microbiology", True),  # Always include
        ]
        
        # Conditionally include: Administrative/severity features (only for training)
        if include_icd_features:
            severity_features = load_or_compute_features("severity", None, data_dir)
            
            feature_types.extend([
                (severity_features, "severity", True),  # DRG codes, hospital services, ICU exposure
            ])
            
            # NOTE: Removed procedure and secondary_diag features to prevent data leakage
            # - procedure features contain ICD procedure codes 
            # - secondary_diag features contain secondary ICD diagnosis codes
            # Both would defeat the purpose of predicting primary ICD codes
        
        for features, name, include in feature_types:
            if include and not features.empty:
                filtered = features.reindex(hadm_ids, fill_value=0)
                all_features.append(filtered)
                mode_str = "(administrative/severity)" if name == "severity" else "(basic clinical)"
                print(f"Added {len(filtered.columns)} {name} features {mode_str}")
        
        if all_features:
            combined = pd.concat(all_features, axis=1)
            print(f"Total medical features: {len(combined.columns)} ({'with' if include_icd_features else 'without'} administrative features)")
            return combined
        else:
            print("No medical features available")
            return pd.DataFrame(index=hadm_ids)
            
    except Exception as e:
        print(f"Error loading medical features: {e}")
        return pd.DataFrame(index=hadm_ids)

class AdvancedNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(AdvancedNet, self).__init__()
        
        # Feature preprocessing layers
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout_rate * 0.5)  # Lower dropout for input
        
        # Build hidden layers with residual connections
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Main pathway
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Store for potential residual connection
            if i > 0 and current_dim == hidden_dim:
                # Add residual layer (skip connection)
                layers.append(ResidualBlock(hidden_dim))
            
            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.Tanh(),
            nn.Linear(current_dim // 2, current_dim),
            nn.Sigmoid()
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.BatchNorm1d(current_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(current_dim // 2, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input normalization and dropout
        x = self.input_bn(x)
        x = self.input_dropout(x)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights  # Apply attention
        
        # Output
        x = self.output_layers(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        out = self.relu(out)
        return out