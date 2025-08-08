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
TOP_K_CODES = 75
TOP_ICD_CODES = []
ICD_CODE_TO_INDEX = {}
INDEX_TO_ICD_CODE = {}

# Initializes global ICD code mappings for the top K most frequent codes
def initialize_top_icd_codes(data: pd.DataFrame, top_k_codes: int = 75):
    global TOP_ICD_CODES, ICD_CODE_TO_INDEX, INDEX_TO_ICD_CODE
    
    if len(TOP_ICD_CODES) > 0:  # Already initialized
        print(f"ICD codes already initialized with {len(TOP_ICD_CODES)} codes")
        return
    
    print(f"Finding top {top_k_codes} most frequent ICD codes...")
    global TOP_K_CODES
    TOP_K_CODES = top_k_codes
    
    # Count frequency of each ICD code
    icd_counts = data['icd_code'].value_counts()
    
    # Get top K codes
    top_codes = icd_counts.head(top_k_codes).index.tolist()
    
    # Create mappings
    TOP_ICD_CODES.extend(top_codes)
    ICD_CODE_TO_INDEX.update({code: idx for idx, code in enumerate(top_codes)})
    INDEX_TO_ICD_CODE.update({idx: code for idx, code in enumerate(top_codes)})
    
    print(f"Top {len(TOP_ICD_CODES)} ICD codes selected for prediction")
    print(f"Most frequent codes: {TOP_ICD_CODES[:10]}...")
    print(f"Code frequencies: {[icd_counts[code] for code in TOP_ICD_CODES[:10]]}")

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