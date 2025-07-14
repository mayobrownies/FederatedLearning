"""
Advanced Feature Engineering Suggestions for ICD Prediction
==========================================================

This file contains implementation ideas for additional features that could
significantly improve ICD prediction accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

def create_feature_interactions(df, top_features=20):
    """Create polynomial feature interactions for top predictive features."""
    selector = SelectKBest(score_func=f_classif, k=top_features)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    print(f"Creating feature interactions for top {top_features} features")
    return pd.DataFrame(poly.fit_transform(df), index=df.index)

def create_clinical_risk_scores(medical_features):
    """Create composite clinical risk scores."""
    risk_scores = pd.DataFrame(index=medical_features.index)
    
    risk_scores['sepsis_risk'] = (
        medical_features.get('lab_wbc_high', 0) +
        medical_features.get('lab_lactate_high', 0) +
        medical_features.get('icu_fever', 0) +
        medical_features.get('icu_hypotension', 0)
    )
    
    risk_scores['cardiac_risk'] = (
        medical_features.get('lab_troponin_high', 0) +
        medical_features.get('med_cardiovascular', 0) +
        medical_features.get('icu_arrhythmia', 0)
    )
    
    risk_scores['respiratory_risk'] = (
        medical_features.get('icu_mechanical_ventilation', 0) +
        medical_features.get('med_respiratory', 0) +
        medical_features.get('lab_blood_gas_abnormal', 0)
    )
    
    return risk_scores

def create_temporal_patterns(admission_data):
    """Create advanced temporal feature patterns."""
    temporal_features = pd.DataFrame(index=admission_data.index)
    
    temporal_features['admission_hour'] = pd.to_datetime(admission_data['admittime']).dt.hour
    temporal_features['is_night_admission'] = (temporal_features['admission_hour'] >= 22) | (temporal_features['admission_hour'] <= 6)
    
    temporal_features['admission_weekday'] = pd.to_datetime(admission_data['admittime']).dt.dayofweek
    temporal_features['is_weekend_admission'] = temporal_features['admission_weekday'] >= 5
    
    temporal_features['admission_month'] = pd.to_datetime(admission_data['admittime']).dt.month
    temporal_features['is_flu_season'] = temporal_features['admission_month'].isin([11, 12, 1, 2, 3])
    
    return temporal_features

def create_medication_complexity_features(medication_data):
    """Create advanced medication analysis features."""
    med_features = pd.DataFrame(index=medication_data.index)
    
    high_risk_combinations = [
        ['warfarin', 'aspirin'],
        ['insulin', 'beta_blocker'],
        ['ace_inhibitor', 'diuretic']
    ]
    
    med_features['polypharmacy_burden'] = medication_data.filter(like='med_').sum(axis=1)
    med_features['high_alert_drug_count'] = medication_data.filter(regex='insulin|warfarin|heparin').sum(axis=1)
    
    return med_features 