## Directory Structure

```
FederatedLearning/
├── feature_cache/                 # Cached processed features (shared across all implementations)
├── heterogeneous_fedavg/          # FedAvg implementation
│   ├── client.py                 # Federated client implementation
│   ├── feature_suggestions.py    # Feature suggestion utilities
│   ├── run.py                    # Main execution script
│   ├── server.py                 # Federated server implementation
│   ├── task.py                   # Core ML tasks and data loading
│   └── test_prediction_display.py # Prediction analysis tools
├── heterogeneous_fedprox/         # FedProx implementation  
│   ├── client.py                 # Federated client implementation
│   ├── feature_suggestions.py    # Feature suggestion utilities
│   ├── run.py                    # Main execution script
│   ├── server.py                 # Federated server implementation
│   ├── task.py                   # Core ML tasks and data loading
│   └── test_prediction_display.py # Prediction analysis tools
├── hybrid_fedprox/                # Hybrid FedProx implementation
│   ├── __init__.py
│   ├── client.py                 # Federated client implementation
│   ├── feature_suggestions.py    # Feature suggestion utilities
│   ├── run.py                    # Main execution script
│   ├── server.py                 # Federated server implementation
│   ├── task.py                   # Core ML tasks and data loading
│   └── test_prediction_display.py # Prediction analysis tools
├── mimic-iv-3.1/                 # MIMIC-IV dataset (excluded from git)
├── preprocess_features.py         # Feature preprocessing script
└── README.md
```

Place the MIMIC-IV dataset in the `mimic-iv-3.1/` directory:
```
mimic-iv-3.1/
├── hosp/
│   ├── admissions.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── patients.csv.gz
│   ├── prescriptions.csv.gz
│   └── [other hospital data files]
└── icu/
    ├── chartevents.csv.gz
    └── [other ICU data files]
```

First, run the "preprocess_features" program to get the feature_caches, then "run" program.