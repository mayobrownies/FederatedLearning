## Directory Structure

```
FederatedLearning/
├── feature_cache/                 # Cached processed features (shared across all implementations)
├── shared/                        # Shared federated learning components (main implementation)
│   ├── client.py                 # Federated client implementation
│   ├── data_utils.py             # ICD code mappings and data utilities
│   ├── models.py                 # Neural network architectures and model factory
│   ├── partitioning.py           # Data partitioning strategies
│   ├── run.py                    # Main execution script
│   ├── strategies.py             # Federated learning strategies (FedAvg, FedProx)
│   └── task.py                   # Core ML tasks and data loading
├── heterogeneous_fedavg/          # Legacy FedAvg implementation
│   ├── client.py                 # Federated client implementation
│   ├── feature_suggestions.py    # Feature suggestion utilities
│   ├── run.py                    # Main execution script
│   ├── server.py                 # Federated server implementation
│   ├── task.py                   # Core ML tasks and data loading
│   └── test_prediction_display.py # Prediction analysis tools
├── heterogeneous_fedprox/         # Legacy FedProx implementation  
│   ├── client.py                 # Federated client implementation
│   ├── feature_suggestions.py    # Feature suggestion utilities
│   ├── run.py                    # Main execution script
│   ├── server.py                 # Federated server implementation
│   ├── task.py                   # Core ML tasks and data loading
│   └── test_prediction_display.py # Prediction analysis tools
├── hybrid_fedprox/                # Legacy Hybrid FedProx implementation
│   ├── __init__.py
│   ├── client.py                 # Federated client implementation
│   ├── feature_suggestions.py    # Feature suggestion utilities
│   ├── run.py                    # Main execution script
│   ├── server.py                 # Federated server implementation
│   ├── task.py                   # Core ML tasks and data loading
│   └── test_prediction_display.py # Prediction analysis tools
├── mimic-iv-3.1/                 # MIMIC-IV dataset (excluded from git)
├── preprocess_features.py         # Feature preprocessing script
├── train_isolated.py             # Non-federated training script
├── requirements.txt              # Python dependencies
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

First, run the "preprocess_features.py" program to get the feature_cache, then "shared/run.py" program.
The other federated learning folders likely do not work.