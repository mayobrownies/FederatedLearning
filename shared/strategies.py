from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, EvaluateIns, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from .task import TOP_ICD_CODES, get_model, load_and_partition_data, create_features_and_labels, get_global_feature_space, set_weights
from .models import MimicDataset
from torch.utils.data import DataLoader

def weighted_average(metrics):
    """Aggregate evaluation metrics using weighted average."""
    try:
        print(f"Aggregating metrics from {len(metrics)} clients...")
        
        total_examples = sum([num_examples for num_examples, _ in metrics])
        
        if total_examples == 0:
            print("Warning: No examples found in metrics")
            return {}
        
        weighted_metrics = {}
        for key in ["accuracy", "f1", "precision", "recall"]:
            if any(key in m for _, m in metrics):
                weighted_sum = sum([num_examples * m.get(key, 0) for num_examples, m in metrics])
                weighted_metrics[key] = weighted_sum / total_examples
                print(f"Weighted {key}: {weighted_metrics[key]:.4f}")
        
        return weighted_metrics
        
    except Exception as e:
        print(f"Error in weighted_average: {e}")
        return {}

def weighted_average_fit(fit_metrics):
    """Aggregate fit metrics using weighted average."""
    try:
        print(f"Aggregating fit metrics from {len(fit_metrics)} clients...")
        
        total_examples = sum([num_examples for num_examples, _ in fit_metrics])
        
        if total_examples == 0:
            print("Warning: No examples found in fit metrics")
            return {}
        
        weighted_metrics = {}
        
        if any("train_loss" in m for _, m in fit_metrics):
            weighted_sum = sum([num_examples * m.get("train_loss", 0) for num_examples, m in fit_metrics])
            weighted_metrics["train_loss"] = weighted_sum / total_examples
            print(f"Weighted training loss: {weighted_metrics['train_loss']:.4f}")
        
        if any("top_k_ratio" in m for _, m in fit_metrics):
            weighted_sum = sum([num_examples * m.get("top_k_ratio", 0) for num_examples, m in fit_metrics])
            weighted_metrics["top_k_ratio"] = weighted_sum / total_examples
            print(f"Weighted top-K ratio: {weighted_metrics['top_k_ratio']:.2%}")
        
        return weighted_metrics
        
    except Exception as e:
        print(f"Error in weighted_average_fit: {e}")
        return {}

def display_prediction_analysis(predictions, true_labels):
    """Display detailed prediction analysis."""
    try:
        print("\n" + "="*60)
        print("PREDICTION ANALYSIS")
        print("="*60)
        
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        unique_true, true_counts = np.unique(true_labels, return_counts=True)
        
        print(f"Total predictions: {len(predictions)}")
        print(f"Unique predicted classes: {len(unique_preds)}")
        print(f"Unique true classes: {len(unique_true)}")
        
        print("\nPREDICTION DISTRIBUTION:")
        for pred_class, count in zip(unique_preds, pred_counts):
            percentage = (count / len(predictions)) * 100
            if pred_class < len(TOP_ICD_CODES):
                class_name = f"ICD_{pred_class} ({TOP_ICD_CODES[pred_class]})"
            else:
                class_name = "Other"
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print("\nTRUE LABEL DISTRIBUTION:")
        for true_class, count in zip(unique_true, true_counts):
            percentage = (count / len(true_labels)) * 100
            if true_class < len(TOP_ICD_CODES):
                class_name = f"ICD_{true_class} ({TOP_ICD_CODES[true_class]})"
            else:
                class_name = "Other"
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Accuracy calculation
        correct = np.sum(predictions == true_labels)
        accuracy = correct / len(predictions)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Check if model is only predicting "other" class
        other_class_id = len(TOP_ICD_CODES)
        other_predictions = np.sum(predictions == other_class_id)
        other_percentage = (other_predictions / len(predictions)) * 100
        
        if other_percentage > 90:
            print(f"\n⚠️  WARNING: Model is mostly predicting 'Other' class ({other_percentage:.1f}%)")
            print("   This suggests severe class imbalance or learning issues.")
            print("   Consider:")
            print("   - Reducing TOP_K_CODES")
            print("   - Using focal loss")
            print("   - Adjusting learning rate or epochs")
            print("   - Better data balancing")
        elif other_percentage > 50:
            print(f"\n⚡ NOTICE: High 'Other' class predictions ({other_percentage:.1f}%)")
            print("   Model is learning but may benefit from class balancing")
        else:
            print(f"\n✅ Good prediction diversity - 'Other' class: {other_percentage:.1f}%")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in prediction analysis: {e}")
        import traceback
        traceback.print_exc()

def test_global_model(parameters: Parameters, data_dir: str = "mimic-iv-3.1", partition_scheme: str = "heterogeneous", top_k_codes: int = 75) -> None:
    """Test the global aggregated model and display prediction analysis."""
    try:
        print("\n" + "=" * 80)
        print("TESTING GLOBAL AGGREGATED MODEL")
        print("=" * 80)
        
        partitions = load_and_partition_data(data_dir, min_partition_size=1000, partition_scheme=partition_scheme, top_k_codes=top_k_codes)
        global_feature_space = get_global_feature_space(data_dir)
        
        test_features_list = []
        test_labels_list = []
        
        samples_per_partition = 50
        for partition_name, partition_data in partitions.items():
            if len(partition_data) >= samples_per_partition:
                sampled_data = partition_data.sample(n=samples_per_partition, random_state=42)
                features, labels = create_features_and_labels(
                    sampled_data, partition_name, global_feature_space, include_icd_features=True
                )
                test_features_list.append(features)
                test_labels_list.append(labels)
        
        if not test_features_list:
            print("ERROR: Could not create test dataset")
            return
            
        import pandas as pd
        reference_columns = test_features_list[0].columns
        test_labels_combined = []
        aligned_features_list = []
        
        for features, labels in zip(test_features_list, test_labels_list):
            aligned_features = features.reindex(columns=reference_columns, fill_value=0)
            aligned_features_list.append(aligned_features)
            test_labels_combined.extend(labels.tolist())
        
        test_features = pd.concat(aligned_features_list, ignore_index=True)
        test_labels = np.array(test_labels_combined)
        
        print(f"Created global test dataset: {len(test_features)} samples")
        print(f"Features shape: {test_features.shape}")
        print(f"Labels shape: {test_labels.shape}")
        
        input_dim = test_features.shape[1]
        output_dim = len(TOP_ICD_CODES) + 1
        model = get_model("advanced", input_dim, output_dim=output_dim)
        
        from flwr.common import parameters_to_ndarrays
        weights = parameters_to_ndarrays(parameters)
        set_weights(model, weights)
        
        X_test = test_features.values.astype(np.float32)
        y_test = test_labels.astype(np.int64)
        test_dataset = MimicDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        print(f"Global model predictions collected: {len(all_predictions)} samples")
        
        display_prediction_analysis(all_predictions, all_labels)
        
        print("=" * 80)
        print("END OF GLOBAL MODEL PREDICTION ANALYSIS")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"Error testing global model: {e}")
        import traceback
        traceback.print_exc()

def get_strategy(strategy_name: str, run_config=None, **kwargs):
    """Get the appropriate federated learning strategy with proper aggregation functions."""
    
    def evaluate_metrics_aggregation_fn(eval_metrics):
        try:
            print(f"Evaluate metrics aggregation called with {len(eval_metrics)} results")
            return weighted_average(eval_metrics)
        except Exception as e:
            print(f"Error in evaluate_metrics_aggregation_fn: {e}")
            return {}
    
    def fit_metrics_aggregation_fn(fit_metrics):
        try:
            print(f"Fit metrics aggregation called with {len(fit_metrics)} results")
            return weighted_average_fit(fit_metrics)
        except Exception as e:
            print(f"Error in fit_metrics_aggregation_fn: {e}")
            return {}
    
    # Common strategy parameters
    num_clients = len(run_config.get("cached_partitions", {})) if run_config else 2
    strategy_kwargs = {
        "evaluate_metrics_aggregation_fn": evaluate_metrics_aggregation_fn,
        "fit_metrics_aggregation_fn": fit_metrics_aggregation_fn,
        "min_available_clients": num_clients,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": num_clients,
        "min_evaluate_clients": num_clients,
    }
    
    # Add any additional kwargs
    additional_kwargs = {k: v for k, v in kwargs.items() if k not in strategy_kwargs}
    strategy_kwargs.update(additional_kwargs)
    
    if strategy_name == "fedavg":
        # FedAvg doesn't use proximal_mu, so remove it from kwargs
        fedavg_kwargs = {k: v for k, v in strategy_kwargs.items() if k != 'proximal_mu'}
        return FedAvg(**fedavg_kwargs)
    elif strategy_name == "fedprox":
        return FedProx(**strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")