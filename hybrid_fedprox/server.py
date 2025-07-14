from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

import flwr as fl
from flwr.server.strategy import FedProx
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, EvaluateIns, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hybrid_fedprox.task import TOP_ICD_CODES, display_prediction_analysis, get_model, load_and_partition_data, create_features_and_labels, get_global_feature_space, set_weights, MimicDataset
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

def test_global_model(parameters: Parameters, data_dir: str = "mimic-iv-3.1") -> None:
    """Test the global aggregated model and display prediction analysis."""
    try:
        print("\n" + "=" * 80)
        print("TESTING GLOBAL AGGREGATED MODEL")
        print("=" * 80)
        
        partitions = load_and_partition_data(data_dir, min_partition_size=1000)
        global_feature_space = get_global_feature_space(data_dir)
        
        test_features_list = []
        test_labels_list = []
        
        samples_per_partition = 50
        for partition_name, partition_data in partitions.items():
            if len(partition_data) >= samples_per_partition:
                sampled_data = partition_data.sample(n=samples_per_partition, random_state=42)
                features, labels = create_features_and_labels(
                    sampled_data, partition_name, global_feature_space, include_icd_features=False
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
        model = get_model(input_dim, output_dim=output_dim)
        
        from flwr.common import parameters_to_ndarrays
        weights = parameters_to_ndarrays(parameters)
        set_weights(model, weights)
        
        X_test = test_features.values.astype(np.float32)
        y_test = test_labels.astype(np.int64)
        test_dataset = MimicDataset(X_test, y_test)
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

class CustomFedProx(FedProx):
    def __init__(self, *args, **kwargs):
        self.total_rounds = kwargs.pop("total_rounds", 1)
        self.data_dir = kwargs.pop("data_dir", "mimic-iv-3.1")
        super().__init__(*args, **kwargs)
        self.current_round = 0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | Tuple[ClientProxy, Exception]],
    ) -> Tuple[Optional[Parameters], Dict[str, any]]:
        """Aggregate fit results and test global model on final round."""
        
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if server_round == self.total_rounds and aggregated_result[0] is not None:
            print(f"\nFINAL ROUND ({server_round}) - TESTING GLOBAL AGGREGATED MODEL")
            test_global_model(aggregated_result[0], self.data_dir)
        
        return aggregated_result

def get_strategy(run_config):
    """Create FedProx strategy with custom aggregation functions."""
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
    
    return CustomFedProx(
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        proximal_mu=run_config.get("proximal-mu", 0.1),
        min_available_clients=run_config.get("num-partitions", 2),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=run_config.get("num-partitions", 2),
        min_evaluate_clients=run_config.get("num-partitions", 2),
        accept_failures=True,
        total_rounds=run_config.get("num-server-rounds", 1),
        data_dir=run_config.get("data-dir", "mimic-iv-3.1"),
    ) 