# ULCD Strategy for Flower Federated Learning
# Implements latent consensus distillation instead of traditional weight averaging

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import pickle
import base64

import flwr as fl
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, EvaluateIns, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from .ulcd_components import ULCDServer, visualize_latents

# ============================================================================
# ULCD STRATEGY
# ============================================================================
class ULCDStrategy(Strategy):
    """
    ULCD (Unified Latent Consensus Distillation) Strategy
    
    Instead of aggregating model weights, this strategy:
    1. Collects latent representations from clients
    2. Detects anomalous clients
    3. Aggregates trusted latents into a prototype
    4. Sends prototype back to clients for fine-tuning
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        latent_dim: int = 64,
        anomaly_threshold: float = 0.3,
        enable_visualization: bool = True,
    ):
        super().__init__()
        
        # Flower strategy parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        
        # ULCD-specific parameters
        self.latent_dim = latent_dim
        self.anomaly_threshold = anomaly_threshold
        self.enable_visualization = enable_visualization
        
        # Initialize ULCD server
        self.ulcd_server = ULCDServer(latent_dim=latent_dim)
        self.round_num = 0
        
        print(f"[ULCD Strategy] Initialized with latent_dim={latent_dim}, threshold={anomaly_threshold}")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize with dummy parameters (ULCD doesn't need initial model weights)"""
        # Return dummy parameters - ULCD doesn't use traditional weight initialization
        dummy_weights = [np.array([0.0])]
        return ndarrays_to_parameters(dummy_weights)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Configure fit round for ULCD - request latent summaries from clients"""
        
        self.round_num = server_round
        print(f"\n[ULCD Strategy] Configuring fit round {server_round}")
        
        # Sample clients
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=self.min_available_clients
        )
        
        # Create fit instructions
        fit_ins = []
        for client in clients:
            # For ULCD, we send the current prototype (if available) to clients
            if hasattr(self.ulcd_server, 'prototype'):
                prototype_array = self.ulcd_server.prototype.detach().cpu().numpy()
                fit_parameters = ndarrays_to_parameters([prototype_array])
            else:
                fit_parameters = parameters  # Use dummy parameters for first round
            
            config = {
                "ulcd_mode": True,
                "server_round": server_round,
                "request_latent_summary": True,
                "anomaly_threshold": self.anomaly_threshold
            }
            
            fit_ins.append((client, fl.common.FitIns(fit_parameters, config)))
        
        print(f"[ULCD Strategy] Configured {len(fit_ins)} clients for round {server_round}")
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate latent summaries using ULCD consensus"""
        
        print(f"\n[ULCD Strategy] Aggregating round {server_round}")
        print(f"Received {len(results)} results, {len(failures)} failures")
        
        if not results:
            print("[ULCD Strategy] No results to aggregate")
            return None, {}
        
        # Extract latent summaries from client results
        latent_summaries = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            try:
                # Decode latent summary from fit_res.parameters
                latent_array = parameters_to_ndarrays(fit_res.parameters)[0]
                latent_tensor = torch.from_numpy(latent_array).float()
                
                client_id = len(latent_summaries)  # Simple client ID assignment
                latent_summaries.append((client_id, latent_tensor))
                
                # Collect metrics
                if fit_res.metrics:
                    client_metrics.append((fit_res.num_examples, fit_res.metrics))
                
                print(f"  Client {client_id}: latent shape {latent_tensor.shape}")
                
            except Exception as e:
                print(f"[WARNING] Failed to decode latent from client: {e}")
                continue
        
        if not latent_summaries:
            print("[ULCD Strategy] No valid latent summaries received")
            return None, {}
        
        # Visualize before aggregation
        if self.enable_visualization:
            visualize_latents(
                latent_summaries, 
                self.ulcd_server.prototype, 
                f"Round_{server_round}_Before_Aggregation"
            )
        
        # Detect anomalies and filter trusted clients
        trusted_clients, flagged_clients = self.ulcd_server.detect_anomalies(
            [latent for _, latent in latent_summaries], 
            threshold=self.anomaly_threshold
        )
        
        print(f"  Trusted clients: {[cid for cid, _ in trusted_clients]}")
        print(f"  Flagged clients: {flagged_clients}")
        
        # Aggregate trusted latents
        if trusted_clients:
            trusted_latents = [latent for _, latent in trusted_clients]
            self.ulcd_server.aggregate_latents(trusted_latents)
            
            print(f"  Aggregated {len(trusted_latents)} trusted latent summaries")
            
            # Visualize after aggregation
            if self.enable_visualization:
                visualize_latents(
                    latent_summaries, 
                    self.ulcd_server.prototype, 
                    f"Round_{server_round}_After_Aggregation"
                )
        else:
            print("  [WARNING] No trusted clients - skipping aggregation")
        
        # Return updated prototype as parameters
        prototype_array = self.ulcd_server.prototype.detach().cpu().numpy()
        aggregated_parameters = ndarrays_to_parameters([prototype_array])
        
        # Aggregate metrics
        aggregated_metrics = {}
        if client_metrics and self.fit_metrics_aggregation_fn:
            aggregated_metrics = self.fit_metrics_aggregation_fn(client_metrics)
        
        # Add ULCD-specific metrics
        aggregated_metrics.update({
            "ulcd_trusted_clients": len(trusted_clients),
            "ulcd_flagged_clients": len(flagged_clients),
            "ulcd_consensus_ratio": len(trusted_clients) / len(latent_summaries) if latent_summaries else 0,
            "ulcd_prototype_norm": float(torch.norm(self.ulcd_server.prototype).item())
        })
        
        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation round"""
        
        # Sample clients for evaluation
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_evaluate),
            self.min_evaluate_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )
        
        # Create evaluation instructions
        evaluate_ins = []
        for client in clients:
            config = {
                "ulcd_mode": True,
                "server_round": server_round,
                "evaluate_with_prototype": True
            }
            evaluate_ins.append((client, fl.common.EvaluateIns(parameters, config)))
        
        return evaluate_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        # Extract evaluation metrics
        eval_metrics = []
        total_examples = 0
        total_loss = 0.0
        
        for _, eval_res in results:
            total_examples += eval_res.num_examples
            total_loss += eval_res.loss * eval_res.num_examples
            
            if eval_res.metrics:
                eval_metrics.append((eval_res.num_examples, eval_res.metrics))
        
        # Calculate weighted average loss
        aggregated_loss = total_loss / total_examples if total_examples > 0 else None
        
        # Aggregate other metrics
        aggregated_metrics = {}
        if eval_metrics and self.evaluate_metrics_aggregation_fn:
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        
        return aggregated_loss, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Server-side evaluation (optional)"""
        # ULCD strategy doesn't need server-side evaluation
        return None

# ============================================================================
# ULCD STRATEGY FACTORY
# ============================================================================
def get_ulcd_strategy(run_config=None, **kwargs):
    """Factory function to create ULCD strategy"""
    
    # Extract ULCD-specific parameters
    latent_dim = kwargs.get('latent_dim', 64)
    anomaly_threshold = kwargs.get('anomaly_threshold', 0.3)
    enable_visualization = kwargs.get('enable_visualization', True)
    
    # Extract standard Flower parameters
    num_clients = len(run_config.get("cached_partitions", {})) if run_config else 2
    
    strategy_kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": num_clients,
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
        "latent_dim": latent_dim,
        "anomaly_threshold": anomaly_threshold,
        "enable_visualization": enable_visualization,
    }
    
    # Add aggregation functions if provided
    if "evaluate_metrics_aggregation_fn" in kwargs:
        strategy_kwargs["evaluate_metrics_aggregation_fn"] = kwargs["evaluate_metrics_aggregation_fn"]
    if "fit_metrics_aggregation_fn" in kwargs:
        strategy_kwargs["fit_metrics_aggregation_fn"] = kwargs["fit_metrics_aggregation_fn"]
    
    print(f"[ULCD Strategy Factory] Creating ULCD strategy with {num_clients} clients")
    
    return ULCDStrategy(**strategy_kwargs)