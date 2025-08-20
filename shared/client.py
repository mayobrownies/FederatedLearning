import copy
from typing import Dict

import flwr as fl
import torch

from .task import get_model, load_data, train, test, get_weights, set_weights, TOP_ICD_CODES

# ============================================================================
# CLIENT
# ============================================================================
# Flower client for federated learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, run_config: Dict, partition_id: int):
        self.run_config = run_config
        self.partition_id = partition_id
        
        print(f"[CLIENT {partition_id}] Initializing client...")
        
        # Use different loading strategies based on mode
        if run_config.get("use_local_mode", True):
            # Local mode: use cached partitions and precomputed data
            self.trainloader, self.testloader, input_dim, output_dim = load_data(
                partition_id=partition_id,
                batch_size=run_config.get("batch_size", run_config.get("batch-size", 64)),
                data_dir=run_config.get("data_dir", run_config.get("data-dir", "mimic-iv-3.1")),
                min_partition_size=run_config.get("min_partition_size", run_config.get("min-partition-size", 1000)),
                partition_scheme=run_config.get("partitioning", "heterogeneous"),
                top_k_codes=run_config.get("top_k_codes", 75),
                cached_partitions=run_config.get("cached_partitions"),
                precomputed_icd_data=run_config,
                quiet_mode=run_config.get("quiet_mode", False)
            )
        else:
            # Distributed mode: use global partitions
            self.trainloader, self.testloader, input_dim, output_dim = load_data(
                partition_id=partition_id,
                batch_size=run_config.get("batch_size", run_config.get("batch-size", 64)),
                data_dir=run_config.get("data_dir", run_config.get("data-dir", "mimic-iv-3.1")),
                min_partition_size=run_config.get("min_partition_size", run_config.get("min-partition-size", 1000)),
                partition_scheme=run_config.get("partitioning", "heterogeneous"),
                top_k_codes=run_config.get("top_k_codes", 75),
                cached_partitions=None,  # Will use GLOBAL_PARTITIONS
                precomputed_icd_data=run_config,
                quiet_mode=run_config.get("quiet_mode", False)
            )
        
        self.net = get_model(run_config.get("model_name", "advanced"), input_dim, output_dim=output_dim)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[CLIENT {partition_id}] Using device: {self.device}")
        print(f"[CLIENT {partition_id}] Client initialized successfully")

    # Returns model parameters
    def get_parameters(self, config):
        return get_weights(self.net)

    # Trains model with given parameters
    def fit(self, parameters, config):
        print(f"[CLIENT {self.partition_id}] Starting training...")
        set_weights(self.net, parameters)
        
        global_net = copy.deepcopy(self.net)
        
        train_loss = train(
            net=self.net,
            global_net=global_net,
            trainloader=self.trainloader,
            epochs=self.run_config.get("local_epochs", self.run_config.get("local-epochs", 3)),
            learning_rate=self.run_config.get("learning_rate", self.run_config.get("learning-rate", 0.0002)),
            proximal_mu=self.run_config.get("proximal_mu", 0.0),
            device=self.device
        )
        
        all_labels = []
        for _, labels in self.trainloader:
            all_labels.extend(labels.numpy())
        
        top_k_count = sum(1 for label in all_labels if label < len(TOP_ICD_CODES))
        top_k_ratio = top_k_count / len(all_labels) if all_labels else 0.0
        
        print(f"[CLIENT {self.partition_id}] Training completed - Loss: {train_loss:.4f}, Top-K ratio: {top_k_ratio:.2%}")
        
        # Force garbage collection after training
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return get_weights(self.net), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "top_k_ratio": top_k_ratio
        }

    # Evaluates model with given parameters
    def evaluate(self, parameters, config):
        try:
            print(f"[CLIENT {self.partition_id}] Starting evaluation...")
            set_weights(self.net, parameters)
            
            # Force garbage collection before evaluation
            import gc
            gc.collect()
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            loss, metrics = test(self.net, self.testloader, device=self.device)
            
            # Force cleanup after evaluation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[CLIENT {self.partition_id}] Evaluation completed - Loss: {loss:.4f}, Acc: {metrics.get('accuracy', 0):.3f}")
            
            return loss, len(self.testloader.dataset), metrics
        
        except Exception as e:
            print(f"Client {self.partition_id} evaluation failed: {e}")
            
            # Force cleanup on failure
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return 0.0, len(self.testloader.dataset), {
                "accuracy": 0.0, 
                "f1": 0.0, 
                "precision": 0.0, 
                "recall": 0.0
            }

# ============================================================================
# CLIENT FACTORY
# ============================================================================
# Creates Flower client instance
def get_client(run_config: Dict, partition_id: int) -> FlowerClient:
    return FlowerClient(run_config, partition_id) 