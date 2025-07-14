import copy
from typing import Dict

import flwr as fl
import torch

from .task import get_model, load_data, train, test, get_weights, set_weights, TOP_ICD_CODES

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, run_config: Dict, partition_id: int):
        self.run_config = run_config
        self.partition_id = partition_id
        
        self.trainloader, self.testloader, input_dim, output_dim = load_data(
            partition_id=partition_id,
            batch_size=run_config["batch-size"],
            data_dir=run_config["data-dir"],
            min_partition_size=run_config["min-partition-size"]
        )
        
        self.net = get_model(input_dim, output_dim=output_dim)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        """Return current model weights."""
        return get_weights(self.net)

    def fit(self, parameters, config):
        """Train model and return updated weights."""
        set_weights(self.net, parameters)
        
        global_net = copy.deepcopy(self.net)
        
        train_loss = train(
            net=self.net,
            global_net=global_net,
            trainloader=self.trainloader,
            epochs=self.run_config["local-epochs"],
            learning_rate=self.run_config["learning-rate"],
            proximal_mu=self.run_config["proximal-mu"],
            device=self.device
        )
        
        all_labels = []
        for _, labels in self.trainloader:
            all_labels.extend(labels.numpy())
        
        top_k_count = sum(1 for label in all_labels if label < len(TOP_ICD_CODES))
        top_k_ratio = top_k_count / len(all_labels) if all_labels else 0.0
        
        return get_weights(self.net), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "top_k_ratio": top_k_ratio
        }

    def evaluate(self, parameters, config):
        """Evaluate model and return metrics."""
        try:
            set_weights(self.net, parameters)
            
            loss, metrics = test(self.net, self.testloader, device=self.device)
            
            return loss, len(self.testloader.dataset), metrics
        
        except Exception as e:
            print(f"Client {self.partition_id} evaluation failed: {e}")
            
            return 0.0, len(self.testloader.dataset), {
                "accuracy": 0.0, 
                "f1": 0.0, 
                "precision": 0.0, 
                "recall": 0.0
            }

def get_client(run_config: Dict, partition_id: int) -> FlowerClient:
    """Create Flower client for given configuration and partition."""
    return FlowerClient(run_config, partition_id) 