import flwr as fl
import torch
from flwr.common import Context
import os
from typing import Dict, Any

from .task import get_model, set_weights, train, test, load_data, get_weights

# Defines a Flower client for federated learning.
class FlowerClient(fl.client.NumPyClient):
    
    # Initializes the Flower client.
    def __init__(self, net, trainloader, valloader, local_epochs, learning_rate, device):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device

    # Returns the model's parameters.
    def get_parameters(self, config):
        print(f"Client: get_parameters called, config: {config}")
        return get_weights(self.net)

    # Trains the model on the local dataset.
    def fit(self, parameters, config):
        print(f"Client: fit called, config: {config}")
        set_weights(self.net, parameters)
        epochs = config.get("local_epochs", self.local_epochs)
        learning_rate = config.get("learning_rate", self.learning_rate)

        avg_loss = train(
            self.net, 
            self.trainloader, 
            epochs=epochs, 
            learning_rate=learning_rate, 
            device=self.device
        )
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": avg_loss}

    # Evaluates the model on the local validation set.
    def evaluate(self, parameters, config):
        print(f"Client: evaluate called, config: {config}")
        set_weights(self.net, parameters)
        loss, accuracy, y_true, y_pred = test(self.net, self.valloader, device=self.device)
        
        metrics = {"val_loss": float(loss), "accuracy": float(accuracy)}
        
        if config.get("get_predictions", False):
            metrics["y_true"] = y_true
            metrics["y_pred"] = y_pred
            
        return float(loss), len(self.valloader.dataset), metrics

# Creates a Flower client for a simulation.
def client_fn_simulation(context: Context):
    run_config = context.run_config
    
    if run_config.get("num-partitions", 2) == 1:
        partition_id = 0
    else:
        partition_id = context.node_config.get("partition_index", 0)

    return get_client(run_config=run_config, partition_id=partition_id)

# Creates and configures a Flower client instance.
def get_client(run_config: Dict[str, Any], partition_id: int):
    local_epochs = run_config.get("local-epochs", 1)
    batch_size = run_config.get("batch-size", 32)
    learning_rate = run_config.get("learning-rate", 0.01)
    data_dir = run_config.get("data-dir", "cifar-10-batches-py")
    num_data_partitions_total = run_config.get("num-partitions", 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model().to(device)

    trainloader, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_data_partitions_total,
        batch_size=batch_size,
        data_dir=data_dir
    )

    return FlowerClient(net, trainloader, valloader, local_epochs, learning_rate, device).to_client()

app = fl.client.ClientApp(
    client_fn=client_fn_simulation,
) 