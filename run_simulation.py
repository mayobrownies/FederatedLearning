import flwr as fl
from flwr.common import ndarrays_to_parameters, EvaluateRes, FitRes
from flwr.common.typing import Scalar
from federated_learning.server_app import get_strategy
from federated_learning.client_app import get_client
import torch
from typing import List, Tuple, Dict, Optional
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from federated_learning.task import CIFAR10_CLASSES
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Simulation configuration
NUM_CLIENTS = 5
NUM_SERVER_ROUNDS = 3
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Defines a mock client manager for the simulation.
class MockClientManager(ClientManager):
    # Initializes the client manager.
    def __init__(self, clients: List[fl.client.Client]):
        self.clients = {str(i): client for i, client in enumerate(clients)}

    # Returns the number of available clients.
    def num_available(self) -> int:
        return len(self.clients)

    # Samples a number of clients.
    def sample(self, num_clients: int, min_num_clients: Optional[int] = None, criterion=None) -> List[ClientProxy]:
        return list(self.clients.values())[:num_clients]
    
    # Registers a client.
    def register(self, client: ClientProxy) -> bool:
        return True
    
    # Unregisters a client.
    def unregister(self, client: ClientProxy) -> None:
        pass
    
    # Returns all clients.
    def all(self) -> Dict[str, ClientProxy]:
        return self.clients

    # Waits for a number of clients.
    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        return True

# Runs the federated learning simulation.
def main():

    run_config = {
        "num-partitions": NUM_CLIENTS,
        "num-server-rounds": NUM_SERVER_ROUNDS,
        "local-epochs": LOCAL_EPOCHS,
        "batch-size": BATCH_SIZE,
        "learning-rate": LEARNING_RATE,
        "data-dir": "cifar-10-batches-py"
    }

    # Create strategy
    strategy = get_strategy(run_config)

    # Create clients
    clients = [get_client(run_config, i) for i in range(NUM_CLIENTS)]
    client_manager = MockClientManager(clients)

    # Run simulation
    print("Starting simulation")
    
    # Initialize parameters
    parameters = strategy.initialize_parameters(client_manager=client_manager)

    for server_round in range(1, NUM_SERVER_ROUNDS + 1):
        print(f"Round {server_round}/{NUM_SERVER_ROUNDS}")

        # Fit round
        fit_instructions = strategy.configure_fit(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager
        )
        results = [(client, client.fit(ins)) for client, ins in fit_instructions]
        
        aggregated_fit = strategy.aggregate_fit(server_round, results, [])
        if aggregated_fit:
            parameters, _ = aggregated_fit
        else:
            print("Warning: aggregate_fit returned None")


        # Evaluate round
        evaluate_instructions = strategy.configure_evaluate(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager
        )
        eval_results = [(client, client.evaluate(ins)) for client, ins in evaluate_instructions]

        aggregated_eval = strategy.aggregate_evaluate(server_round, eval_results, [])
        if aggregated_eval:
            loss_aggregated, metrics_aggregated = aggregated_eval
            print(f"Round {server_round} accuracy: {metrics_aggregated['accuracy']:.4f}")

    print("Simulation finished.")

    # Collect predictions from the final round
    all_y_true = []
    all_y_pred = []

    # eval_results holds the results from the last round
    for _, eval_res in eval_results:
        if "y_true" in eval_res.metrics and "y_pred" in eval_res.metrics:
            all_y_true.extend(eval_res.metrics["y_true"])
            all_y_pred.extend(eval_res.metrics["y_pred"])

    # Plot confusion matrix
    if all_y_true and all_y_pred:
        cm = confusion_matrix(all_y_true, all_y_pred)
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CIFAR10_CLASSES,
            yticklabels=CIFAR10_CLASSES
        )
        plt.xlabel("Predicted Label", fontsize=14)
        plt.ylabel("True Label", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16)
        plt.show()

        print("\nClassification Report:")
        report = classification_report(
            all_y_true,
            all_y_pred,
            target_names=CIFAR10_CLASSES
        )
        print(report)

if __name__ == "__main__":
    main() 