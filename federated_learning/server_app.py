import flwr as fl
from flwr.server.server_config import ServerConfig
from flwr.common import Context, Metrics
from flwr.common.parameter import ndarrays_to_parameters
from typing import Dict, Any, Optional, List, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import EvaluateRes

from .task import get_model, get_weights

# Aggregates metrics by calculating a weighted average.
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    num_total_examples = sum(num_examples for num_examples, _ in metrics)

    if num_total_examples == 0:
        return {}

    weights = [num_examples / num_total_examples for num_examples, _ in metrics]
    
    aggregated_metrics: Metrics = {}
    for i, (_, m) in enumerate(metrics):
        for key, value in m.items():
            if isinstance(value, (int, float)):
                current_value = aggregated_metrics.get(key, 0.0)
                aggregated_metrics[key] = current_value + weights[i] * value

    return aggregated_metrics

# Aggregates evaluation metrics using a weighted average.
def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    return weighted_average(metrics)

# Creates and returns a federated learning strategy.
def get_strategy(run_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
    num_partitions = run_config.get("num-partitions", 2)
    
    initial_model = get_model()
    initial_parameters = ndarrays_to_parameters(get_weights(initial_model))

    # Returns the training configuration for a given server round.
    def fit_config_fn(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
            "local_epochs": run_config.get("local-epochs", 1),
            "batch_size": run_config.get("batch-size", 32),
            "learning_rate": run_config.get("learning-rate", 0.01),
        }
        return config

    # Returns the evaluation configuration for a given server round.
    def evaluate_config_fn(server_round: int) -> Dict[str, Any]:
        config = {"server_round": server_round}
        if server_round == run_config.get("num-server-rounds"):
            config["get_predictions"] = True
        return config

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_partitions,
        min_evaluate_clients=num_partitions,
        min_available_clients=num_partitions,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    return strategy

# Defines the server-side logic for a single federated learning run.
def server_fn(context: Context):
    run_config = context.run_config
    strategy = get_strategy(run_config)
    num_server_rounds = run_config.get("num-server-rounds", 5)
    config = ServerConfig(num_rounds=num_server_rounds)

    return fl.server.ServerAppComponents(
        strategy=strategy,
        config=config,
    )

app = fl.server.ServerApp(server_fn=server_fn) 