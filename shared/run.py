import sys
import os
import warnings
import logging
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flwr as fl
from flwr.common import Context
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ray

from shared.client import get_client
from shared.strategies import get_strategy
from shared.task import load_and_partition_data, TOP_ICD_CODES, create_features_and_labels, get_global_feature_space

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("flwr").setLevel(logging.WARNING)

# ==============================================================================
# CONFIGURATION OPTIONS - Customize these to run different experiments
# ==============================================================================

# Data and partitioning configuration
MIMIC_DATA_DIR = "mimic-iv-3.1"
MIN_PARTITION_SIZE = 1000

# Model and training configuration
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.0002

# Choose FL strategy: ["fedavg", "fedprox"]
AVAILABLE_STRATEGIES = ["fedavg", "fedprox"]
FL_STRATEGY = AVAILABLE_STRATEGIES[0]

# Choose partitioning scheme: ["heterogeneous", "balanced"]
AVAILABLE_PARTITIONING = ["heterogeneous", "balanced"]
PARTITIONING_SCHEME = AVAILABLE_PARTITIONING[0]

# Choose model from the list: ["basic", "advanced"]
AVAILABLE_MODELS = ["basic", "advanced"]
MODEL_NAME = AVAILABLE_MODELS[1]

# FedProx configuration (only used when FL_STRATEGY = "fedprox")
PROXIMAL_MU = 0.1  # Set > 0 for fedprox; = 0 for fedavg

# ICD prediction configuration
TOP_K_CODES = 100

# Advanced options
USE_FOCAL_LOSS = False
HIDDEN_DIMS = [512, 256, 128]
DROPOUT_RATE = 0.3

# ==============================================================================
# ANALYSIS AND FILTERING FUNCTIONS
# ==============================================================================

def analyze_class_distribution(partitions):
    """Analyze class distribution across all clients."""
    print("\n" + "="*80)
    print("ANALYZING CLASS DISTRIBUTION ACROSS CLIENTS")
    print("="*80)
    
    global_feature_space = get_global_feature_space(MIMIC_DATA_DIR)
    
    total_samples = 0
    total_top_k_samples = 0
    total_other_samples = 0
    client_stats = []
    
    for client_id, (partition_name, partition_data) in enumerate(partitions.items()):
        print(f"\nClient {client_id} - Partition '{partition_name}':")
        print(f"  Total samples: {len(partition_data)}")
        
        _, labels = create_features_and_labels(
            partition_data, partition_name, global_feature_space, include_icd_features=True
        )
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        top_k_count = sum(counts[unique_labels < len(TOP_ICD_CODES)])
        other_count = counts[unique_labels == len(TOP_ICD_CODES)][0] if len(TOP_ICD_CODES) in unique_labels else 0
        
        print(f"  Top-K ICD codes: {top_k_count} samples ({top_k_count/len(labels)*100:.1f}%)")
        print(f"  'Other' class: {other_count} samples ({other_count/len(labels)*100:.1f}%)")
        
        if top_k_count > 0:
            top_k_classes = unique_labels[unique_labels < len(TOP_ICD_CODES)]
            print(f"  Unique top-K classes: {len(top_k_classes)} out of {len(TOP_ICD_CODES)}")
        
        total_samples += len(labels)
        total_top_k_samples += top_k_count
        total_other_samples += other_count
        
        client_stats.append({
            'client_id': client_id,
            'partition': partition_name,
            'total_samples': len(labels),
            'top_k_samples': top_k_count,
            'other_samples': other_count,
            'top_k_ratio': top_k_count/len(labels),
            'unique_top_k_classes': len(unique_labels[unique_labels < len(TOP_ICD_CODES)])
        })
    
    print(f"\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total samples across all clients: {total_samples}")
    print(f"Total top-K samples: {total_top_k_samples} ({total_top_k_samples/total_samples*100:.1f}%)")
    print(f"Total 'other' samples: {total_other_samples} ({total_other_samples/total_samples*100:.1f}%)")
    
    problematic_clients = [s for s in client_stats if s['top_k_ratio'] < 0.1]
    print(f"\nProblematic clients (< 10% top-K samples): {len(problematic_clients)}")
    
    if problematic_clients:
        print("These clients will mostly learn to predict 'other' class:")
        for client in problematic_clients:
            print(f"  Client {client['client_id']} ({client['partition']}): {client['top_k_ratio']*100:.1f}% top-K")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if total_top_k_samples / total_samples < 0.3:
        print("     SEVERE CLASS IMBALANCE DETECTED!")
        print("   - Most samples are 'other' class")
        print("   - Model will likely only predict 'other'")
        print("   - Solutions:")
        print("     1. Reduce TOP_K_CODES to focus on most common codes")
        print("     2. Use different partitioning strategy")
        print("     3. Apply class balancing techniques")
        print("     4. Use focal loss instead of cross-entropy")
    
    if len(problematic_clients) > len(client_stats) * 0.5:
        print("     TOO MANY PROBLEMATIC CLIENTS!")
        print("   - Most clients have very few top-K samples")
        print("   - Solutions:")
        print("     1. Increase MIN_PARTITION_SIZE")
        print("     2. Use different data partitioning")
        print("     3. Implement client selection based on data quality")
    
    return client_stats

def filter_quality_clients(partitions, min_top_k_ratio=0.20):
    """Remove clients with poor data quality (for balanced partitioning)."""
    if PARTITIONING_SCHEME == "heterogeneous":
        # For heterogeneous partitioning, apply minimal filtering
        print(f"\nApplying minimal safety filtering for heterogeneous partitioning...")
        safe_partitions = {}
        for name, data in partitions.items():
            if len(data) >= 500:  # Lower threshold for heterogeneous
                safe_partitions[name] = data
                print(f"  ✓ Keeping partition '{name}': {len(data)} samples")
            else:
                print(f"  ✗ Filtering partition '{name}': {len(data)} samples (too small, may cause crashes)")
        return safe_partitions
    
    else:
        # For balanced partitioning, apply quality filtering
        print(f"\nFiltering clients with <{min_top_k_ratio*100:.0f}% top-K samples...")
        
        global_feature_space = get_global_feature_space(MIMIC_DATA_DIR)
        quality_partitions = {}
        filtered_count = 0
        
        for name, data in partitions.items():
            _, labels = create_features_and_labels(data, name, global_feature_space, include_icd_features=True)
            top_k_count = sum(1 for label in labels if label < len(TOP_ICD_CODES))
            top_k_ratio = top_k_count / len(labels)
            
            if top_k_ratio >= min_top_k_ratio:
                quality_partitions[name] = data
                print(f"  ✓ Keeping client '{name}': {len(data)} samples, {top_k_ratio:.1%} top-K ratio")
            else:
                print(f"  ✗ Filtering out client '{name}': {len(data)} samples, {top_k_ratio:.1%} top-K ratio (below {min_top_k_ratio:.0%})")
                filtered_count += 1
        
        print(f"\nFiltered {filtered_count} clients, kept {len(quality_partitions)} quality clients")
        return quality_partitions

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Start the federated learning simulation with the configured options."""
    
    print(f"\n" + "="*80)
    print("FEDERATED LEARNING SIMULATION CONFIGURATION")
    print("="*80)
    print(f"FL Strategy: {FL_STRATEGY}")
    print(f"Partitioning: {PARTITIONING_SCHEME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Top-K ICD Codes: {TOP_K_CODES}")
    if FL_STRATEGY == "fedprox":
        print(f"Proximal μ: {PROXIMAL_MU}")
    print(f"Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
    print("="*80)
    
    # Load and partition data
    print(f"Loading data with TOP_K_CODES = {TOP_K_CODES}")
    VALID_PARTITIONS = load_and_partition_data(
        data_dir=MIMIC_DATA_DIR, 
        min_partition_size=MIN_PARTITION_SIZE,
        partition_scheme=PARTITIONING_SCHEME,
        top_k_codes=TOP_K_CODES
    )
    
    if not VALID_PARTITIONS:
        print("No valid partitions found. Please check your data directory and configuration.")
        return
    
    # Apply filtering based on partitioning scheme (this also does analysis)
    VALID_PARTITIONS = filter_quality_clients(VALID_PARTITIONS, min_top_k_ratio=0.20)
    NUM_CLIENTS = len(VALID_PARTITIONS)
    
    if NUM_CLIENTS == 0:
        print("No clients remaining after filtering. Please adjust your configuration.")
        return
    
    print(f"Using {NUM_CLIENTS} clients for {PARTITIONING_SCHEME} {FL_STRATEGY} simulation")
    
    # Create run configuration - using both naming conventions for compatibility
    run_config = {
        "strategy": FL_STRATEGY,
        "partitioning": PARTITIONING_SCHEME,
        "model_name": MODEL_NAME,
        "proximal_mu": PROXIMAL_MU if FL_STRATEGY == "fedprox" else 0.0,
        "num_server_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "top_k_codes": TOP_K_CODES,
        "min_partition_size": MIN_PARTITION_SIZE,
        "data_dir": MIMIC_DATA_DIR,
        # Add hyphenated versions for compatibility
        "local-epochs": LOCAL_EPOCHS,
        "batch-size": BATCH_SIZE,
        "learning-rate": LEARNING_RATE,
        "min-partition-size": MIN_PARTITION_SIZE,
        "data-dir": MIMIC_DATA_DIR,
        # Pass the pre-computed partitions to avoid recomputing in each client
        "cached_partitions": VALID_PARTITIONS,
    }

    def client_fn(context: Context) -> fl.client.Client:
        partition_id = hash(context.node_id) % NUM_CLIENTS
        return get_client(run_config=run_config, partition_id=partition_id).to_client()

    strategy = get_strategy(FL_STRATEGY, run_config=run_config, proximal_mu=run_config["proximal_mu"])

    print(f"\nStarting {FL_STRATEGY.upper()} simulation with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds...")
    print(f"Available partitions: {list(VALID_PARTITIONS.keys())}")
    
    strategy_desc = f"μ={PROXIMAL_MU}" if FL_STRATEGY == "fedprox" else "no proximal regularization"
    print(f"Configuration: lr={LEARNING_RATE}, {strategy_desc}, local_epochs={LOCAL_EPOCHS}")
    print(f"Total ICD codes in model: {len(TOP_ICD_CODES)}")
    
    # Initialize Ray
    try:
        ray.shutdown()
        print("Existing Ray instance shut down successfully")
    except Exception as e:
        print(f"No existing Ray instance to shut down: {e}")
    
    ray_init_args = {
        "ignore_reinit_error": True,
        "log_to_driver": False,
        "include_dashboard": False,
        "num_cpus": None,
        "object_store_memory": 1000000000,  # 1GB object store
        "_temp_dir": None,
        "local_mode": False,
    }
    
    client_resources = {"num_cpus": 1, "memory": 2000000000}  # 2GB per client
    
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args
    )

    print("Simulation finished.")
    
    time.sleep(5)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL SIMULATION RESULTS")
    print("="*80)
    
    # Training loss progression
    train_losses_found = False
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        if "train_loss" in history.metrics_distributed_fit:
            train_losses = history.metrics_distributed_fit["train_loss"]
            rounds = [r for r, _ in train_losses]
            losses = [loss for _, loss in train_losses]
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, losses, 'b-', linewidth=2, marker='o', 
                    label=f'{FL_STRATEGY.upper()} Training Loss')
            plt.xlabel('Communication Rounds', fontsize=12)
            plt.ylabel('Training Loss', fontsize=12)
            plt.title(f'Training Loss vs Communication Rounds - {FL_STRATEGY.upper()} ({PARTITIONING_SCHEME})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            print(f"Training loss progression: {losses}")
            print(f"Final training loss: {losses[-1]:.4f}")
            train_losses_found = True
    
    if not train_losses_found:
        print("Training loss data not found in history.metrics_distributed_fit")
    
    # Final performance metrics
    if hasattr(history, 'metrics_distributed') and history.metrics_distributed and "accuracy" in history.metrics_distributed:
        print("FINAL PERFORMANCE METRICS")
        print("-" * 40)
        
        final_accuracy = history.metrics_distributed["accuracy"][-1][1]
        print(f"│ Accuracy:   {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        if "f1" in history.metrics_distributed:
            final_f1 = history.metrics_distributed["f1"][-1][1]
            print(f"│ F1 Score:   {final_f1:.4f}")
        
        if "precision" in history.metrics_distributed:
            final_precision = history.metrics_distributed["precision"][-1][1]
            print(f"│ Precision:  {final_precision:.4f}")
        
        if "recall" in history.metrics_distributed:
            final_recall = history.metrics_distributed["recall"][-1][1]
            print(f"│ Recall:     {final_recall:.4f}")
        
        print("-" * 40)
        
        # Training progression analysis
        print("\nTRAINING PROGRESSION")
        print("-" * 40)
        if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit and "train_loss" in history.metrics_distributed_fit:
            train_losses = history.metrics_distributed_fit["train_loss"]
            losses = [loss for _, loss in train_losses]
            print(f"│ Training loss: {losses[0]:.4f} → {losses[-1]:.4f}")
            print(f"│ Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        
        accuracy_history = history.metrics_distributed.get("accuracy", [])
        if len(accuracy_history) > 1:
            accuracies = [acc for _, acc in accuracy_history]
            print(f"│ Accuracy: {accuracies[0]:.4f} → {accuracies[-1]:.4f}")
            print(f"│ Accuracy change: {((accuracies[-1] - accuracies[0]) / accuracies[0] * 100):+.1f}%")
            
            if all(abs(acc - accuracies[0]) < 1e-6 for acc in accuracies):
                print("│    WARNING: Accuracy is stagnant!")
                print("│    Model may only be predicting 'other' class")
            else:
                print("│    Metrics are changing - good learning progress")
        
        print("-" * 40)
        
    else:
        print("Could not retrieve final metrics. The simulation might have failed.")
        
    print("\n" + "="*80)
    print(f"EXPERIMENT COMPLETED: {FL_STRATEGY.upper()} with {PARTITIONING_SCHEME} partitioning")
    print("="*80)
    
    # Cleanup Ray
    try:
        ray.shutdown()
        print("Ray instance shut down successfully")
    except Exception as e:
        print(f"Error shutting down Ray: {e}")

if __name__ == "__main__":
    main()