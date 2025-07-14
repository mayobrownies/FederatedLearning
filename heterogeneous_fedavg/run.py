import sys
import os
import warnings
import logging
import time

# Add the project root to the Python path to enable absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import flwr as fl
from flwr.common import Context
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ray

from heterogeneous_fedavg.client import get_client
from heterogeneous_fedavg.server import get_strategy
from heterogeneous_fedavg.task import load_and_partition_data, ALL_CHAPTERS, TOP_K_CODES, TOP_ICD_CODES, create_features_and_labels, get_global_feature_space

# Suppress warnings and reduce Ray logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("flwr").setLevel(logging.WARNING)

""" To change the number of ICD codes to predict, modify the number for TOP_K_CODES in the task.py file """

# Simulation parameters
MIMIC_DATA_DIR = "mimic-iv-3.1"
MIN_PARTITION_SIZE = 1000  # Number of samples per chapter; will affect the number of clients

# Model and training configuration
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
PROXIMAL_MU = 0  # Set > 0 for fedprox; = 0 for fedavg

# Advanced training options
USE_ADVANCED_MODEL = True
USE_FOCAL_LOSS = True
HIDDEN_DIMS = [512, 256, 128]
DROPOUT_RATE = 0.3

print(f"Loading data with TOP_K_CODES = {TOP_K_CODES}")
print(f"Advanced model: {USE_ADVANCED_MODEL}, Focal loss: {USE_FOCAL_LOSS}")

VALID_PARTITIONS = load_and_partition_data(data_dir=MIMIC_DATA_DIR, min_partition_size=MIN_PARTITION_SIZE)
NUM_CLIENTS = len(VALID_PARTITIONS)

def analyze_class_distribution():
    """Analyze the class distribution across all clients to identify the root cause of the issue"""
    print("\n" + "="*80)
    print("ANALYZING CLASS DISTRIBUTION ACROSS CLIENTS")
    print("="*80)
    
    global_feature_space = get_global_feature_space(MIMIC_DATA_DIR)
    
    total_samples = 0
    total_top_k_samples = 0
    total_other_samples = 0
    client_stats = []
    
    for client_id, (chapter_name, chapter_data) in enumerate(VALID_PARTITIONS.items()):
        print(f"\nClient {client_id} - Chapter '{chapter_name}':")
        print(f"  Total samples: {len(chapter_data)}")
        
        # Get labels for this client
        _, labels = create_features_and_labels(
            chapter_data, chapter_name, global_feature_space, include_icd_features=True
        )
        
        # Count class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        top_k_count = sum(counts[unique_labels < len(TOP_ICD_CODES)])
        other_count = counts[unique_labels == len(TOP_ICD_CODES)][0] if len(TOP_ICD_CODES) in unique_labels else 0
        
        print(f"  Top-K ICD codes: {top_k_count} samples ({top_k_count/len(labels)*100:.1f}%)")
        print(f"  'Other' class: {other_count} samples ({other_count/len(labels)*100:.1f}%)")
        
        if top_k_count > 0:
            top_k_classes = unique_labels[unique_labels < len(TOP_ICD_CODES)]
            print(f"  Unique top-K classes: {len(top_k_classes)} out of {len(TOP_ICD_CODES)}")
            print(f"  Top-K class distribution: {dict(zip(top_k_classes, counts[unique_labels < len(TOP_ICD_CODES)]))}")
        
        total_samples += len(labels)
        total_top_k_samples += top_k_count
        total_other_samples += other_count
        
        client_stats.append({
            'client_id': client_id,
            'chapter': chapter_name,
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
    
    # Find problematic clients
    problematic_clients = [s for s in client_stats if s['top_k_ratio'] < 0.1]
    print(f"\nProblematic clients (< 10% top-K samples): {len(problematic_clients)}")
    
    if problematic_clients:
        print("These clients will mostly learn to predict 'other' class:")
        for client in problematic_clients:
            print(f"  Client {client['client_id']} ({client['chapter']}): {client['top_k_ratio']*100:.1f}% top-K")
    
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
    """Remove clients with poor data quality for better federated learning"""
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

# Starts the FedAvg federated learning simulation for MIMIC-IV
def main():
    # First, analyze the class distribution to identify issues
    client_stats = analyze_class_distribution()
    
    # Filter out clients with poor data quality (same threshold as partitioning)
    global VALID_PARTITIONS, NUM_CLIENTS
    VALID_PARTITIONS = filter_quality_clients(VALID_PARTITIONS, min_top_k_ratio=0.20)
    NUM_CLIENTS = len(VALID_PARTITIONS)
    
    print(f"\n" + "="*80)
    print("SIMULATION CONFIGURATION")
    print("="*80)
    
    run_config = {
        "num-partitions": NUM_CLIENTS,
        "num-server-rounds": NUM_ROUNDS,
        "local-epochs": LOCAL_EPOCHS,
        "batch-size": BATCH_SIZE,
        "learning-rate": LEARNING_RATE,
        "proximal-mu": PROXIMAL_MU,
        "data-dir": MIMIC_DATA_DIR,
        "min-partition-size": MIN_PARTITION_SIZE,
    }

    # Creates a Flower client representing a single medical partition
    def client_fn(context: Context) -> fl.client.Client:
        partition_id = hash(context.node_id) % NUM_CLIENTS
        return get_client(run_config=run_config, partition_id=partition_id).to_client()

    strategy = get_strategy(run_config)

    print(f"Starting FedAvg simulation with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds...")
    print(f"Available chapters: {list(VALID_PARTITIONS.keys())}")
    print(f"Configuration: lr={LEARNING_RATE}, mu={PROXIMAL_MU}, local_epochs={LOCAL_EPOCHS}")
    print(f"Total ICD codes in model: {len(TOP_ICD_CODES)}")
    
    # Shutdown any existing Ray instance to avoid conflicts
    try:
        ray.shutdown()
        print("Existing Ray instance shut down successfully")
    except Exception as e:
        print(f"No existing Ray instance to shut down: {e}")
    
    # Configure Ray with more robust parameters
    ray_init_args = {
        "ignore_reinit_error": True,
        "log_to_driver": False,
        "include_dashboard": False,
        "num_cpus": None,  # Use all available CPUs
        "object_store_memory": None,  # Use default memory allocation
        "_temp_dir": None,  # Use default temp directory
        "local_mode": False,  # Use cluster mode for better stability
    }
    
    # Reduce client resources to avoid oversubscription
    client_resources = {"num_cpus": 1}  # Reduced from 2 to 1
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args
    )

    print("Simulation finished.")
    
    time.sleep(5)  # Wait for all distributed threads to finish logging
    
    print("\n" + "="*80)
    print("FINAL SIMULATION RESULTS")
    print("="*80)
    
    # Plot training loss over rounds
    train_losses_found = False
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        if "train_loss" in history.metrics_distributed_fit:
            train_losses = history.metrics_distributed_fit["train_loss"]
            rounds = [r for r, _ in train_losses]
            losses = [loss for _, loss in train_losses]
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, losses, 'b-', linewidth=2, marker='o', label='FedAvg Training Loss')
            plt.xlabel('Communication Rounds', fontsize=12)
            plt.ylabel('Training Loss', fontsize=12)
            plt.title('Training Loss vs Communication Rounds (MIMIC-IV)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            print(f"Training loss progression: {losses}")
            print(f"Final training loss: {losses[-1]:.4f}")
            train_losses_found = True
    
    if not train_losses_found:
        print("Training loss data not found in history.metrics_distributed_fit")
        print("Available history attributes:")
        for attr in dir(history):
            if not attr.startswith('_') and hasattr(history, attr):
                attr_value = getattr(history, attr)
                if attr_value:
                    print(f"  {attr}: {type(attr_value)}")
    
    # Display final metrics with better formatting
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
        
        # Show training progression
        print("\nTRAINING PROGRESSION")
        print("-" * 40)
        if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit and "train_loss" in history.metrics_distributed_fit:
            train_losses = history.metrics_distributed_fit["train_loss"]
            losses = [loss for _, loss in train_losses]
            print(f"│ Training loss: {losses[0]:.4f} → {losses[-1]:.4f}")
            print(f"│ Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        
        # Accuracy progression
        accuracy_history = history.metrics_distributed.get("accuracy", [])
        if len(accuracy_history) > 1:
            accuracies = [acc for _, acc in accuracy_history]
            print(f"│ Accuracy: {accuracies[0]:.4f} → {accuracies[-1]:.4f}")
            print(f"│ Accuracy change: {((accuracies[-1] - accuracies[0]) / accuracies[0] * 100):+.1f}%")
            
            # Check for stagnant metrics
            if all(abs(acc - accuracies[0]) < 1e-6 for acc in accuracies):
                print("│    WARNING: Accuracy is stagnant!")
                print("│    Model may only be predicting 'other' class")
            else:
                print("│    Metrics are changing")
        
        print("-" * 40)
        
        # Final prediction analysis
        print("\nFINAL MODEL PREDICTION DISTRIBUTION")
        print("-" * 40)
        print("Note: Detailed prediction analysis should appear above during final round.")
        print("Look for 'TESTING GLOBAL AGGREGATED MODEL' section.")
        
    else:
        print("Could not retrieve final metrics. The simulation might have failed.")
        
    print("\n" + "="*80)
    
    # Clean up Ray resources
    try:
        ray.shutdown()
        print("Ray instance shut down successfully")
    except Exception as e:
        print(f"Error shutting down Ray: {e}")

if __name__ == "__main__":
    main() 