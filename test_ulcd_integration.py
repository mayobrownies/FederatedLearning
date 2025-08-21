# Test script to verify ULCD integration with the federated learning framework

import torch
import numpy as np
from transformers import BertTokenizer
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_ulcd_components():
    """Test basic ULCD components"""
    print("="*60)
    print("TESTING ULCD COMPONENTS")
    print("="*60)
    
    try:
        from shared.ulcd_components import MultimodalULCDClient, ULCDServer, create_dummy_multimodal_data
        print("[OK] Successfully imported ULCD components")
        
        # Test multimodal client creation
        client = MultimodalULCDClient(tabular_dim=20, latent_dim=64, task_out=76)
        print("[OK] Created MultimodalULCDClient")
        
        # Test server creation
        server = ULCDServer(latent_dim=64)
        print("[OK] Created ULCDServer")
        
        # Test dummy data creation
        dummy_tabular = torch.randn(10, 20)  # 10 samples, 20 features
        dummy_data = create_dummy_multimodal_data(dummy_tabular, num_samples=10)
        print("[OK] Created dummy multimodal data")
        
        # Test forward pass
        text_input = {
            'input_ids': dummy_data['text_input_ids'][:5],  # First 5 samples
            'attention_mask': dummy_data['text_attention_mask'][:5]
        }
        tabular_input = dummy_tabular[:5]
        
        z, y = client(tabular_input, text_input)
        print(f"[OK] Forward pass successful: latent shape {z.shape}, output shape {y.shape}")
        
        # Test latent summary
        from torch.utils.data import DataLoader, TensorDataset
        dummy_labels = torch.ones(10, 1)
        dataset = TensorDataset(dummy_tabular, dummy_data['text_input_ids'], dummy_data['text_attention_mask'], dummy_labels)
        loader = DataLoader(dataset, batch_size=5)
        
        latent_summary = client.get_latent_summary(loader)
        print(f"[OK] Latent summary extracted: shape {latent_summary.shape}")
        
        # Test server aggregation
        latents = [torch.randn(64) for _ in range(3)]
        server.aggregate_latents(latents)
        print("[OK] Server aggregation successful")
        
        # Test anomaly detection
        trusted, flagged = server.detect_anomalies(latents, threshold=0.1)
        print(f"[OK] Anomaly detection: {len(trusted)} trusted, {len(flagged)} flagged")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing ULCD components: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_factory():
    """Test model creation through the factory"""
    print("\n" + "="*60)
    print("TESTING MODEL FACTORY INTEGRATION")
    print("="*60)
    
    try:
        from shared.models import get_model, get_model_type
        
        # Test traditional ULCD
        ulcd_model = get_model("ulcd", input_dim=100, output_dim=76, latent_dim=64)
        print(f"[OK] Created traditional ULCD: {get_model_type(ulcd_model)}")
        
        # Test multimodal ULCD
        ulcd_multimodal = get_model("ulcd_multimodal", input_dim=100, output_dim=76, latent_dim=64, tabular_dim=20)
        print(f"[OK] Created multimodal ULCD: {get_model_type(ulcd_multimodal)}")
        
        # Test parameter counts
        ulcd_params = sum(p.numel() for p in ulcd_model.parameters())
        ulcd_mm_params = sum(p.numel() for p in ulcd_multimodal.parameters())
        
        print(f"[OK] Traditional ULCD parameters: {ulcd_params:,}")
        print(f"[OK] Multimodal ULCD parameters: {ulcd_mm_params:,}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing model factory: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_factory():
    """Test strategy creation"""
    print("\n" + "="*60)
    print("TESTING STRATEGY FACTORY INTEGRATION")
    print("="*60)
    
    try:
        from shared.strategies import get_strategy
        
        # Test traditional strategies
        fedavg_strategy = get_strategy("fedavg")
        print("[OK] Created FedAvg strategy")
        
        fedprox_strategy = get_strategy("fedprox", proximal_mu=0.1)
        print("[OK] Created FedProx strategy")
        
        # Test ULCD strategy
        run_config = {"cached_partitions": {"client1": None, "client2": None}}
        ulcd_strategy = get_strategy("ulcd", run_config=run_config, latent_dim=64)
        print("[OK] Created ULCD strategy")
        
        print(f"[OK] ULCD strategy type: {type(ulcd_strategy).__name__}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing strategy factory: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_handling():
    """Test data handling for multimodal ULCD"""
    print("\n" + "="*60)
    print("TESTING MULTIMODAL DATA HANDLING")
    print("="*60)
    
    try:
        # Test tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        sample_texts = [
            "Patient presents with acute symptoms",
            "Vital signs are stable, no immediate concerns",
            "Treatment plan includes medication and monitoring"
        ]
        
        encodings = tokenizer(
            sample_texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        print(f"[OK] Text tokenization successful: {encodings['input_ids'].shape}")
        print(f"[OK] Attention mask shape: {encodings['attention_mask'].shape}")
        
        # Test with multimodal client
        from shared.ulcd_components import MultimodalULCDClient
        
        client = MultimodalULCDClient(tabular_dim=6, latent_dim=64, task_out=2)
        
        # Simulate vital signs data
        tabular_data = torch.randn(3, 6)  # 3 patients, 6 vital signs
        text_input = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        z, y = client(tabular_data, text_input)
        print(f"[OK] Multimodal forward pass: latent {z.shape}, output {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing data handling: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("ULCD INTEGRATION TEST SUITE")
    print("="*80)
    
    results = {
        "ULCD Components": test_ulcd_components(),
        "Model Factory": test_model_factory(), 
        "Strategy Factory": test_strategy_factory(),
        "Data Handling": test_data_handling()
    }
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{test_name:<20}: {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("ALL TESTS PASSED! ULCD integration is ready.")
        print("\nNext steps:")
        print("1. Update shared/run.py to use FL_STRATEGY = 'ulcd' for comparison")
        print("2. Set MODELS_TO_TEST = {'ulcd', 'ulcd_multimodal', 'logistic'}")
        print("3. Run your federated learning experiment!")
    else:
        print("Some tests failed. Please fix the errors before proceeding.")
    
    print("="*80)

if __name__ == "__main__":
    main()