"""
Quick test script to verify the enhanced prediction analysis display
"""

import sys
import os
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from heterogeneous_fedprox.task import display_prediction_analysis

def test_prediction_display():
    print("Testing enhanced prediction analysis display...")
    
    # Create mock ICD codes for testing (since TOP_ICD_CODES is empty until data is loaded)
    mock_top_icd_codes = [
        "I50.9", "N17.9", "J44.1", "E87.2", "K92.2", 
        "I10", "N39.0", "J96.00", "E11.9", "F10.10",
        "I48.91", "G93.1", "J18.9", "E66.9", "K59.00",
        "I25.10", "N18.6", "J44.0", "E78.5", "R06.02"
    ]
    
    # Temporarily replace the empty TOP_ICD_CODES with our mock data
    import heterogeneous_fedprox.task as task_module
    original_top_icd_codes = task_module.TOP_ICD_CODES.copy()
    task_module.TOP_ICD_CODES.clear()
    task_module.TOP_ICD_CODES.extend(mock_top_icd_codes)
    
    try:
        # Create sample data that mimics real scenario
        total_samples = 1000
        
        # Simulate predictions: mostly 'other' class with some top-K predictions
        # This mimics the balanced hybrid approach working
        all_predictions = []
        all_labels = []
        
        # 60% other class, 40% distributed across top ICD codes
        other_class_idx = len(mock_top_icd_codes)
        
        # Generate predictions
        for i in range(total_samples):
            if i < 600:  # 60% other
                all_predictions.append(other_class_idx)
                all_labels.append(other_class_idx) 
            else:  # 40% distributed across top-K codes
                class_idx = np.random.randint(0, min(10, len(mock_top_icd_codes)))  # Use first 10 codes
                all_predictions.append(class_idx)
                all_labels.append(class_idx)
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        print(f"Sample data: {len(all_predictions)} samples")
        print(f"Top-K codes available: {len(mock_top_icd_codes)}")
        print(f"Sample predictions range: {all_predictions.min()} to {all_predictions.max()}")
        
        # Test the display function
        display_prediction_analysis(all_predictions, all_labels)
        
        print("\n" + "="*80)
        print("TEST COMPLETED")
        print("="*80)
        print("If you see a nicely formatted table above, the display is working correctly!")
        print("When you run the actual federated learning, look for:")
        print("• 'MODEL PREDICTION ANALYSIS' sections during evaluation")
        print("• Detailed tables showing ICD code predictions")
        print("• Warnings/success messages about prediction diversity")
        
    finally:
        # Restore original TOP_ICD_CODES
        task_module.TOP_ICD_CODES.clear()
        task_module.TOP_ICD_CODES.extend(original_top_icd_codes)

if __name__ == "__main__":
    test_prediction_display() 