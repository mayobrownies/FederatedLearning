import argparse
import sys
import os
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hybrid_fedprox.task import preprocess_all_medical_features

# Main preprocessing function
def main():
    parser = argparse.ArgumentParser(description="Preprocess medical features for MIMIC-IV federated learning")
    parser.add_argument('--force-recompute', action='store_true', 
                       help='Force recomputation of all features (ignore cache)')
    parser.add_argument('--data-dir', default='mimic-iv-3.1', 
                       help='Path to MIMIC-IV data directory (default: mimic-iv-3.1)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MIMIC-IV Medical Features Preprocessing")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Force recompute: {args.force_recompute}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("Please ensure you have the MIMIC-IV data extracted in the correct location.")
        sys.exit(1)
    
    # Check for key files
    required_files = [
        'hosp/admissions.csv.gz',
        'hosp/diagnoses_icd.csv.gz',
        'hosp/patients.csv.gz',
        'hosp/prescriptions.csv.gz',
        'hosp/procedures_icd.csv.gz',
        'hosp/drgcodes.csv.gz',
        'hosp/services.csv.gz'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("Please ensure you have all MIMIC-IV files extracted.")
        sys.exit(1)
    
    print("All required files found")
    print()
    
    # Start preprocessing
    try:
        start_time = time.time()
        
        print("Starting medical feature preprocessing...")
        print("This may take several minutes on the first run...")
        print()
        
        results = preprocess_all_medical_features(
            data_dir=args.data_dir,
            force_recompute=args.force_recompute
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print()
        print("=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print()
        
        # Show feature counts
        total_features = 0
        for feature_type, features in results.items():
            if not features.empty:
                print(f"{feature_type.capitalize()} features: {len(features.columns)} columns, {len(features)} rows")
                total_features += len(features.columns)
        
        print(f"Total medical features: {total_features}")
        print()
        print("You can now run the federated learning simulation with fast medical feature access!")
        print("   python heterogeneous_mimic/run.py")
        print()
        
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 