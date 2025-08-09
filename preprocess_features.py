import argparse
import sys
import os
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.task import preprocess_all_medical_features

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================
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
    print("NOTE: Diagnoses and procedures are excluded from training data")
    print("      to prevent data leakage in ICD code prediction task.")
    print()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("Please ensure you have the MIMIC-IV data extracted in the correct location.")
        sys.exit(1)
    
    required_files = [
        'hosp/admissions.csv.gz',
        'hosp/patients.csv.gz',
        'hosp/prescriptions.csv.gz',
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
        print()
        print("NOTE: diagnoses_icd.csv.gz and procedures_icd.csv.gz are not required")
        print("      as they are excluded to prevent data leakage in ICD prediction.")
        sys.exit(1)
    
    print("All required files found")
    print()
    
    try:
        start_time = time.time()
        
        print("Starting medical feature preprocessing...")
        print("This may take several minutes on the first run...")
        print()
        print("Features being processed:")
        print("  ✓ Lab values (clinical indicators)")
        print("  ✓ Medications (drug categories and interactions)")
        print("  ✓ ICU monitoring (vital signs, interventions)")
        print("  ✓ Microbiology (infection markers)")
        print("  ✓ Severity indicators (DRG codes, services)")
        print("  ✗ Diagnoses (excluded - would leak ICD information)")
        print("  ✗ Procedures (excluded - would leak ICD information)")
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
        print("Data leakage prevention measures:")
        print("  - Primary diagnoses excluded (would reveal target ICD codes)")
        print("  - Secondary diagnoses excluded (would reveal related ICD codes)")
        print("  - Procedures excluded (contain ICD procedure codes)")
        print("  - Only clinical indicators and administrative data included")
        print()
        print("You can now run the federated learning simulation with clean medical features!")
        print("   python shared/run.py")
        print()
        
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 