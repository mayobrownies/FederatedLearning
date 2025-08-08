import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# ICD CODE MAPPINGS
# ============================================================================
ICD9_CHAPTERS = {
    "infectious_parasitic": (1, 139), "neoplasms": (140, 239),
    "endocrine_metabolic": (240, 279), "blood": (280, 289),
    "mental": (290, 319), "nervous": (320, 389), "circulatory": (390, 459),
    "respiratory": (460, 519), "digestive": (520, 579), "genitourinary": (580, 629),
    "pregnancy_childbirth": (630, 679), "skin": (680, 709),
    "musculoskeletal": (710, 739), "congenital": (740, 759),
    "perinatal": (760, 779), "symptoms_ill_defined": (780, 799),
    "injury_poisoning": (800, 999), "e_codes": (1000, 2000),
}

ICD10_CHAPTERS = {
    "infectious_parasitic": ("A00", "B99"), "neoplasms": ("C00", "D49"),
    "blood": ("D50", "D89"), "endocrine_metabolic": ("E00", "E89"),
    "mental": ("F01", "F99"), "nervous": ("G00", "G99"), "eye": ("H00", "H59"),
    "ear": ("H60", "H95"), "circulatory": ("I00", "I99"),
    "respiratory": ("J00", "J99"), "digestive": ("K00", "K95"),
    "skin": ("L00", "L99"), "musculoskeletal": ("M00", "M99"),
    "genitourinary": ("N00", "N99"), "pregnancy_childbirth": ("O00", "O9A"),
    "perinatal": ("P00", "P96"), "congenital": ("Q00", "Q99"),
    "symptoms_ill_defined": ("R00", "R99"), "injury_poisoning": ("S00", "T88"),
    "external_causes": ("V00", "Y99"), "health_factors": ("Z00", "Z99"),
}

# ============================================================================
# DISTRIBUTION ANALYZER
# ============================================================================
class ICDDistributionAnalyzer:
    def __init__(self, diagnoses_file, icd_dict_file, k=75):
        self.diagnoses_file = diagnoses_file
        self.icd_dict_file = icd_dict_file
        self.k = k
        self.diagnoses_df = None
        self.icd_dict_df = None
        
    def load_data(self):
        print("Loading diagnoses data...")
        self.diagnoses_df = pd.read_csv(self.diagnoses_file, compression='gzip')
        print(f"Loaded {len(self.diagnoses_df)} diagnosis records")
        
        print("Loading ICD dictionary...")
        self.icd_dict_df = pd.read_csv(self.icd_dict_file, compression='gzip')
        print(f"Loaded {len(self.icd_dict_df)} ICD codes")
        
        self.diagnoses_df = self.diagnoses_df.merge(
            self.icd_dict_df[['icd_code', 'long_title']], 
            on='icd_code', 
            how='left'
        )
        
    def get_diagnosis_chapter(self, row):
        version = row['icd_version']
        code = str(row['icd_code'])
        
        if version == 9:
            return self.get_chapter_from_icd9(code)
        elif version == 10:
            return self.get_chapter_from_icd10(code)
        return "unknown"
    
    def get_chapter_from_icd10(self, icd_code):
        if not isinstance(icd_code, str) or len(icd_code) < 3:
            return "unknown"
        code_prefix = icd_code[:3].upper()
        for chapter, (start, end) in ICD10_CHAPTERS.items():
            if start <= code_prefix <= end:
                return chapter
        return "unknown"
    
    def get_chapter_from_icd9(self, icd_code):
        if icd_code.startswith('E'):
            try:
                code_num = int(icd_code[1:4])
                if 1000 <= code_num <= 2000: 
                    return "e_codes"
            except (ValueError, IndexError): 
                return "unknown"
        if icd_code.startswith('V'): 
            return "health_factors"
        try:
            code_num = int(float(icd_code))
            for chapter, (start, end) in ICD9_CHAPTERS.items():
                if start <= code_num <= end: 
                    return chapter
        except ValueError: 
            return "unknown"
        return "unknown"
    
    def get_icd_distribution(self, top_k_values=None):
        if self.diagnoses_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if top_k_values is None:
            top_k_values = [10, 20, 50, 100, 200, 500]
            
        # Get overall ICD code frequencies
        icd_counts = self.diagnoses_df['icd_code'].value_counts()
        total_cases = len(self.diagnoses_df['subject_id'].unique())
        
        results = {}
        
        for k in top_k_values:
            # Get top K ICD codes
            top_k_icds = icd_counts.head(k)
            
            # Count cases that have at least one of the top K ICD codes
            top_k_cases = self.diagnoses_df[
                self.diagnoses_df['icd_code'].isin(top_k_icds.index)
            ]['subject_id'].nunique()
            
            results[k] = {
                'total_cases': total_cases,
                'cases_with_top_k': top_k_cases,
                'coverage_percentage': (top_k_cases / total_cases) * 100,
                'top_k_icds': top_k_icds
            }
            
        return results
    
    def get_heterogeneous_distribution(self, top_k_values=None):
        """
        Calculate federated learning client heterogeneity statistics (by medical chapters)
        This shows the percentage of top-K codes that appear in each chapter (method 1)
        
        Args:
            top_k_values: List of K values to analyze (defaults to [10, 20, 50, 100, 200, 500])
            
        Returns:
            Dictionary with heterogeneity statistics per chapter
        """
        if self.diagnoses_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if top_k_values is None:
            top_k_values = [10, 20, 50, 100, 200, 500]
        
        # Add diagnosis chapter to the data
        self.diagnoses_df['diagnosis_chapter'] = self.diagnoses_df.apply(self.get_diagnosis_chapter, axis=1)
        
        # Get overall ICD code frequencies
        icd_counts = self.diagnoses_df['icd_code'].value_counts()
        
        # Group by chapter
        chapter_groups = self.diagnoses_df.groupby('diagnosis_chapter')
        
        results = {}
        
        for k in top_k_values:
            top_k_icds = icd_counts.head(k)
            chapter_results = {}
            
            for chapter, chapter_data in chapter_groups:
                if len(chapter_data) < 100:  # Skip very small chapters
                    continue
                    
                total_chapter_cases = len(chapter_data)
                
                # Add a flag for whether each sample has a top-K code (same as federated learning)
                chapter_data['has_top_k_code'] = chapter_data['icd_code'].isin(top_k_icds.index)
                top_k_count = chapter_data['has_top_k_code'].sum()
                top_k_ratio = (top_k_count / len(chapter_data)) * 100
                
                chapter_results[chapter] = {
                    'total_cases': total_chapter_cases,
                    'samples_with_top_k': top_k_count,
                    'top_k_ratio_percentage': top_k_ratio
                }
            
            results[k] = chapter_results
            
        return results
    
    def plot_coverage_analysis(self, results, heterogeneous_results=None, save_path=None):
        """
        Create visualization focusing only on cases covered by top K ICD codes
        
        Args:
            results: Results from get_icd_distribution()
            heterogeneous_results: Results from get_heterogeneous_distribution()
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'ICD Code Coverage Analysis (K={self.k})', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        k_values = list(results.keys())
        coverage_pct = [results[k]['coverage_percentage'] for k in k_values]
        
        # Plot 1: Overall Coverage Percentage
        axes[0].plot(k_values, coverage_pct, 'o-', linewidth=3, markersize=10, color='#2E86AB')
        axes[0].set_xlabel('Top K ICD Codes')
        axes[0].set_ylabel('Coverage Percentage (%)')
        axes[0].set_title('Cases Covered by Top K ICD Codes')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # Add percentage labels on points
        for i, (x, y) in enumerate(zip(k_values, coverage_pct)):
            axes[0].annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=10, fontweight='bold')
        
        # Plot 2: Heterogeneous Distribution (if available)
        if heterogeneous_results is not None:
            # Use the current K value for heterogeneous analysis
            if self.k in heterogeneous_results:
                chapter_data = heterogeneous_results[self.k]
                
                # Sort chapters by coverage percentage
                sorted_chapters = sorted(chapter_data.items(), 
                                       key=lambda x: x[1]['top_k_ratio_percentage'], 
                                       reverse=True)
                
                chapters = [item[0] for item in sorted_chapters]
                ratios = [item[1]['top_k_ratio_percentage'] for item in sorted_chapters]
                case_counts = [item[1]['total_cases'] for item in sorted_chapters]
                
                # Create horizontal bar plot
                bars = axes[1].barh(range(len(chapters)), ratios, color='#A23B72', alpha=0.8)
                axes[1].set_yticks(range(len(chapters)))
                axes[1].set_yticklabels([ch.replace('_', ' ').title() for ch in chapters], fontsize=9)
                axes[1].set_xlabel('Top-K Ratio Percentage (%)')
                axes[1].set_title(f'Federated Learning Client Heterogeneity (Top {self.k} ICD Codes)')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xlim(0, 100)
                
                # Add percentage labels on bars
                for i, (bar, ratio, count) in enumerate(zip(bars, ratios, case_counts)):
                    axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                                f'{ratio:.1f}% ({count:,})', 
                                va='center', fontsize=8)
            else:
                # If the specific K value isn't available, show a message
                axes[1].text(0.5, 0.5, f'Heterogeneous data not available\nfor K={self.k}', 
                           ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title(f'Coverage by Medical Chapter (Top {self.k} ICD Codes)')
                axes[1].set_xlabel('Coverage Percentage (%)')
        else:
            # If no heterogeneous results at all
            axes[1].text(0.5, 0.5, 'No heterogeneous data available', 
                       ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title(f'Coverage by Medical Chapter (Top {self.k} ICD Codes)')
            axes[1].set_xlabel('Coverage Percentage (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def print_summary_statistics(self, results, heterogeneous_results=None):
        """Print a summary table of the distribution statistics"""
        print("\n" + "="*80)
        print(f"ICD CODE DISTRIBUTION SUMMARY (K={self.k})")
        print("="*80)
        
        print(f"{'Top K':<8} {'Coverage %':<12} {'Cases':<10}")
        print("-" * 40)
        
        for k in sorted(results.keys()):
            r = results[k]
            print(f"{k:<8} {r['coverage_percentage']:<12.1f} {r['cases_with_top_k']:<10}")
        
        if heterogeneous_results is not None and self.k in heterogeneous_results:
            print(f"\n" + "="*80)
            print(f"FEDERATED LEARNING CLIENT HETEROGENEITY (K={self.k})")
            print("="*80)
            
            chapter_data = heterogeneous_results[self.k]
            sorted_chapters = sorted(chapter_data.items(), 
                                   key=lambda x: x[1]['top_k_ratio_percentage'], 
                                   reverse=True)
            
            print(f"{'Chapter':<25} {'Top-K Ratio %':<15} {'Total Cases':<12} {'Samples w/ Top-K':<15}")
            print("-" * 70)
            
            for chapter, data in sorted_chapters:
                chapter_name = chapter.replace('_', ' ').title()
                print(f"{chapter_name:<25} {data['top_k_ratio_percentage']:<15.1f} "
                      f"{data['total_cases']:<12} {data['samples_with_top_k']:<15}")
        
        print("\n" + "="*80)
        print(f"TOP {self.k} ICD CODES")
        print("="*80)
        
        # Get the top K ICD codes from the overall data
        icd_counts = self.diagnoses_df['icd_code'].value_counts()
        top_k_icds = icd_counts.head(self.k)
        
        print(f"{'Rank':<5} {'ICD Code':<12} {'Frequency':<12} {'Description'}")
        print("-" * 80)
        
        for i, (icd_code, freq) in enumerate(top_k_icds.head(self.k).items(), 1):
            description = self.icd_dict_df[self.icd_dict_df['icd_code'] == icd_code]['long_title']
            desc = description.iloc[0] if len(description) > 0 else "Unknown"
            print(f"{i:<5} {icd_code:<12} {freq:<12} {desc[:50]}...")
        
        # Print Word-compatible table format
        print(f"\n{'='*80}")
        print("WORD-FORMATTED TABLE (Copy and paste into Word)")
        print(f"{'='*80}")
        print("Rank   ICD Code   Frequency   Description")
        for i, (icd_code, freq) in enumerate(top_k_icds.head(self.k).items(), 1):
            description = self.icd_dict_df[self.icd_dict_df['icd_code'] == icd_code]['long_title']
            desc = description.iloc[0] if len(description) > 0 else "Unknown"
            print(f"{i}   {icd_code}   {freq:,}   {desc}")
    
    def print_word_tables(self, results, heterogeneous_results=None):
        """Print all data in Word-compatible table format"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE WORD TABLES")
        print(f"{'='*80}")
        
        # Table 1: Overall Coverage Summary
        print("\nTABLE 1: OVERALL COVERAGE SUMMARY")
        print("Top K   Coverage %   Cases")
        for k in sorted(results.keys()):
            r = results[k]
            print(f"{k}   {r['coverage_percentage']:.1f}%   {r['cases_with_top_k']:,}")
        
        # Table 2: Federated Learning Client Heterogeneity
        if heterogeneous_results is not None and self.k in heterogeneous_results:
            print(f"\nTABLE 2: FEDERATED LEARNING CLIENT HETEROGENEITY (K={self.k})")
            print("Chapter   Top-K Ratio %   Total Cases   Samples w/ Top-K")
            
            chapter_data = heterogeneous_results[self.k]
            sorted_chapters = sorted(chapter_data.items(), 
                                   key=lambda x: x[1]['top_k_ratio_percentage'], 
                                   reverse=True)
            
            for chapter, data in sorted_chapters:
                chapter_name = chapter.replace('_', ' ').title()
                print(f"{chapter_name}   {data['top_k_ratio_percentage']:.1f}%   {data['total_cases']:,}   {data['samples_with_top_k']}")
        
        # Table 3: Top K ICD Codes
        print(f"\nTABLE 3: TOP {self.k} ICD CODES")
        print("Rank   ICD Code   Frequency   Description")
        
        icd_counts = self.diagnoses_df['icd_code'].value_counts()
        top_k_icds = icd_counts.head(self.k)
        
        for i, (icd_code, freq) in enumerate(top_k_icds.head(self.k).items(), 1):
            description = self.icd_dict_df[self.icd_dict_df['icd_code'] == icd_code]['long_title']
            desc = description.iloc[0] if len(description) > 0 else "Unknown"
            print(f"{i}   {icd_code}   {freq:,}   {desc}")

def main():
    """Main function to run the analysis"""
    # File paths
    diagnoses_file = "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz"
    icd_dict_file = "mimic-iv-3.1/hosp/d_icd_diagnoses.csv.gz"
    
    k = 100  # Change this value to analyze different numbers of top ICD codes
    
    # Initialize analyzer
    analyzer = ICDDistributionAnalyzer(diagnoses_file, icd_dict_file, k=k)
    
    try:
        # Load data
        analyzer.load_data()
        
        # Analyze distribution for different K values
        print(f"\nAnalyzing ICD code distributions (K={k})...")
        k_values = [10, 20, 50, 100, 200, 500]
        results = analyzer.get_icd_distribution(k_values)
        
        # Analyze heterogeneous distribution
        print("\nAnalyzing heterogeneous distribution by medical chapters...")
        print(f"K values being analyzed: {k_values}")
        heterogeneous_results = analyzer.get_heterogeneous_distribution(k_values)
        print(f"Available K values in heterogeneous results: {list(heterogeneous_results.keys()) if heterogeneous_results else 'None'}")
        
        # Print summary statistics
        analyzer.print_summary_statistics(results, heterogeneous_results)
        
        # Print Word-compatible tables
        analyzer.print_word_tables(results, heterogeneous_results)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        analyzer.plot_coverage_analysis(results, heterogeneous_results, f"icd_coverage_analysis_k{k}.png")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        print("Please ensure the MIMIC-IV data files are in the correct location.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 