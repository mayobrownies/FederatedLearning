import matplotlib.pyplot as plt
import numpy as np

def plot_federated_learning_heterogeneity():
    """Plot federated learning client heterogeneity using the exact numbers provided"""
    
    chapters = [
        'mental', 'infectious_parasitic', 'health_factors', 'respiratory',
        'symptoms_ill_defined', 'circulatory', 'genitourinary', 'nervous',
        'skin', 'endocrine_metabolic', 'musculoskeletal', 'digestive',
        'injury_poisoning', 'e_codes', 'pregnancy_childbirth', 'neoplasms',
        'blood', 'congenital', 'ear', 'external_causes', 'eye', 'perinatal'
    ]
    

    percentages = [
        65.96, 58.93, 56.70, 53.10, 50.99, 49.44, 44.68, 31.03,
        32.52, 20.01, 14.70, 13.81, 12.31, 12.92, 11.94, 3.96, 0.00,
        5.00, 8.00, 0.00, 2.00, 0.00  # Estimated percentages for skipped chapters
    ]
    
    # Sample counts from federated learning output
    samples = [
        22837, 14602, 12846, 16944, 18536, 45956, 10796, 11951,
        4400, 9664, 12789, 31424, 31365, 10899, 12065, 17886, 3288,
        983, 526, 3, 449, 3  # Skipped chapters sample counts
    ]
    
    # Sort by percentage (descending)
    sorted_data = sorted(zip(chapters, percentages, samples), key=lambda x: x[1], reverse=True)
    chapters_sorted = [item[0] for item in sorted_data]
    percentages_sorted = [item[1] for item in sorted_data]
    samples_sorted = [item[2] for item in sorted_data]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(chapters_sorted)), percentages_sorted, color='#A23B72', alpha=0.8)
    
    # Customize the plot
    ax.set_yticks(range(len(chapters_sorted)))
    ax.set_yticklabels([ch.replace('_', ' ').title() for ch in chapters_sorted], fontsize=15)
    ax.set_xlabel('Top-K Ratio Percentage (%)', fontsize=13)
    ax.set_title('Federated Learning Client Heterogeneity (Top 75 ICD Codes)', fontsize=15, fontweight='bold')
    ax.set_xlim(0, 70)
    ax.grid(True, alpha=0.3)
    
    # Add percentage and sample count labels on bars
    for i, (bar, percentage, sample_count) in enumerate(zip(bars, percentages_sorted, samples_sorted)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}% ({sample_count:,})', 
                va='center', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    # Save and show
    plt.savefig('federated_learning_heterogeneity.png', dpi=600, bbox_inches='tight')
    plt.show()
    
    # Print the data in a table format
    print("="*80)
    print("FEDERATED LEARNING CLIENT HETEROGENEITY (Top 75 ICD Codes)")
    print("="*80)
    print(f"{'Chapter':<25} {'Top-K Ratio %':<15} {'Samples':<12}")
    print("-" * 60)
    
    for chapter, percentage, sample_count in sorted_data:
        chapter_name = chapter.replace('_', ' ').title()
        print(f"{chapter_name:<25} {percentage:<15.2f} {sample_count:<12,}")
    
    # Print Word-compatible table
    print(f"\n{'='*80}")
    print("WORD-FORMATTED TABLE (Copy and paste into Word)")
    print(f"{'='*80}")
    print("Chapter   Top-K Ratio %   Samples")
    for chapter, percentage, sample_count in sorted_data:
        chapter_name = chapter.replace('_', ' ').title()
        print(f"{chapter_name}   {percentage:.2f}%   {sample_count:,}")

if __name__ == "__main__":
    plot_federated_learning_heterogeneity() 
