import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# HETEROGENEITY VISUALIZATION
# ============================================================================
def plot_federated_learning_heterogeneity():
    
    chapters = [
        'mental', 'infectious_parasitic', 'health_factors', 'symptoms_ill_defined', 
        'circulatory', 'genitourinary', 'respiratory', 'endocrine_metabolic', 
        'nervous', 'skin', 'digestive', 'musculoskeletal', 'e_codes', 
        'pregnancy_childbirth', 'injury_poisoning', 'neoplasms', 'blood',
        'congenital', 'ear', 'external_causes', 'eye', 'perinatal'
    ]

    percentages = [
        65.96, 62.55, 60.65, 56.86, 56.07, 53.99, 53.10, 35.55,
        34.88, 32.52, 24.77, 18.56, 17.33, 11.94, 12.31, 3.96, 0.00,
        0.0, 0.0, 0.0, 0.0, 0.0
    ]
    
    samples = [
        22837, 14602, 12846, 18536, 45956, 10796, 16944, 9664,
        11951, 4400, 31424, 12789, 10899, 12065, 31365, 17886, 3288,
        983, 526, 3, 449, 3
    ]
    
    sorted_data = sorted(zip(chapters, percentages, samples), key=lambda x: x[1], reverse=True)
    chapters_sorted = [item[0] for item in sorted_data]
    percentages_sorted = [item[1] for item in sorted_data]
    samples_sorted = [item[2] for item in sorted_data]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    bars = ax.barh(range(len(chapters_sorted)), percentages_sorted, color='#A23B72', alpha=0.8)
    
    ax.set_yticks(range(len(chapters_sorted)))
    ax.set_yticklabels([ch.replace('_', ' ').title() for ch in chapters_sorted], fontsize=15)
    ax.set_xlabel('Top-K Ratio Percentage (%)', fontsize=13)
    ax.set_title('Federated Learning Client Heterogeneity (Top 100 ICD Codes)', fontsize=15, fontweight='bold')
    ax.set_xlim(0, 70)
    ax.grid(True, alpha=0.3)
    
    for i, (bar, percentage, sample_count) in enumerate(zip(bars, percentages_sorted, samples_sorted)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}% ({sample_count:,})', 
                va='center', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig('federated_learning_heterogeneity.png', dpi=600, bbox_inches='tight')
    plt.show()
    
    # ============================================================================
    # RESULTS TABLE
    # ============================================================================
    print("="*80)
    print("FEDERATED LEARNING CLIENT HETEROGENEITY (Top 100 ICD Codes)")
    print("="*80)
    print(f"{'Chapter':<25} {'Top-K Ratio %':<15} {'Samples':<12}")
    print("-" * 60)
    
    for chapter, percentage, sample_count in sorted_data:
        chapter_name = chapter.replace('_', ' ').title()
        print(f"{chapter_name:<25} {percentage:<15.2f} {sample_count:<12,}")
    
    # ============================================================================
    # WORD-COMPATIBLE FORMAT
    # ============================================================================
    print(f"\n{'='*80}")
    print("WORD-FORMATTED TABLE (Copy and paste into Word)")
    print(f"{'='*80}")
    print("Chapter   Top-K Ratio %   Samples")
    for chapter, percentage, sample_count in sorted_data:
        chapter_name = chapter.replace('_', ' ').title()
        print(f"{chapter_name}   {percentage:.2f}%   {sample_count:,}")

if __name__ == "__main__":
    plot_federated_learning_heterogeneity() 
