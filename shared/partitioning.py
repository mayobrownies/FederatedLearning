from typing import Dict
import pandas as pd


def create_heterogeneous_partitions(data: pd.DataFrame, min_partition_size: int = 1000, top_k_codes: int = 50) -> Dict[str, pd.DataFrame]:
    """
    Create heterogeneous partitions where each client gets exactly one medical chapter.
    This creates true data heterogeneity since each client specializes in one medical domain.
    """
    print("Creating heterogeneous partitions (one chapter per client)...")
    
    # First, identify which ICD codes are in the top K for reporting
    icd_counts = data['icd_code'].value_counts()
    top_codes = set(icd_counts.head(top_k_codes).index.tolist())
    
    # Add a flag for whether each sample has a top-K code
    data['has_top_k_code'] = data['icd_code'].isin(top_codes)
    
    # Group by chapter
    chapter_groups = data.groupby('diagnosis_chapter')
    
    partitions = {}
    
    # Create one partition per chapter (if it meets minimum size)
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size:
            # Use this chapter as a separate client
            partitions[f"{chapter}"] = chapter_data
            print(f"Chapter client '{chapter}': {len(chapter_data)} samples, {top_k_ratio:.2%} top-K")
        else:
            print(f"Skipping chapter '{chapter}': only {len(chapter_data)} samples (< {min_partition_size})")
    
    print(f"Created {len(partitions)} heterogeneous partitions (one per chapter)")
    return partitions

def create_balanced_partitions(data: pd.DataFrame, min_partition_size: int = 1000, top_k_codes: int = 50) -> Dict[str, pd.DataFrame]:
    """
    Create more balanced partitions by combining chapters and ensuring each client 
    has a good mix of top-K codes
    """
    print("Creating balanced partitions...")
    
    # First, identify which ICD codes are in the top K
    icd_counts = data['icd_code'].value_counts()
    top_codes = set(icd_counts.head(top_k_codes).index.tolist())
    
    # Add a flag for whether each sample has a top-K code
    data['has_top_k_code'] = data['icd_code'].isin(top_codes)
    
    # Group by chapter
    chapter_groups = data.groupby('diagnosis_chapter')
    
    partitions = {}
    
    # Strategy 1: For chapters with many top-K codes, use them as-is
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size and top_k_ratio > 0.15:
            # This chapter has a reasonable number of top-K codes
            partitions[f"{chapter}"] = chapter_data
            print(f"Single chapter partition '{chapter}': {len(chapter_data)} samples, {top_k_ratio:.2%} top-K")
    
    # Strategy 2: For chapters with few top-K codes, combine them
    remaining_chapters = []
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size and top_k_ratio <= 0.15:
            remaining_chapters.append((chapter, chapter_data))
        elif len(chapter_data) < min_partition_size:
            remaining_chapters.append((chapter, chapter_data))
    
    # Combine remaining chapters to create balanced partitions
    if remaining_chapters:
        combined_data = pd.concat([data for _, data in remaining_chapters], ignore_index=True)
        
        # Split combined data by top-K presence to create balanced partitions
        top_k_data = combined_data[combined_data['has_top_k_code']]
        other_data = combined_data[~combined_data['has_top_k_code']]
        
        # Create balanced partitions only if we have sufficient top-K data
        min_top_k_ratio = 0.20  # Minimum 20% top-K samples required
        partition_size = max(min_partition_size, len(combined_data) // 3)  # Aim for 3 combined partitions
        
        for i in range(3):
            if len(top_k_data) + len(other_data) < partition_size:
                break
                
            # Calculate how many top-K samples we need for minimum ratio
            min_top_k_needed = int(partition_size * min_top_k_ratio)
            
            # Only create partition if we have enough top-K samples
            if len(top_k_data) < min_top_k_needed:
                print(f"Skipping mixed partition {i+1}: insufficient top-K samples ({len(top_k_data)} < {min_top_k_needed})")
                break
                
            # Take a mix of top-K and other samples
            target_top_k = min(len(top_k_data), max(min_top_k_needed, partition_size // 3))  # At least 20% or 33% top-K
            target_other = min(len(other_data), partition_size - target_top_k)
            
            if target_top_k > 0:
                partition_top_k = top_k_data.sample(n=target_top_k, random_state=42+i)
                top_k_data = top_k_data.drop(partition_top_k.index)
            else:
                partition_top_k = pd.DataFrame()
            
            if target_other > 0:
                partition_other = other_data.sample(n=target_other, random_state=42+i)
                other_data = other_data.drop(partition_other.index)
            else:
                partition_other = pd.DataFrame()
            
            # Double-check that we meet the minimum requirements
            if len(partition_top_k) + len(partition_other) >= min_partition_size:
                top_k_ratio = len(partition_top_k) / (len(partition_top_k) + len(partition_other))
                
                if top_k_ratio >= min_top_k_ratio:
                    partition_data = pd.concat([partition_top_k, partition_other], ignore_index=True)
                    partition_name = f"mixed_{i+1}"
                    partitions[partition_name] = partition_data
                    print(f"Mixed partition '{partition_name}': {len(partition_data)} samples, {top_k_ratio:.2%} top-K")
                else:
                    print(f"Skipping mixed partition {i+1}: top-K ratio too low ({top_k_ratio:.2%} < {min_top_k_ratio:.2%})")
                    # Put the samples back for potential future use
                    top_k_data = pd.concat([top_k_data, partition_top_k], ignore_index=True)
                    other_data = pd.concat([other_data, partition_other], ignore_index=True)
                    break
    
    print(f"Created {len(partitions)} balanced partitions")
    return partitions
