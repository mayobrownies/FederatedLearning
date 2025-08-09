from typing import Dict
import pandas as pd


# Creates heterogeneous partitions - one chapter per client
def create_heterogeneous_partitions(data: pd.DataFrame, min_partition_size: int = 1000, top_k_codes: int = 50) -> Dict[str, pd.DataFrame]:
    print("Creating heterogeneous partitions (one chapter per client)...")
    
    icd_counts = data['icd_code'].value_counts()
    top_codes = set(icd_counts.head(top_k_codes).index.tolist())
    data['has_top_k_code'] = data['icd_code'].isin(top_codes)
    chapter_groups = data.groupby('diagnosis_chapter')
    
    partitions = {}
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size:
            partitions[f"{chapter}"] = chapter_data
            print(f"Chapter client '{chapter}': {len(chapter_data)} samples, {top_k_ratio:.2%} top-K")
        else:
            print(f"Skipping chapter '{chapter}': only {len(chapter_data)} samples (< {min_partition_size})")
    
    print(f"Created {len(partitions)} heterogeneous partitions (one per chapter)")
    return partitions

# Creates balanced partitions with mixed chapters and good top-K distribution
def create_balanced_partitions(data: pd.DataFrame, min_partition_size: int = 1000, top_k_codes: int = 50) -> Dict[str, pd.DataFrame]:
    print("Creating balanced partitions...")
    
    icd_counts = data['icd_code'].value_counts()
    top_codes = set(icd_counts.head(top_k_codes).index.tolist())
    data['has_top_k_code'] = data['icd_code'].isin(top_codes)
    chapter_groups = data.groupby('diagnosis_chapter')
    
    partitions = {}
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size and top_k_ratio > 0.15:
            partitions[f"{chapter}"] = chapter_data
            print(f"Single chapter partition '{chapter}': {len(chapter_data)} samples, {top_k_ratio:.2%} top-K")
    
    remaining_chapters = []
    for chapter, chapter_data in chapter_groups:
        top_k_count = chapter_data['has_top_k_code'].sum()
        top_k_ratio = top_k_count / len(chapter_data)
        
        if len(chapter_data) >= min_partition_size and top_k_ratio <= 0.15:
            remaining_chapters.append((chapter, chapter_data))
        elif len(chapter_data) < min_partition_size:
            remaining_chapters.append((chapter, chapter_data))
    
    if remaining_chapters:
        combined_data = pd.concat([data for _, data in remaining_chapters], ignore_index=True)
        
        top_k_data = combined_data[combined_data['has_top_k_code']]
        other_data = combined_data[~combined_data['has_top_k_code']]
        
        min_top_k_ratio = 0.20
        partition_size = max(min_partition_size, len(combined_data) // 3)
        
        for i in range(3):
            if len(top_k_data) + len(other_data) < partition_size:
                break
                
            min_top_k_needed = int(partition_size * min_top_k_ratio)
            if len(top_k_data) < min_top_k_needed:
                print(f"Skipping mixed partition {i+1}: insufficient top-K samples ({len(top_k_data)} < {min_top_k_needed})")
                break
                
            target_top_k = min(len(top_k_data), max(min_top_k_needed, partition_size // 3))
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
            
            if len(partition_top_k) + len(partition_other) >= min_partition_size:
                top_k_ratio = len(partition_top_k) / (len(partition_top_k) + len(partition_other))
                
                if top_k_ratio >= min_top_k_ratio:
                    partition_data = pd.concat([partition_top_k, partition_other], ignore_index=True)
                    partition_name = f"mixed_{i+1}"
                    partitions[partition_name] = partition_data
                    print(f"Mixed partition '{partition_name}': {len(partition_data)} samples, {top_k_ratio:.2%} top-K")
                else:
                    print(f"Skipping mixed partition {i+1}: top-K ratio too low ({top_k_ratio:.2%} < {min_top_k_ratio:.2%})")
                    top_k_data = pd.concat([top_k_data, partition_top_k], ignore_index=True)
                    other_data = pd.concat([other_data, partition_other], ignore_index=True)
                    break
    
    print(f"Created {len(partitions)} balanced partitions")
    return partitions
