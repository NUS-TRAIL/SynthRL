#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
from datasets import Dataset, load_from_disk, concatenate_datasets, Features
import argparse

def merge_datasets(chunks_dir, output_path, info=None):
    """Merge multiple dataset chunks into one"""
    print(f"Looking for chunks in {chunks_dir}")
    
    # Find all chunks in the directory
    if info:
        chunk_pattern = os.path.join(chunks_dir, f"{info}_output_chunk*")
    else:
        chunk_pattern = os.path.join(chunks_dir, "output_chunk*")
        
    chunk_paths = sorted(glob.glob(chunk_pattern))
    
    if not chunk_paths:
        print(f"Error: No chunks found matching pattern {chunk_pattern}")
        # Try alternate pattern as fallback
        alternate_pattern = os.path.join(chunks_dir, "output_chunk*")
        chunk_paths = sorted(glob.glob(alternate_pattern))
        if chunk_paths:
            print(f"Found chunks using alternate pattern: {[os.path.basename(p) for p in chunk_paths]}")
        else:
            print(f"No chunks found with alternate pattern either. Exiting.")
            sys.exit(1)
    
    print(f"Found {len(chunk_paths)} chunks: {[os.path.basename(p) for p in chunk_paths]}")
    
    # Rest of the function remains the same
    # Load each chunk
    datasets = []
    all_features = {}
    
    # First pass: load datasets and collect all feature types
    for chunk_path in chunk_paths:
        try:
            chunk_dataset = load_from_disk(chunk_path)
            print(f"Loaded {os.path.basename(chunk_path)} with {len(chunk_dataset)} samples")
            
            # Store the dataset
            datasets.append(chunk_dataset)
            
            # Collect feature types
            for key, feature in chunk_dataset.features.items():
                if key not in all_features:
                    all_features[key] = feature
                # If there's a conflict, prefer string type over null type
                elif "string" in str(feature) and "null" in str(all_features[key]):
                    all_features[key] = feature
                
        except Exception as e:
            print(f"Error loading chunk {chunk_path}: {e}")
    
    if not datasets:
        print("No datasets could be loaded. Exiting.")
        sys.exit(1)
    
    # Second pass: normalize features across all datasets
    normalized_datasets = []
    for idx, dataset in enumerate(datasets):
        # Create a new dictionary with the normalized features
        new_features = {}
        for key in dataset.features:
            if key in all_features:
                new_features[key] = all_features[key]
            else:
                new_features[key] = dataset.features[key]
        
        # Create a new dataset with the normalized features
        # We need to cast the dataset to make features compatible
        try:
            normalized_dataset = dataset.cast(Features(new_features))
            normalized_datasets.append(normalized_dataset)
            print(f"Normalized features for chunk {idx+1}")
        except Exception as e:
            print(f"Warning: Could not normalize chunk {idx+1}: {e}")
            # If normalization fails, we'll still try to use the original dataset
            normalized_datasets.append(dataset)
    
    # Concatenate all chunks with normalized features
    try:
        merged_dataset = concatenate_datasets(normalized_datasets)
        print(f"Merged dataset has {len(merged_dataset)} samples")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the merged dataset
        merged_dataset.save_to_disk(output_path)
        print(f"Saved merged dataset to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error during concatenation: {e}")
        
        # Fallback: If concatenation fails, try a manual merge approach
        print("Trying alternative merge approach...")
        
        # Create an empty list to hold all examples
        all_examples = []
        
        # Collect all examples from all datasets
        for idx, dataset in enumerate(normalized_datasets):
            for example in dataset:
                all_examples.append(example)
        
        # Create a new dataset from all examples
        if all_examples:
            merged_dataset = Dataset.from_dict({k: [d.get(k) for d in all_examples] 
                                            for k in all_examples[0].keys()})
            
            print(f"Merged dataset has {len(merged_dataset)} samples")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the merged dataset
            merged_dataset.save_to_disk(output_path)
            print(f"Saved merged dataset to {output_path}")
            
            return output_path
        else:
            print("Failed to merge datasets through any method.")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge dataset chunks")
    parser.add_argument("--chunks_dir", type=str, required=True, help="Directory containing chunks")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged dataset")
    parser.add_argument("--info", type=str, help="INFO prefix used in chunk naming", default=None)
    
    args = parser.parse_args()
    merge_datasets(args.chunks_dir, args.output_path, args.info)