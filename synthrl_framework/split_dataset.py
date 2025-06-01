#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from datasets import Dataset, load_from_disk
import argparse

def split_dataset(dataset_path, output_dir, num_chunks=4):
    """Split a dataset into multiple chunks"""
    print(f"Loading dataset from {dataset_path}")
    
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    total_samples = len(dataset)
    chunk_size = total_samples // num_chunks
    remainder = total_samples % num_chunks
    
    print(f"Total samples: {total_samples}")
    print(f"Splitting into {num_chunks} chunks")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the dataset and save each chunk
    start_idx = 0
    for i in range(num_chunks):
        # Add one extra sample to some chunks if we can't divide evenly
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        # Extract chunk
        chunk_data = dataset.select(range(start_idx, end_idx))
        
        # Create a meaningful filename
        base_name = os.path.basename(dataset_path)
        output_file = os.path.join(output_dir, f"{base_name}_chunk{i+1}")
        
        # Save the chunk
        chunk_data.save_to_disk(output_file)
        print(f"Saved chunk {i+1} with {len(chunk_data)} samples to {output_file}")
        
        start_idx = end_idx
    
    return [os.path.join(output_dir, f"{os.path.basename(dataset_path)}_chunk{i+1}") for i in range(num_chunks)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset into chunks")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save chunks")
    parser.add_argument("--num_chunks", type=int, default=4, help="Number of chunks to split into")
    
    args = parser.parse_args()
    split_dataset(args.dataset_path, args.output_dir, args.num_chunks)