#!/usr/bin/env python3
"""
Script to combine multiple chunked Hugging Face datasets into a single dataset.

Usage:
  python combine_chunks.py --input_dir <input_dir> [--output_dir <output_dir>]

Arguments:
  --input_dir       Path to the directory containing chunked datasets (e.g., outputs/kk/perturbed_leaf/temp_folder)
  --output_dir      Optional: Directory to save the combined dataset (default: <input_dir>/combined_dataset)

Examples:
  python combine_chunks.py --input_dir ./outputs/kk/perturbed_leaf/temp_20250507_125314
  python combine_chunks.py --input_dir /path/to/job/dir --output_dir ./results/kk/perturbed_leaf

Notes:
  - The script looks for datasets in the exact pattern: chunk_*/data/chunk_*
  - All paths can be absolute or relative to the current working directory

Requirements:
  - datasets (pip install datasets)
"""

import argparse
import os
import glob
import re
from datasets import load_from_disk, concatenate_datasets


def find_dataset_dirs(base_dir):
    """
    Find all dataset directories matching the pattern chunk_*/data/
    """
    # Convert to absolute path for clarity in messages
    base_dir = os.path.abspath(base_dir)
    print(f"Searching in absolute path: {base_dir}")
    
    # Exact pattern: chunk_*/data/chunk_*
    dataset_dirs = []
    
    # First, find all chunk_* directories
    chunk_dirs = glob.glob(os.path.join(base_dir, "chunk_*"))
    if not chunk_dirs:
        print(f"No chunk_* directories found in {base_dir}")
        return []
    
    print(f"Found {len(chunk_dirs)} chunk directories")
    
    # For each chunk directory, look for data directory
    for chunk_dir in chunk_dirs:
        data_dir = os.path.join(chunk_dir, "data")
        if not os.path.isdir(data_dir):
            print(f"Data directory not found in {chunk_dir}")
            continue
            
        # Check if the data directory itself is a valid dataset directory
        if (os.path.exists(os.path.join(data_dir, "dataset_info.json")) and 
            any(f.endswith(".arrow") for f in os.listdir(data_dir) 
                if os.path.isfile(os.path.join(data_dir, f)))):
            dataset_dirs.append(data_dir)
            print(f"Found valid dataset: {data_dir}")
    
    # Sort dataset directories by chunk number to ensure consistent order
    def extract_chunk_number(dir_path):
        # Extract the chunk number from the parent directory name
        match = re.search(r'chunk_(\d+)/data', dir_path)
        if match:
            return int(match.group(1))
        return 0
    
    sorted_dirs = sorted(dataset_dirs, key=extract_chunk_number)
    
    if sorted_dirs:
        print(f"Found {len(sorted_dirs)} valid datasets to combine")
    else:
        print("No valid datasets found")
        
    return sorted_dirs


def combine_datasets(input_dir, output_dir=None):
    """
    Find and combine all chunk datasets in the input directory.
    Save the combined dataset to the output directory.
    """
    if not input_dir:
        raise ValueError("Input directory must be specified")
    
    # Normalize paths
    input_dir = os.path.normpath(input_dir)
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "combined_dataset")
    else:
        output_dir = os.path.normpath(output_dir)
    
    # Find all dataset directories
    print(f"Searching for datasets in {input_dir}...")
    dataset_dirs = find_dataset_dirs(input_dir)
    
    if not dataset_dirs:
        print(f"No datasets found in {input_dir}")
        print("Make sure the directory contains the structure: chunk_*/data/chunk_*")
        return
    
    # Load all datasets
    datasets = []
    for i, dataset_dir in enumerate(dataset_dirs):
        print(f"Loading dataset {i+1}/{len(dataset_dirs)} from: {os.path.basename(os.path.dirname(os.path.dirname(dataset_dir)))}")
        try:
            ds = load_from_disk(dataset_dir)
            datasets.append(ds)
            print(f"  Loaded dataset with {len(ds)} examples")
        except Exception as e:
            print(f"  Error loading dataset: {e}")
    
    if not datasets:
        print("No datasets could be loaded")
        return
    
    # Combine datasets
    print("Combining datasets...")
    combined = concatenate_datasets(datasets)
    print(f"Combined dataset has {len(combined)} examples")
    
    # Save combined dataset
    print(f"Saving combined dataset to {output_dir}")
    # Create the output directory directly
    os.makedirs(output_dir, exist_ok=True)
    combined.save_to_disk(output_dir)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Combine chunked Hugging Face datasets")
    parser.add_argument("--input_dir", required=True, help="Directory containing the chunked datasets")
    parser.add_argument("--output_dir", help="Directory to save the combined dataset")
    
    args = parser.parse_args()
    
    combine_datasets(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main() 