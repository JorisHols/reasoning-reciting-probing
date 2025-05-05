#!/usr/bin/env python
"""
This script runs a short test of the K&K experiment with only 20 data points
to verify that the full pipeline works and checkpointing is functional.
"""

import os
import logging
import argparse
from k_and_k import KKProbe

# Set up logging to match the main script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("short_test")

class ShortTestProbe(KKProbe):
    """Modified KKProbe that limits the data to 20 items for quick testing"""
    
    def _load_jsonl_file(self, file_path, dataset_name):
        """Override to limit the number of items loaded"""
        items = super()._load_jsonl_file(file_path, dataset_name)
        
        # Limit to 20 items maximum per dataset
        if len(items) > 20:
            logger.info(f"Limiting {dataset_name} from {len(items)} to 20 items for testing")
            items = items[:20]
        
        return items

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a short test of the K&K experiment"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./inputs/k_and_k/3ppl",
        help="Path to the input dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/test_run",
        help="Directory to save the output"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,  # Reduced for faster testing
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,  # Set to 1 to ensure checkpointing is tested
        help="Save checkpoint after this many batches"
    )
    
    return parser.parse_args()

def main():
    """Run a short test of the K&K experiment."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set up the probe with the short test class
    logger.info(f"Setting up short test with data from {args.input_path}")
    experiment = ShortTestProbe(
        input_path=args.input_path,
        output_path=args.output_path
    )
    
    # Set up the probe with parameters tuned for quick testing
    logger.info(
        f"Setting up probe with batch_size={args.batch_size}, "
        f"max_new_tokens={args.max_new_tokens}, "
        f"checkpoint_every={args.checkpoint_every}"
    )
    experiment.setup_probe(
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        checkpoint_every=args.checkpoint_every
    )
    
    # Run the experiment
    logger.info("Starting short test experiment")
    experiment.run_experiment()
    
    logger.info(f"Short test complete! Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
