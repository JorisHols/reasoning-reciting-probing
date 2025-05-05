"""Main script to probe LLaMA model with different datasets."""

import argparse
import logging
import os

from k_and_k import KKProbe


def setup_basic_logging():
    """Configure basic logging for all modules."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get module loggers
    loggers = {
        "main": logging.getLogger("main"),
        "probe_llama": logging.getLogger("probe_llama"),
        "k_and_k": logging.getLogger("k_and_k")
    }
    
    return loggers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Probe LLaMA model with different datasets"
    )
    
    # Dataset type
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="kk",
        choices=["chess", "kk"],
        help="Type of dataset to use for probing"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input dataset"
    )
    # Output parameters
    parser.add_argument(
        "--output_path",
        type=str,
        # No default, making this a required argument
        help="Directory to save the output"
    )
    
    # Processing parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate"
    )
    
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save checkpoint after this many batches"
    )

    # Whether to generate response text or just capture activations
    parser.add_argument(
        "--no_generate",
        action="store_true",
        help="Don't generate responses, only capture activations"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the script."""
    # Set up logging
    loggers = setup_basic_logging()
    logger = loggers["main"]

    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Get prompts based on dataset type
    logger.info(f"Setting up experiment: {args.experiment}")
    if args.experiment == "kk":
        experiment = KKProbe(input_path=args.input_path, output_path=args.output_path)
    elif args.experiment == "chess":
        # You can implement this function similarly to get_k_and_k_prompts
        logger.warning("Chess dataset not implemented, using test prompts")
        return
    else:  # Default to test prompts
        raise ValueError(f"Dataset {args.experiment} not implemented")

    # Set up the probe with command line parameters
    logger.info(
        f"Setting up probe with batch size {args.batch_size}, "
        f"max new tokens {args.max_new_tokens}, "
        f"generate response {not args.no_generate}, "
        f"checkpoint every {args.checkpoint_every} batches"
    )
    
    experiment.setup_probe(
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        generate_response=not args.no_generate,
        checkpoint_every=args.checkpoint_every
    )

    logger.info("Starting experiment")

    experiment.run_experiment()
    
    logger.info(f"Experiment complete. Results saved to {args.output_path}")
    
    return


if __name__ == "__main__":
    main()