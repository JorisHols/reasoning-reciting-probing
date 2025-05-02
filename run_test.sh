#!/bin/bash


# Create output directory
OUTPUT_DIR="./outputs/kk_test"
mkdir -p $OUTPUT_DIR

# Run the main.py script with the K&K experiment
python code/main.py \
    --experiment kk \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --max_new_tokens 512 \
    --checkpoint_every 10

echo "Test complete. Results saved to $OUTPUT_DIR" 