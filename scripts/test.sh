# Set up the variables manually
INPUT_PATH="./inputs/k_and_k/3ppl/clean/clean_people3_num5000.jsonl"
OUTPUT_PATH="./outputs/k_and_k_debug"
CHUNK_ID=1
CHUNK_SIZE=50

# Create output directory
mkdir -p "${OUTPUT_PATH}/chunk_${CHUNK_ID}"

# Run experiment for a single chunk
bash ./scripts/run_experiment.sh \
  --input_path "$INPUT_PATH" \
  --output_path "${OUTPUT_PATH}/chunk_${CHUNK_ID}" \
  --chunk_id $CHUNK_ID \
  --chunk_size $CHUNK_SIZE