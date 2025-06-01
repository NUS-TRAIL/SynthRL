#!/bin/bash

# Define variables for frequently changed parameters
DATASET_NAME="K12-Freeform-8K"
INFO="VERIFIABLE-V10-EVOLVE-QWEN"
NUM_SAMPLES=9999 # allow maximum samples
MAX_ATTEMPTS=8
JUDGE_THRESHOLD=8
MODEL_NAME="gemini-2.5-flash-preview-04-17"
SOLVER_NAME="Pro/Qwen/Qwen2.5-VL-7B-Instruct"
BATCH_SIZE=1024
NUM_WORKERS=12  # Reduced per process since we'll run 4 processes
SEED=42
ROLLOUT=16
LEVEL=12

# Extract dataset basename for finding the rollout output
DATASET_BASENAME=$(basename $DATASET_NAME)
ROLLOUT_OUTPUT="${DATASET_BASENAME}_rollout${ROLLOUT}_qwen7b_v2"
SAMPLED_DIR="./sampled_data"
ROLLOUT_OUTPUT_PATH="${SAMPLED_DIR}/${ROLLOUT_OUTPUT}"

# Determine output directory using INFO for naming
OUTPUT_DIR="./synthrl_data/${INFO}_${DATASET_BASENAME}_attempts${MAX_ATTEMPTS}_thres${JUDGE_THRESHOLD}"
TEMP_DIR="${SAMPLED_DIR}/${INFO}_${ROLLOUT_OUTPUT}_temp"

# Set the path to the scripts
ROLLOUT_SCRIPT="./synthrl_framework/original_question_rollout.py"
EVOLVE_SCRIPT="./synthrl_framework/evolve_instruction_verifiable.py"
SPLIT_SCRIPT="./synthrl_framework/split_dataset.py"
MERGE_SCRIPT="./synthrl_framework/merge_dataset.py"

echo "Step 1: Running original question rollout..."
# Run the rollout script first to verify original questions
python $ROLLOUT_SCRIPT \
    --dataset_path $DATASET_NAME \
    --model_name $SOLVER_NAME \
    --rollout $ROLLOUT \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --sample_limit $NUM_SAMPLES \
    --seed $SEED

echo "Original question rollout completed!"
echo "Results saved to ${ROLLOUT_OUTPUT_PATH}"

# Create temp directory if doesn't exist
mkdir -p $TEMP_DIR

echo "Step 2: Splitting dataset into chunks..."
python $SPLIT_SCRIPT \
    --dataset_path $ROLLOUT_OUTPUT_PATH \
    --output_dir $TEMP_DIR \
    --num_chunks 4

echo "Step 3: Running evolution with verification in parallel..."
# Run 4 processes in parallel with different API keys
GOOGLE_API_KEYS=(
    "YOUR_API_KEY_1"
    "YOUR_API_KEY_2"
    "YOUR_API_KEY_3"
    "YOUR_API_KEY_4"
)

for i in {1..4}; do
    # Updated to include INFO in the path names
    CHUNK_PATH="${INFO}_${ROLLOUT_OUTPUT}_temp/${ROLLOUT_OUTPUT}_chunk${i}"
    OUTPUT_CHUNK_PATH="${TEMP_DIR}/${INFO}_output_chunk${i}"
    
    echo "Starting process for chunk ${i} with GOOGLE API key placeholder ${i}..."
    (
        GOOGLE_API_KEY=${GOOGLE_API_KEYS[$i-1]} python $EVOLVE_SCRIPT \
            --dataset_path $CHUNK_PATH \
            --output_dir $OUTPUT_CHUNK_PATH \
            --model_name $MODEL_NAME \
            --solver_name $SOLVER_NAME \
            --judge_threshold $JUDGE_THRESHOLD \
            --max_attempts $MAX_ATTEMPTS \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --sample_limit $NUM_SAMPLES \
            --rollout $ROLLOUT \
            --seed $SEED \
            --difficulty_level $LEVEL \
            --images_in_list
    ) &
    
    # Add a small delay to prevent any race conditions
    sleep 2
done

# Wait for all background processes to complete
wait

echo "Step 4: Merging results..."
python $MERGE_SCRIPT \
    --chunks_dir "${TEMP_DIR}" \
    --output_path $OUTPUT_DIR \
    --info $INFO

# Print a message when the script finishes
echo "Answer-preserving question evolution completed!"
echo "Results saved to $OUTPUT_DIR"