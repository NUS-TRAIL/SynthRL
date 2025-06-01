#!/bin/bash

PROJECT_DIR="/path/to/your/project"  # CHANGE THIS TO YOUR PROJECT DIRECTORY

cd ${PROJECT_DIR}/evaluation

# Function to print usage
print_usage() {
    echo "Usage: $0 MODEL_PATH"
    echo "Example:"
    echo "  $0 QWEN25-3B-1   # Process all steps in parallel"
}

# Show usage if help requested or no arguments
if [[ "$1" == "-h" || "$1" == "--help" || $# -eq 0 ]]; then
    print_usage
    exit 0
fi

# Set up environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

# Base checkpoints directory
BASE_CHECKPOINTS_DIR="${PROJECT_DIR}/checkpoints"

# System prompt
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

# Process command line arguments
MODEL_PATH="$1"

# Full path to the checkpoint directory
CHECKPOINT_DIR="${BASE_CHECKPOINTS_DIR}/${MODEL_PATH}"
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Check if the checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    exit 1
fi

# Get all global_step directories
echo "Looking for checkpoint steps in: $CHECKPOINT_DIR"
ALL_STEPS=($(ls -d "$CHECKPOINT_DIR"/global_step_* 2>/dev/null))

if [ ${#ALL_STEPS[@]} -eq 0 ]; then
    echo "No checkpoint steps found in '$CHECKPOINT_DIR'!"
    exit 1
fi

# Sort steps numerically
IFS=$'\n' SORTED_STEPS=($(for step in "${ALL_STEPS[@]}"; do echo "$step"; done | sort -V))
unset IFS

# Print all steps found
echo "Found ${#SORTED_STEPS[@]} steps for evaluation:"
for step in "${SORTED_STEPS[@]}"; do
    echo "  - $(basename "$step")"
done

# Function to evaluate a checkpoint with specified CUDA devices
evaluate_checkpoint() {
    local step_dir=$1
    local cuda_devices=$2

    local step_name=$(basename "$step_dir")

    # Check if actor/huggingface exists
    if [ -d "${step_dir}/actor/huggingface" ]; then
        model_path="${step_dir}/actor/huggingface"
    else
        model_path="${step_dir}"
    fi

    # Create results directory path
    RESULTS_DIR="./logs_vlm/${MODEL_PATH}/${step_name}"

    echo "Evaluating: $step_name (CUDA: $cuda_devices)"
    echo "Model path: $model_path"
    echo "Results dir: $RESULTS_DIR"

    # Create the results directory
    mkdir -p "$RESULTS_DIR"

    # Run evaluation with CUDA devices specified at command line
    CUDA_VISIBLE_DEVICES=$cuda_devices python main.py \
      --model "$model_path" \
      --output-dir "$RESULTS_DIR" \
      --datasets "wemath,mathvista,mathverse,dynamath,geo3k,mathvision" \
      --tsv-data-path "/shared_data/zijian/LMUData" \
      --tensor-parallel-size 2 \
      --system-prompt="$SYSTEM_PROMPT" \
      --temperature 0.0 \
      --eval-threads 64

    echo "Completed evaluation for $step_name"
    echo "-----------------------------------"
}

# Divide steps into groups of maximum 4
TOTAL_STEPS=${#SORTED_STEPS[@]}
MAX_PER_GROUP=4
GROUP_COUNT=$(( (TOTAL_STEPS + MAX_PER_GROUP - 1) / MAX_PER_GROUP ))

echo "Dividing $TOTAL_STEPS steps into $GROUP_COUNT groups for parallel evaluation"

# Process each group
for ((group=0; group<GROUP_COUNT; group++)); do
    echo "Starting evaluation group $((group+1)) of $GROUP_COUNT"

    # Calculate the start and end indices for this group
    START_IDX=$((group * MAX_PER_GROUP))
    END_IDX=$(( (START_IDX + MAX_PER_GROUP - 1) < (TOTAL_STEPS - 1) ? (START_IDX + MAX_PER_GROUP - 1) : (TOTAL_STEPS - 1) ))

    # Start background processes for each checkpoint in this group
    PROCESSES=()
    for ((i=START_IDX; i<=END_IDX; i++)); do
        step_dir="${SORTED_STEPS[i]}"

        # Calculate CUDA devices (0,1 for first item, 2,3 for second, etc.)
        position=$((i - START_IDX))
        cuda_start=$((position * 2))
        cuda_devices="$cuda_start,$((cuda_start + 1))"

        # Run in background
        evaluate_checkpoint "$step_dir" "$cuda_devices" &
        PROCESSES+=($!)

        echo "Started background process for $(basename "$step_dir") with PID ${PROCESSES[-1]}"
    done

    # Wait for all processes in this group to finish
    echo "Waiting for group $((group+1)) to complete..."
    for pid in "${PROCESSES[@]}"; do
        wait $pid
        echo "Process $pid completed"
    done

    echo "Group $((group+1)) completed"
    echo "============================"
done

echo "All evaluations completed successfully"