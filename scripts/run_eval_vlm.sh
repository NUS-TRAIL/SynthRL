#!/bin/bash

cd evaluation

# Function to print usage
print_usage() {
    echo "Usage: $0 MODEL_PATH [STEP_PATTERN]"
    echo "Examples:"
      echo "  $0 QWEN25-3B-1                      # Process all steps"
      echo "  $0 QWEN2.5-VERIFY-K12-V7/qwen2_5_vl_7b_k12-evolved-v7-qwen-nokl   # Process nested path"
      echo "  $0 QWEN25-3B-1 '96:'                # Only steps >= 96"
      echo "  $0 QWEN25-3B-1 ':96'                # Only steps <= 96"
      echo "  $0 QWEN25-3B-1 '50|96'              # Only steps 50 and 96"
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
BASE_CHECKPOINTS_DIR="PROJECT_DIR/VerifiableSynthesis/EasyR1/checkpoints"

# System prompt
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

# Process command line arguments
MODEL_PATH="$1"
STEP_PATTERN="$2"

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

# Filter steps if pattern is provided
if [ -n "$STEP_PATTERN" ]; then
    FILTERED_STEPS=()
    
    for step_dir in "${ALL_STEPS[@]}"; do
        step_num=$(basename "$step_dir" | sed 's/global_step_//')
        
        if [[ "$STEP_PATTERN" == *: ]]; then
            # Pattern like "96:" - steps >= specified number
            min_step="${STEP_PATTERN%:}"
            if [ "$step_num" -ge "$min_step" ]; then
                FILTERED_STEPS+=("$step_dir")
            fi
        elif [[ "$STEP_PATTERN" == :* ]]; then
            # Pattern like ":96" - steps <= specified number
            max_step="${STEP_PATTERN#:}"
            if [ "$step_num" -le "$max_step" ]; then
                FILTERED_STEPS+=("$step_dir")
            fi
        elif [[ "$STEP_PATTERN" == *\|* ]]; then
            # Pattern like "50|96" - specific steps
            IFS='|' read -ra SPECIFIC_STEPS <<< "$STEP_PATTERN"
            for specific in "${SPECIFIC_STEPS[@]}"; do
                if [ "$step_num" -eq "$specific" ]; then
                    FILTERED_STEPS+=("$step_dir")
                    break
                fi
            done
        else
            # Single specific step
            if [ "$step_num" -eq "$STEP_PATTERN" ]; then
                FILTERED_STEPS+=("$step_dir")
            fi
        fi
    done
    
    SELECTED_STEPS=("${FILTERED_STEPS[@]}")
else
    # No pattern - use all steps
    SELECTED_STEPS=("${ALL_STEPS[@]}")
fi

# Sort steps numerically
IFS=$'\n' SORTED_STEPS=($(for step in "${SELECTED_STEPS[@]}"; do echo "$step"; done | sort -V))
unset IFS

# Print selected steps
echo "Selected steps for evaluation:"
for step in "${SORTED_STEPS[@]}"; do
    echo "  - $(basename "$step")"
done

# Process each step
for step_dir in "${SORTED_STEPS[@]}"; do
    step_name=$(basename "$step_dir")
    
    # Check if actor/huggingface exists
    if [ -d "${step_dir}/actor/huggingface" ]; then
        model_path="${step_dir}/actor/huggingface"
    else
        model_path="${step_dir}"
    fi
    
    # Create results directory path
    RESULTS_DIR="./logs_vlm/${MODEL_PATH}/${step_name}"
    
    echo "Evaluating: $step_name"
    echo "Model path: $model_path"
    echo "Results dir: $RESULTS_DIR"
    
    # Create the results directory
    mkdir -p "$RESULTS_DIR"
    
    # Run evaluation
    python main.py \
      --model "$model_path" \
      --output-dir "$RESULTS_DIR" \
      --datasets "dynamath" \
      --tsv-data-path "PROJECT_DIR/LMUData" \
      --tensor-parallel-size 2 \
      --system-prompt="$SYSTEM_PROMPT" \
      --temperature 0.0 \
      --eval-threads 64
    
    echo "Completed evaluation for $step_name"
    echo "-----------------------------------"
done

echo "All evaluations completed successfully"