#!/bin/bash

cd evaluation

# Function to print usage
print_usage() {
    echo "Usage: $0 [INFO] [STEP_PATTERN]"
    echo "Examples:"
      echo "  $0                          # Run default Qwen/Qwen2.5-VL-7B-Instruct model"
      echo "  $0 QWEN25-3B-1             # Run models from PROJECT_DIR/EasyR1/checkpoints/QWEN25-3B-1"
      echo "  $0 QWEN25-3B-1 '5000|10000' # Run only steps 5000 and 10000 from the specified directory"
      echo "  $0 QWEN25-3B-1 '5000:'     # Run step 5000 and all steps after it from the specified directory"
}

# Show usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Set up environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

# Base checkpoints directory
BASE_CHECKPOINTS_DIR="./logs_vlm"

# System prompt
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

# Set up models based on arguments
if [ $# -eq 0 ]; then
    # No arguments - use default Qwen/Qwen2.5-VL-7B-Instruct model
    echo "Running evaluation on default model: Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Set up for the default model
    DEFAULT_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
    SAVE_NAME="Qwen2.5-VL-7B-Instruct"
    RESULTS_DIR="./logs_vlm/${SAVE_NAME}"
    
    # Create the results directory
    mkdir -p $RESULTS_DIR
    
    # Run evaluation on default model
    python main.py \
      --model $DEFAULT_MODEL_PATH \
      --output-dir $RESULTS_DIR \
      --datasets "olympiadbench" \
      --tsv-data-path "~/LMUData" \
      --tensor-parallel-size 2 \
      --system-prompt="$SYSTEM_PROMPT" \
      --temperature 0.0 \
      --eval-threads 32
    
    echo "Evaluation completed for default model"
    exit 0
else
    INFO="$1"
    STEP_PATTERN="$2"
    
    # Set checkpoints directory with INFO appended
    CHECKPOINTS_DIR="${BASE_CHECKPOINTS_DIR}/${INFO}"
    
    # Check if the checkpoints directory exists
    if [ ! -d "$CHECKPOINTS_DIR" ]; then
        echo "Error: Checkpoints directory '$CHECKPOINTS_DIR' not found!"
        exit 1
    fi
    
    # Find checkpoint step directories
    if [ -n "$STEP_PATTERN" ]; then
        # Find directories matching the pattern (should match global_step_XXXX)
        selected_steps=($(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -path "*global_step_*" | grep -E "global_step_(${STEP_PATTERN})" | sort -V))
    else
        # List all step directories in the checkpoints directory
        selected_steps=($(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -path "*global_step_*" | sort -V))
    fi
    
    # Check if any steps were found
    if [ ${#selected_steps[@]} -eq 0 ]; then
        echo "No checkpoint steps found in '$CHECKPOINTS_DIR'!"
        exit 1
    fi
fi

# Print selected configuration
echo "Selected checkpoint steps:"
for step in "${selected_steps[@]}"; do
    echo "  - $(basename "$step")"
done

# Iterate through each checkpoint step
for step_dir in "${selected_steps[@]}"; do
    # The actual model path includes /actor/huggingface
    model_path="${step_dir}"
    
    # Check if the huggingface path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path '$model_path' not found, skipping..."
        continue
    fi
    
    step_name=$(basename "$step_dir")
    SAVE_NAME="${INFO}/${step_name}"
    RESULTS_DIR="./logs_vlm/${SAVE_NAME}"
    
    echo "Evaluating checkpoint: ${step_name}"
    echo "Model Path: $model_path"
    echo "Results Directory: $RESULTS_DIR"
    
    # Create the results directory
    mkdir -p $RESULTS_DIR
    
    python main.py \
      --model $model_path \
      --output-dir $RESULTS_DIR \
      --datasets "all" \
      --tsv-data-path "~/LMUData" \
      --tensor-parallel-size 2 \
      --system-prompt="$SYSTEM_PROMPT" \
      --temperature 0.0 \
      --eval-threads 32
    
    echo "Evaluation completed for checkpoint: ${step_name}"
    echo "-----------------------------------------"
done

echo "Done"