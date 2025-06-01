#!/bin/bash
set -x

PROJECT_DIR="/path/to/your/project"  # CHANGE THIS TO YOUR PROJECT DIRECTORY

cd ${PROJECT_DIR}

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export WANDB_MODE='offline'

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

# Get INFO, TAG, SAVE_FREQ, and TOTAL_EPISODES from command line arguments
INFO=$1
TAG=$2
SAVE_FREQ=$3
TOTAL_EPISODES=$4

# Check if required parameters are provided
if [ -z "$TAG" ]; then
    echo "Error: INFO and TAG parameters are required."
    echo "Usage: $0 INFO TAG [SAVE_FREQ] [TOTAL_EPISODES]"
    exit 1
fi

# Set default value for SAVE_FREQ if not provided
if [ -z "$SAVE_FREQ" ]; then
    SAVE_FREQ=16  # Default value
fi

# Set default value for TOTAL_EPISODES if not provided
if [ -z "$TOTAL_EPISODES" ]; then
    TOTAL_EPISODES=16  # Default value
fi

# Set default data
DATA="Jakumetsu/A-MMK12-8K"

# Construct the checkpoint directory path
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/${INFO}/qwen2_5_vl_7b_${TAG}"

# Check if checkpoint directory exists and look for latest step
RESUME_PATH="null"
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_STEP_FILE="${CHECKPOINT_DIR}/latest_global_step.txt"
    
    if [ -f "$LATEST_STEP_FILE" ]; then
        STEP=$(cat "$LATEST_STEP_FILE" | tr -d '[:space:]')
        RESUME_PATH="${CHECKPOINT_DIR}/global_step_${STEP}"
        echo "Resuming from: $RESUME_PATH"
    else
        echo "No latest_global_step.txt found. Starting training from scratch."
    fi
else
    echo "Checkpoint directory does not exist. Starting training from scratch."
fi

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=./scripts/grpo_example.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.train_files=${DATA}@train \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_${TAG} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=./checkpoints/${INFO}/qwen2_5_vl_7b_${TAG} \
    trainer.project_name=${INFO} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.total_episodes=${TOTAL_EPISODES} \
    trainer.load_checkpoint_path=${RESUME_PATH}

# Convert checkpoints to Hugging Face format
bash ./convert_ckpt.sh ${INFO} ${TAG} 

echo "Done"