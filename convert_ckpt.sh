#!/bin/bash

# Get command line arguments
INFO=$1
TAG=$2

# Set default values if not provided
INFO=${INFO:-}
TAG=${TAG:-}

# Path to your checkpoints directory
CHECKPOINTS_DIR="./checkpoints/${INFO}"

# Iterate through all checkpoint directories
for checkpoint in "$CHECKPOINTS_DIR"/qwen2_5_vl_7b_${TAG}/global_step*/actor; do
    if [ -d "$checkpoint" ]; then
        echo "Converting checkpoint: $checkpoint"
        python3 scripts/model_merger.py --local_dir "$checkpoint"
        echo "Conversion completed for: $checkpoint"

        # Delete all .pt files in the actor directory
        echo "Deleting .pt files in: $checkpoint"
        rm -f "$checkpoint"/*.pt
        echo "Deletion completed for: $checkpoint"
        echo "----------------------------------------"
    fi
done

echo "All checkpoints have been converted and .pt files deleted!"