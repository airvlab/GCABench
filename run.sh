#!/bin/bash

# ==============================================================================
# Script Name: run.sh
# Description: Automates the deployment of the Isaac Lab environment.
#              It validates local assets, builds the Docker image, and runs the
#              container with necessary volume mounts and GPU configurations.
# Usage:       ./run.sh [optional_command]
#              Example: ./run.sh
#              Example: ./run.sh python record_demos_new.py
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---

# Docker image tag
IMAGE_NAME="isaac-lab-gripper:v1"

# Get the absolute path of the project root directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the host cache directory for Isaac Sim
# This ensures that shader caches and kit data are persisted on the host
ISAAC_CACHE_DIR="$HOME/docker/isaac-sim"

# --- Step 1: Asset Validation ---

# Define the expected path for 3D assets
ASSET_DIR="${PROJECT_DIR}/models/objects_MG"

# Check if the directory exists and if it contains files (is not empty)
if [ ! -d "$ASSET_DIR" ] || [ -z "$(ls -A "$ASSET_DIR" 2>/dev/null)" ]; then
    echo "====================================================================="
    echo "❌ CRITICAL ERROR: Missing Required 3D Assets"
    echo "---------------------------------------------------------------------"
    echo "The simulation environment requires external object models to function."
    echo "These files were not found in the expected directory."
    echo ""
    echo "Please complete the following steps to resolve this issue:"
    echo ""
    echo "1. Download the asset package 'objects_usd_fixed_density.zip' from:"
    echo "   https://utdallas.app.box.com/v/multi-gripper-grasp-data"
    echo ""
    echo "2. Extract the contents directly into the following path:"
    echo "   ${ASSET_DIR}"
    echo ""
    echo "   (Ensure the .usd files are directly inside this folder)"
    echo "====================================================================="
    exit 1
fi

echo "[INFO] Asset validation successful. Assets found in: ${ASSET_DIR}"

# --- Step 2: Build the Image ---

echo "[INFO] Building Docker image: ${IMAGE_NAME}..."

# Build the image using the Dockerfile in the current directory
# This ensures local dependencies (like nano/vim) are installed
if ! docker build -t "${IMAGE_NAME}" "${PROJECT_DIR}"; then
    echo "[ERROR] Docker build failed. Please check the Dockerfile."
    exit 1
fi

# --- Step 3: Run the Container ---

echo "[INFO] Starting container..."
echo "[INFO] Project Directory mapped: ${PROJECT_DIR}"

# Explanation of flags:
# --rm:          Remove the container automatically after it exits
# --gpus all:    Pass all GPUs to the container
# --network=host:Share host networking (required for Omniverse streaming)
# -v:            Volume mounts for code and Isaac Sim caches

docker run \
    --name "isaac-lab-gripper-container" \
    --entrypoint bash \
    -it \
    --rm \
    --gpus all \
    --network=host \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=${DISPLAY}" \
    -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v "${ISAAC_CACHE_DIR}/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "${ISAAC_CACHE_DIR}/cache/ov:/root/.cache/ov:rw" \
    -v "${ISAAC_CACHE_DIR}/cache/pip:/root/.cache/pip:rw" \
    -v "${ISAAC_CACHE_DIR}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "${ISAAC_CACHE_DIR}/cache/computecache:/root/.nv/ComputeCache:rw" \
    -v "${ISAAC_CACHE_DIR}/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "${ISAAC_CACHE_DIR}/data:/root/.local/share/ov/data:rw" \
    -v "${ISAAC_CACHE_DIR}/documents:/root/Documents:rw" \
    -v "${PROJECT_DIR}:/workspace/MultiGripper_Grasping_isaaclab" \
    -w "/workspace/MultiGripper_Grasping_isaaclab" \
    "${IMAGE_NAME}" \
    "$@"