# ==============================================================================
# File Name: Dockerfile
# Description: Minimal Docker environment for the Isaac Lab Grasping Project.
#              It extends the official Isaac Lab image and sets up the workspace.
# ==============================================================================

# 1. Use the official Isaac Lab image as the base
# This ensures compatibility with Nvidia Omniverse and Isaac Sim dependencies.
FROM nvcr.io/nvidia/isaac-lab:2.3.0

# 2. Set Metadata
# Adding metadata helps identify the purpose of the image in a registry or local listing.
LABEL description="Minimal Isaac Lab Environment for MultiGripper Grasping Project"
LABEL maintainer="MultiGripper Project Contributors"

# 3. Set the Working Directory
# This ensures that when a user enters the container, they land directly in
# the project folder, eliminating the need to manually 'cd' into directories.
WORKDIR /workspace/MultiGripper_Grasping_isaaclab

# 4. Set the Default Command
# Launch a bash shell by default to allow interactive exploration.
CMD ["bash"]