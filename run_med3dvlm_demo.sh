#!/usr/bin/env bash

# Script to run the Med3DVLM demo inference
# Place this script in the root of the Med3DVLM project directory

# 1. Activate your conda environment (if using conda):
#    conda activate Med3DVLM

# 2. Ensure PYTHONPATH includes the project root so that custom modules load properly:
export PYTHONPATH="$(pwd)":$PYTHONPATH

# 3. Set paths to your model and a sample image:
#    - MODEL_DIR should point to the unpacked Med3DVLM-Qwen-2.5-7B directory
#    - IMAGE_PATH points to one of the demo NIfTI files
MODEL_DIR="./Med3DVLM-Qwen-2.5-7B"
IMAGE_PATH="./data/demo/024421/Axial_bone_window.nii.gz"
QUESTION="Describe the findings of the medical image you see."

# 4. Adjust the path to the demo script entrypoint if needed:
#    If your folder is named 'scr', leave as-is. If 'src', update accordingly.
DEMO_SCRIPT="./scr/demo/demo.py"
# DEMO_SCRIPT="./src/demo/demo.py"

# 5. Run the demo:
python "$DEMO_SCRIPT" \
  --model_name_or_path "$MODEL_DIR" \
  --image_path "$IMAGE_PATH" \
  --question "$QUESTION"
