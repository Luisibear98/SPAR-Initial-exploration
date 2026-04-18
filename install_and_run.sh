#!/bin/bash

# Exit immediately if a command exits with a non-zero status
apt-get update && apt-get install -y tmux
apt-get update && apt-get install -y nvtop

set -e

echo "--- Phase 1: Environment Setup ---"

# 1. Remove old env and reinstall uv
rm -rf unsloth_env
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is in the current shell path (in case it's a fresh install)
export PATH="$HOME/.cargo/bin:$PATH"

# 2. Create and activate the environment
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate

# 3. Install dependencies
uv pip install unsloth --torch-backend=auto
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
fi

echo "--- Phase 2: Training Loop ---"

# 4. Run the training for configs 1 through 4
for i in {1..4}; do
    echo "Starting training with config_$i..."
    # Using 'python' here works because the venv is active
    python document_and_instructions.py --config "config_$i"
    echo "Finished training with config_$i"
    echo "-----------------------------------"
done

echo "--- Phase 3: Evaluation ---"

# 5. Run preference evaluation
echo "Starting preference evaluation on trained models..."
python eval/preference_misalignment/preference_eval.py

echo "Finished preference evaluation. All tasks complete!"

