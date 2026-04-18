# 1. Remove the old environment if it exists
echo "Cleaning up old environment..."
rm -rf unsloth_env

# 2. Install/Update uv
echo "Installing/Updating uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create a new virtual environment with Python 3.13
echo "Creating venv with Python 3.13..."
uv venv unsloth_env --python 3.13

# 4. Activate the environment
# Note: If running this as a script file, use 'source' to run it
source unsloth_env/bin/activate

# 5. Install unsloth and your requirements
echo "Installing packages..."
uv pip install unsloth --torch-backend=auto
uv pip install -r requirements.txt

echo "Setup complete. Environment 'unsloth_env' is ready."