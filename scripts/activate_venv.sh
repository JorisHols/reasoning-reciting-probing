#!/bin/bash

# Script to activate the Poetry virtual environment for the reasoning-reciting-probing project


# Change to the project directory
cd "${SLURM_SUBMIT_DIR}"
PROJECT_DIR=$(pwd)
cd "$PROJECT_DIR"

# Install the poetry dependencies
poetry install

# Get the path to the virtual environment
VENV_PATH=$(poetry env info --path)

if [ -z "$VENV_PATH" ]; then
    echo "Error: Poetry virtual environment not found."
    echo "Make sure you have installed the project dependencies with 'poetry install'."
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Show confirmation message
echo "Poetry virtual environment activated. You're now using Python from: $(which python)"
echo "To deactivate the environment when done, run: deactivate"

# Keep the terminal open by running the user's shell
exec "$SHELL" 