#!/bin/bash

# Script to activate the Poetry virtual environment for the reasoning-reciting-probing project

# Change to the project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$PROJECT_DIR/reasoning-reciting-probing"

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