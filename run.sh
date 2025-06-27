#!/bin/bash

echo "Starting Churn Prediction Pipeline..."
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Virtual Environment Not Found. Creating one...."
    python -m venv $VENV_DIR

    if [[ $? -ne 0 ]]; then
        echo "Failed to create environment"
        exit 1
    fi
fi

echo "Activating Virtual Environment......"
source $VENV_DIR/Scripts/activate

if [[ -f "requirements.txt" ]]; then
    echo "Installing dependencies from requirements.txt......."
    uv add -r requirements.txt
else
    echo "Warning!!!! requirements.txt not found. Skipping installation"

fi

echo "Running the script....."
python main.py

echo "Pipeline execution complete...."