#!/bin/bash

echo "Starting GPRMax Assistant training process..."
echo "-----------------------------------"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed!"
    exit 1
fi

# Check if training data directory exists
if [ ! -d "training_data" ]; then
    echo "Creating training_data directory..."
    mkdir -p training_data
    echo "Please add your training files to the training_data directory:"
    echo "- PDF files for QA mode"
    echo "- .in files for Simulation mode"
    exit 1
fi

# Check if there are any training files (PDF for QA mode and .in for Simulation mode)
pdf_count=$(find training_data -name "*.pdf" | wc -l)
in_count=$(find training_data -name "*.in" | wc -l)

if [ "$pdf_count" -eq 0 ] && [ "$in_count" -eq 0 ]; then
    echo "Error: No training files found in training_data directory!"
    echo "Please add:"
    echo "- PDF files for QA mode"
    echo "- .in files for Simulation mode"
    exit 1
fi

echo "Found $pdf_count PDF files for QA mode"
echo "Found $in_count .in files for Simulation mode"

# Run the training script
echo "Running training script..."
python train.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed!"
    exit 1
fi 