#!/bin/bash

# Activate environment (AWS ONLY):
#source activate pytorch

# Directory where the folders are located
base_directory="/home/ubuntu/models/"
n=5  # Change this to your desired value of n

# Loop through each folder inside the base directory
for folder in "$base_directory"/*/; do
    # Check if it's a directory
    if [ -d "$folder" ]; then
        echo "Model: $folder"
        
        # Loop through each .onnx file in the current folder
        for file in "$folder"*.onnx; do
            for i in $(seq 0 $n)
            do
                echo "Iteration:" $i

                # Check if it's a file (not a directory) and has the .onnx extension
                if [ -f "$file" ]; then
                    echo "Processing file: $file"
                    python benchmark_replicate.py --model $file --iteration $i
                    echo "-------------------------------------------------------------------"
                    python benchmark_replicate_onnxruntime.py --model $file --iteration $i
                    echo "-------------------------------------------------------------------"
                    python benchmark_replicate_openvino.py --model $file --iteration $i
                    echo "-------------------------------------------------------------------"
                    python benchmark_baseline.py --model $file --iteration $i
                    echo "==================================================================="
                fi
            done
        done
    fi
done
