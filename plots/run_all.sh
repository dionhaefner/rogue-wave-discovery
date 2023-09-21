#!/bin/bash

# Generates all plots for the paper
# Usage: ./run_all.sh <data_path> <results_file>
# Example: ./run_all.sh ../data ../experiments/results.json

set -e

data_path=$1
results_file=$2

if [ -z "$data_path" ] || [ -z "$results_file" ]
then
    echo "Usage: ./run_all.sh <data_path> <results_file>"
    exit 1
fi

mkdir -p generated

echo "Generating pareto plot..."
python generate_pareto_plots.py "$results_file"

echo "Generating experiment table..."
python generate_table.py "$results_file"

echo "Evaluating final model..."
papermill \
    -p DATA_PATH "$data_path" \
    paper_plots.ipynb \
    generated/paper_plots.ipynb