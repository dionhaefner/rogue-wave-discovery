#!/bin/bash

# Runs all experiments back-to-back
# Usage: ./run_all.sh <data_path>
# Example: ./run_all.sh ../data/processed/

set -e

data_path=$1
if [ -z "$data_path" ]
then
    echo "Usage: ./run_all.sh <data_path>"
    exit 1
fi

num_experiments=$(jq '. | length' experiments.json)

mkdir -p executed/

for i in $(seq 0 $(($num_experiments - 1)))
do
    experiment_data=$(jq -r ".[$i] | @base64" experiments.json)
    papermill \
        -b "$experiment_data" \
        -p DATA_PATH "$data_path" \
        experiment_template.ipynb \
        executed/experiment_$((i+1)).ipynb
done
