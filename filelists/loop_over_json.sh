#!/bin/bash

json_file="data_2016APV_local_nano.json"

mkdir -p lumis_data_2016APV_local_nano

# Use jq to parse the JSON file
datasets=$(jq -r 'keys[]' "$json_file")

for dataset in $datasets; do
    # Create the directory if it doesn't exist
    mkdir -p "lumis_data_2016APV_local_nano/$dataset"

    # Get all files for this dataset
    files=$(jq -r --arg ds "$dataset" '.[$ds][]' "$json_file")

    i=0
    while read -r nanoaod_file; do
        output_file="lumis_data_2016APV_local_nano/${dataset}/lumi_${i}.json"
        echo "Running: python nano_to_brilcalc.py \"$nanoaod_file\" \"$output_file\""
        python nano_to_brilcalc.py "$nanoaod_file" "$output_file"
        ((i++))
    done <<< "$files"
done
