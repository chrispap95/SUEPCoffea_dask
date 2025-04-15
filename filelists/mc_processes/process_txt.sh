#!/bin/bash
total_size=0
while read -r p; do
  echo "Processing $p"
  # Add your processing commands here
  ((total_size += $(dasgoclient -query="dataset=$p summary" | jq -r '.[].file_size')))
done < "$1"
# Convert total size to GB
total_size=$(echo "scale=2; $total_size / 1024 / 1024 / 1024" | bc)
echo "Total size: $total_size GB"
