#!/bin/bash
# run_palm.sh — Register a single palm scan folder using palm-optimized ICP parameters.
#
# Usage:
#   ./run_palm.sh <scan_folder>
#   ./run_palm.sh dataset/rahmat_20260401_200613
#
# Output:
#   result/[label]/[timestamp]/output.ply

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_folder>"
    echo "Example: $0 dataset/rahmat_20260401_200613"
    exit 1
fi

SCAN_FOLDER="$1"
SCAN_NAME=$(basename "$SCAN_FOLDER")

# Parse label dan timestamp dari nama folder: [label]_YYYYMMDD_HHMMSS
TIMESTAMP=$(echo "$SCAN_NAME" | grep -oE '[0-9]{8}_[0-9]{6}$')
LABEL=$(echo "$SCAN_NAME" | sed "s/_${TIMESTAMP}$//")

if [ -z "$TIMESTAMP" ] || [ -z "$LABEL" ]; then
    echo "Error: nama folder harus berformat [label]_YYYYMMDD_HHMMSS"
    echo "Contoh: rahmat_20260401_200613"
    exit 1
fi

OUTPUT="result/$LABEL/$TIMESTAMP/output.ply"

if [ ! -d "$SCAN_FOLDER" ]; then
    echo "Error: folder '$SCAN_FOLDER' tidak ditemukan"
    exit 1
fi

mkdir -p "result/$LABEL/$TIMESTAMP"
echo "Processing palm scan: $SCAN_FOLDER"
echo "Label: $LABEL  |  Timestamp: $TIMESTAMP"

python3 run.py "$SCAN_FOLDER" \
    --method 0 \
    --output "$OUTPUT" \
    --min_depth 0.10 \
    --max_depth 0.50 \
    --max_point_dist 0.015 \
    --normal_radius 0.008 \
    --voxel_size 0.001 \
    --keep_nearest_cluster 1 \
    --foreground_depth_range 0.12 \
    --cluster_connectivity_eps 0.008 \
    --cluster_connectivity_min_points 20 \
    --outlier_nb_neighbors 30 \
    --outlier_std_ratio 1.5 \
    --viz 0

echo "Done: $OUTPUT"
