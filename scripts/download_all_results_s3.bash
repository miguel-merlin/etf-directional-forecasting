#!/bin/bash
# Example usage:
#   ./scripts/download_all_results_s3.bash my-bucket-name ./downloads
set -euo pipefail

# -------- DEFAULTS --------
DEFAULT_BUCKET_NAME="etf-experiment-results-stevens"
DEFAULT_REGION="us-east-1"
DEFAULT_DEST_DIR="./downloads" # base folder to download into
# --------------------------

# -------- PARAMS ----------
BUCKET_NAME="${1:-$DEFAULT_BUCKET_NAME}"
DEST_DIR="${2:-$DEFAULT_DEST_DIR}"
REGION="$DEFAULT_REGION"
# --------------------------

echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"
echo "Download target: $DEST_DIR"

mkdir -p "$DEST_DIR"

S3_URI="s3://$BUCKET_NAME"
echo "Download source: $S3_URI"

# Download all results from bucket
aws s3 sync "$S3_URI" "$DEST_DIR"

echo "Download complete: $DEST_DIR"
