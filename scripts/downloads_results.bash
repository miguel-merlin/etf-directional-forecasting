#!/bin/bash
# Example usage:
#   ./scripts/download_results.bash my-bucket-name results-2026-02-10 ./downloads
set -euo pipefail

# -------- DEFAULTS --------
DEFAULT_BUCKET_NAME="etf-experiment-results-stevens"
DEFAULT_REGION="us-east-1"
DEFAULT_S3_PREFIX=""          # e.g. "results-2026-02-09" (leave empty to list and pick latest)
DEFAULT_DEST_DIR="./downloads" # base folder to download into
# --------------------------

# -------- PARAMS ----------
BUCKET_NAME="${1:-$DEFAULT_BUCKET_NAME}"
S3_PREFIX="${2:-$DEFAULT_S3_PREFIX}"     # optional
DEST_BASE_DIR="${3:-$DEFAULT_DEST_DIR}"  # optional
REGION="$DEFAULT_REGION"
# --------------------------

echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"
echo "S3 prefix: ${S3_PREFIX:-<auto>}"
echo "Destination base dir: $DEST_BASE_DIR"

mkdir -p "$DEST_BASE_DIR"

# If no prefix given, pick the latest prefix under the bucket (by most recent object timestamp)
if [ -z "$S3_PREFIX" ]; then
  echo "No S3 prefix provided. Discovering latest upload prefix..."

  # Get the most recent object key, then take its top-level prefix (everything before first '/')
  LATEST_KEY="$(
    aws s3api list-objects-v2 \
      --bucket "$BUCKET_NAME" \
      --query 'reverse(sort_by(Contents,&LastModified))[0].Key' \
      --output text
  )"

  if [ -z "$LATEST_KEY" ] || [ "$LATEST_KEY" = "None" ]; then
    echo "Error: No objects found in bucket: $BUCKET_NAME"
    exit 1
  fi

  S3_PREFIX="${LATEST_KEY%%/*}"
fi

S3_URI="s3://$BUCKET_NAME/$S3_PREFIX"
DEST_DIR="$DEST_BASE_DIR/$S3_PREFIX"

echo "Download source: $S3_URI"
echo "Download target: $DEST_DIR"

mkdir -p "$DEST_DIR"

# Download directory
aws s3 sync "$S3_URI" "$DEST_DIR"

echo "Download complete: $DEST_DIR"