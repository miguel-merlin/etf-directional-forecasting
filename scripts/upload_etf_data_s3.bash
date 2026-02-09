#!/bin/bash
set -euo pipefail

# -------- DEFAULTS --------
DEFAULT_BUCKET_NAME="etf-data-stevens"
DEFAULT_REGION="us-east-1"
DEFAULT_LOCAL_DIR="./data"
# --------------------------

# -------- PARAMS ----------
BUCKET_NAME="${1:-$DEFAULT_BUCKET_NAME}"
LOCAL_DIR="${2:-$DEFAULT_LOCAL_DIR}"
REGION="$DEFAULT_REGION"
# --------------------------

echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"
echo "Local directory: $LOCAL_DIR"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "Error: Local directory does not exist: $LOCAL_DIR"
  exit 1
fi

# Create bucket if it does not exist
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
  echo "Bucket already exists"
else
  echo "Creating bucket"
  aws s3api create-bucket \
    --bucket "$BUCKET_NAME" \
    --region "$REGION"
fi

# Upload directory
aws s3 sync "$LOCAL_DIR" "s3://$BUCKET_NAME"

echo "Upload complete: s3://$BUCKET_NAME"