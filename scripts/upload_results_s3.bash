#!/bin/bash
set -euo pipefail

# -------- DEFAULTS --------
DEFAULT_BUCKET_NAME="etf-experiment-results-stevens"
DEFAULT_REGION="us-east-1"
DEFAULT_LOCAL_DIR="./results"
# --------------------------

# -------- PARAMS ----------
BUCKET_NAME="${1:-$DEFAULT_BUCKET_NAME}"
LOCAL_DIR="${2:-$DEFAULT_LOCAL_DIR}"
REGION="$DEFAULT_REGION"
# --------------------------

# -------- DATE PREFIX -----
DATE_SUFFIX="$(date +%Y-%m-%d)"
S3_PREFIX="$(basename "$LOCAL_DIR")-$DATE_SUFFIX"
S3_URI="s3://$BUCKET_NAME/$S3_PREFIX"
# --------------------------

echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"
echo "Local directory: $LOCAL_DIR"
echo "Upload target: $S3_URI"

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

# Upload directory with date suffix
aws s3 sync "$LOCAL_DIR" "$S3_URI"

echo "Upload complete: $S3_URI"