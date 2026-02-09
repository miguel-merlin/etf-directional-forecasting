#!/bin/bash

# -------- CONFIG --------
BUCKET_NAME="etf-data-stevens"
REGION="us-east-1"
LOCAL_DIR="../data"
# ------------------------

# Create bucket
aws s3api create-bucket \
  --bucket "$BUCKET_NAME" \
  --region "$REGION" \
  $( [ "$REGION" != "us-east-1" ] && echo "--create-bucket-configuration LocationConstraint=$REGION" )

# Upload directory
aws s3 sync "$LOCAL_DIR" "s3://$BUCKET_NAME"

echo "Upload complete: s3://$BUCKET_NAME"