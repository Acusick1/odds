#!/bin/bash
# Build and push Lambda container image to ECR
#
# Usage:
#   ./build_and_push.sh <environment> <aws-region> <aws-account-id>
#
# Example:
#   ./build_and_push.sh dev eu-west-1 123456789012
#
# Requirements:
#   - Docker installed and running
#   - AWS credentials configured
#   - ECR repository already created

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-eu-west-1}
AWS_ACCOUNT_ID=${3}

# Validate required arguments
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}Error: AWS account ID is required${NC}"
    echo "Usage: $0 <environment> <aws-region> <aws-account-id>"
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|prod)$ ]]; then
    echo -e "${RED}Error: Environment must be 'dev' or 'prod'${NC}"
    exit 1
fi

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ECR configuration
ECR_REPOSITORY="odds-scheduler"
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_FULL_URL="${ECR_URL}/${ECR_REPOSITORY}"

# Get git SHA for tagging
GIT_SHA=$(git rev-parse --short HEAD)
IMAGE_TAG="${ENVIRONMENT}-${GIT_SHA}"
LATEST_TAG="${ENVIRONMENT}-latest"

echo -e "${GREEN}Building Lambda container image...${NC}"
echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"
echo "ECR Repository: $ECR_FULL_URL"
echo "Image tags: $IMAGE_TAG, $LATEST_TAG"
echo ""

# Login to ECR
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_URL"

# Build image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    --provenance=false \
    --platform linux/amd64 \
    -f deployment/aws/Dockerfile.lambda \
    -t "${ECR_FULL_URL}:${IMAGE_TAG}" \
    -t "${ECR_FULL_URL}:${LATEST_TAG}" \
    .

# Push both tags
echo -e "${YELLOW}Pushing image to ECR...${NC}"
docker push "${ECR_FULL_URL}:${IMAGE_TAG}"
docker push "${ECR_FULL_URL}:${LATEST_TAG}"

# Output image URI for Terraform
IMAGE_URI="${ECR_FULL_URL}:${IMAGE_TAG}"
echo ""
echo -e "${GREEN}✓ Image built and pushed successfully!${NC}"
echo ""
echo "Image URI: ${IMAGE_URI}"
echo "Latest tag: ${ECR_FULL_URL}:${LATEST_TAG}"
echo ""
echo "To deploy with Terraform:"
echo "  cd deployment/aws/terraform"
echo "  terraform apply -var=\"image_tag=${IMAGE_TAG}\""
echo ""

# Set GitHub Actions output if running in CI
if [ -n "$GITHUB_OUTPUT" ]; then
    echo "image_uri=${IMAGE_URI}" >> "$GITHUB_OUTPUT"
    echo "image_tag=${IMAGE_TAG}" >> "$GITHUB_OUTPUT"
fi
