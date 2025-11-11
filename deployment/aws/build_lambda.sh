#!/bin/bash
# Build Lambda deployment package using Docker
#
# This script builds a Lambda-compatible deployment package containing:
# - odds-core: Foundation layer (models, database, config)
# - odds-lambda: Lambda runtime (jobs, storage, scheduling, data fetcher)
# - External dependencies from PyPI
# - Lambda handler entry point
#
# The build uses Dockerfile.lambda with a multi-stage build process that:
# 1. Installs packages using UV (proper workspace dependency resolution)
# 2. Creates a Lambda-compatible directory structure
# 3. Packages everything into lambda.zip
#
# Requirements:
# - Docker installed and running
# - Run from repository root or deployment/aws directory

set -e

echo "Building Lambda deployment package using Docker..."

# Navigate to repository root (script can be run from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Clean previous build
rm -f deployment/aws/lambda.zip deployment/aws/lambda_deployment.zip

# Build Lambda package using Dockerfile
# The --output flag extracts lambda.zip from the final scratch stage
echo "Running Docker build (this may take a minute)..."
docker build \
  -f deployment/aws/Dockerfile.lambda \
  --output type=local,dest=deployment/aws \
  . \
  2>&1 | grep -v "^#" || true  # Filter build output noise

# Rename to standard name
if [ -f deployment/aws/lambda.zip ]; then
    mv deployment/aws/lambda.zip deployment/aws/lambda_deployment.zip
fi

echo ""
echo "✓ Lambda deployment package created: deployment/aws/lambda_deployment.zip"
echo "  Size: $(du -h deployment/aws/lambda_deployment.zip | cut -f1)"
