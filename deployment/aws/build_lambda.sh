#!/bin/bash
# Build Lambda deployment package
#
# This script creates a deployment package for AWS Lambda containing:
# - All application code
# - Python dependencies
# - Lambda handler
#
# Uses Docker to ensure compatibility with Lambda's Amazon Linux 2 runtime

set -e

echo "Building Lambda deployment package..."

# Clean previous build
rm -rf build/
rm -f lambda_deployment.zip

# Create build directory
mkdir -p build/

# Copy application code
echo "Copying application code..."
cp -r ../../core build/
cp -r ../../storage build/
cp -r ../../analytics build/
cp -r ../../alerts build/
cp -r ../../jobs build/
cp lambda_function.py build/

# Install dependencies using Docker with Lambda-compatible environment
echo "Installing Python dependencies (using Docker for Lambda compatibility)..."
docker run --rm \
  --entrypoint="" \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/../../requirements.txt:/requirements.txt:ro" \
  -v "$(pwd)/build:/build" \
  public.ecr.aws/lambda/python:3.11 \
  pip install -r /requirements.txt -t /build --no-cache-dir

# Create deployment package
echo "Creating deployment zip..."
cd build/
zip -r ../lambda_deployment.zip . -q
cd ..

# Cleanup
rm -rf build/

echo "✓ Lambda deployment package created: lambda_deployment.zip"
echo "  Size: $(du -h lambda_deployment.zip | cut -f1)"
