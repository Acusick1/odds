#!/bin/bash
# Build Lambda deployment package for workspace architecture
#
# This script creates a deployment package for AWS Lambda containing:
# - Only odds-core and odds-lambda packages (analytics excluded)
# - Python dependencies for Lambda runtime
# - Lambda handler
#
# Uses Docker to ensure compatibility with Lambda's Amazon Linux 2 runtime

set -e

echo "Building Lambda deployment package (workspace architecture)..."

# Clean previous build
rm -rf build/
rm -f lambda_deployment.zip

# Create build directory
mkdir -p build/

# Copy ONLY Lambda-required packages (core + lambda, excluding analytics)
echo "Copying Lambda packages (odds-core + odds-lambda)..."
cp -r ../../packages/odds-core/odds_core build/
cp -r ../../packages/odds-lambda/odds_lambda build/

# Copy Lambda handler to root (entry point)
cp ../../packages/odds-lambda/odds_lambda/lambda_handler.py build/

# Create requirements.txt from package dependencies
echo "Generating Lambda requirements.txt..."
cat > build/requirements_lambda.txt << 'EOF'
# odds-core dependencies
sqlmodel>=0.0.14
asyncpg>=0.29.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
structlog>=23.3.0

# odds-lambda dependencies
aiohttp>=3.9.0
apscheduler>=3.10.4
tenacity>=8.2.3
boto3>=1.34.0
alembic>=1.13.0
EOF

# Install dependencies using Docker with Lambda-compatible environment
echo "Installing Python dependencies (using Docker for Lambda compatibility)..."
docker run --rm \
  --entrypoint="" \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/build/requirements_lambda.txt:/requirements.txt:ro" \
  -v "$(pwd)/build:/build" \
  public.ecr.aws/lambda/python:3.11 \
  pip install -r /requirements.txt -t /build --no-cache-dir

# Remove unnecessary files to reduce size
echo "Cleaning up unnecessary files..."
cd build/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
rm -f requirements_lambda.txt
cd ..

# Create deployment package
echo "Creating deployment zip..."
cd build/
zip -r ../lambda_deployment.zip . -q
cd ..

# Cleanup
rm -rf build/

echo "✓ Lambda deployment package created: lambda_deployment.zip"
echo "  Size: $(du -h lambda_deployment.zip | cut -f1)"
echo ""
echo "  Package contains:"
echo "    - odds_core/ (foundation layer)"
echo "    - odds_lambda/ (Lambda runtime)"
echo "    - lambda_handler.py (entry point)"
echo "    - Python dependencies"
echo ""
echo "  Analytics package EXCLUDED - size optimized for Lambda"
