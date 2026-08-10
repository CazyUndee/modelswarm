#!/bin/bash
# Run the full ModelSwarm test suite.
# Usage: bash scripts/run_tests.sh

set -e

echo "========================================"
echo "ModelSwarm Test Suite"
echo "========================================"

# Check Python
python --version

# Install test dependencies
pip install pytest pytest-mock -q

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/ -v --tb=short "$@"

echo ""
echo "========================================"
echo "All tests passed!"
echo "========================================"
