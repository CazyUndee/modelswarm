#!/bin/bash
# Initial setup script for ModelSwarm development.
# Usage: bash scripts/setup.sh

set -e

echo "========================================"
echo "ModelSwarm Development Setup"
echo "========================================"

# Check prerequisites
echo ""
echo "[1/4] Checking prerequisites..."
python --version || { echo "Python 3.10+ required"; exit 1; }
pip --version || { echo "pip required"; exit 1; }

# Install Python package
echo ""
echo "[2/4] Installing Python package..."
pip install -e ".[dev]"

# Install worker dependencies
echo ""
echo "[3/4] Installing worker dependencies..."
if command -v npm &> /dev/null; then
    cd worker && npm install && cd ..
else
    echo "  npm not found — skipping worker setup (install Node.js to deploy worker)"
fi

# Run tests
echo ""
echo "[4/4] Running tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Deploy the worker: cd worker && wrangler deploy"
echo "  2. Register: modelswarm register --name 'MyAgent' --model 'claude-opus-5' --role research"
echo "  3. Join: modelswarm join s6e8"
echo "  4. Start: modelswarm start"
