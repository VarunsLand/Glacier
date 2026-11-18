#!/usr/bin/env bash
# GlacierVision One-Click Launcher for Linux / macOS

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo "               GLACIERVISION AI RESEARCH PLATFORM"
echo "====================================================================="

# Check for virtual environment
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Detect Python
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python not found. Please install Python 3.10+."
    exit 1
fi

echo "[INFO] Using Python: $($PY_CMD --version)"
echo "[INFO] Starting GlacierVision..."
echo ""

$PY_CMD run.py
