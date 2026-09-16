#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "=================================================="
echo "HOSP-AI COMMAND — Render Production Build"
echo "=================================================="

# 1. Install frontend dependencies
echo "==> [1/3] Installing Node.js frontend dependencies..."
npm install --legacy-peer-deps

# 2. Build Vite frontend to dist/
echo "==> [2/3] Compiling React 19 / Vite frontend..."
npm run build

# 3. Install Python dependencies
echo "==> [3/3] Installing Python ML & API dependencies..."
if [ -f ".venv/bin/pip" ]; then
    .venv/bin/pip install --no-cache-dir -r requirements.txt
elif command -v uv &>/dev/null; then
    uv pip install -r requirements.txt
elif command -v pip3 &>/dev/null; then
    pip3 install --no-cache-dir -r requirements.txt
elif command -v pip &>/dev/null; then
    pip install --no-cache-dir -r requirements.txt
elif command -v python3 &>/dev/null; then
    python3 -m pip install --no-cache-dir -r requirements.txt
elif command -v python &>/dev/null; then
    python -m pip install --no-cache-dir -r requirements.txt
else
    echo "Warning: Python package manager not found."
fi

echo "=================================================="
echo "Build succeeded! Ready for deployment on Render."
echo "=================================================="

