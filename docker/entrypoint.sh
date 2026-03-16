#!/bin/bash
# Entrypoint for the NeuronCite Docker container.
# Manages two long-running processes: Ollama (LLM inference server) and
# NeuronCite (web UI + API server). Both run as the non-root 'neuroncite'
# user. The script seeds runtime dependencies from the image cache into
# the persistent data volume on first start, then launches both services
# with proper signal forwarding for graceful shutdown.
#
# This entrypoint is GPU-agnostic: the same script runs for NVIDIA (CUDA),
# AMD (ROCm), and CPU-only variants. GPU selection is handled entirely by
# the base image and ONNX Runtime variant installed during docker build.

set -euo pipefail

DATA_DIR="/data/Documents/NeuronCite"

# ---------------------------------------------------------------------------
# First-start volume seeding
# ---------------------------------------------------------------------------
# PDFium is pre-downloaded during the Docker image build and stored at
# /opt/neuroncite/cache/pdfium/. The application checks for the library at
# $HOME/Documents/NeuronCite/runtime/pdfium/libpdfium.so (see deps.rs:146).
# Since the volume mount overlays the image filesystem, the entrypoint copies
# the library into the volume on first start.

PDFIUM_DEST="$DATA_DIR/runtime/pdfium/libpdfium.so"
if [ ! -f "$PDFIUM_DEST" ]; then
    echo "[entrypoint] Seeding PDFium library into data volume..."
    mkdir -p "$DATA_DIR/runtime/pdfium"
    cp /opt/neuroncite/cache/pdfium/libpdfium.so "$PDFIUM_DEST"
fi

# Ensure all data directories exist for the application to write into.
mkdir -p "$DATA_DIR/models" "$DATA_DIR/indexes" "$DATA_DIR/runtime"

# ---------------------------------------------------------------------------
# Start Ollama in background
# ---------------------------------------------------------------------------
# Ollama serves the LLM API on localhost:11434. NeuronCite's citation
# verification agent connects to this address by default (no URL change
# needed in the Web UI).

echo "[entrypoint] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to accept connections (max 30 seconds).
# The /api/tags endpoint returns a list of installed models and is the
# lightest available readiness probe.
OLLAMA_READY=0
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        OLLAMA_READY=1
        echo "[entrypoint] Ollama ready (waited ${i}s)"
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" -eq 0 ]; then
    echo "[entrypoint] WARNING: Ollama did not become ready within 30s, continuing anyway"
fi

# ---------------------------------------------------------------------------
# Start NeuronCite
# ---------------------------------------------------------------------------
# The 'web' subcommand starts the Axum HTTP server with the embedded SolidJS
# frontend, SSE endpoints, and all web-specific handlers. Binding to 0.0.0.0
# allows access from outside the container via the mapped port.
# The 'opener' crate will fail to open a browser (expected in a container),
# but the server runs regardless.

echo "[entrypoint] Starting NeuronCite web server on port 3030..."
neuroncite web --bind 0.0.0.0 --port 3030 &
NC_PID=$!

# ---------------------------------------------------------------------------
# Signal handling and process supervision
# ---------------------------------------------------------------------------
# Forward SIGTERM and SIGINT to both child processes for graceful shutdown.
# Docker sends SIGTERM on 'docker stop', and the trap ensures both Ollama
# and NeuronCite receive it.

cleanup() {
    echo "[entrypoint] Shutting down..."
    kill "$NC_PID" "$OLLAMA_PID" 2>/dev/null || true
    wait "$NC_PID" "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT

# Wait for either process to exit. If one crashes, stop the other and
# propagate the exit code so Docker marks the container as failed.
wait -n "$OLLAMA_PID" "$NC_PID"
EXIT_CODE=$?
echo "[entrypoint] Process exited with code $EXIT_CODE, stopping remaining services..."
cleanup
exit "$EXIT_CODE"
