# NeuronCite Docker

Production-ready Docker deployment for NeuronCite with GPU acceleration support.

## Pre-built Images (ghcr.io)

Images are published to GitHub Container Registry on every tagged release.

| Variant | Tag | Architecture | GPU |
|---------|-----|--------------|-----|
| NVIDIA CUDA | `ghcr.io/<owner>/neuroncite:latest` | x86_64 | CUDA 12.4 + cuDNN |
| AMD ROCm | `ghcr.io/<owner>/neuroncite:latest-rocm` | x86_64 | ROCm 6.4 |
| CPU | `ghcr.io/<owner>/neuroncite:latest-cpu` | x86_64 | none |
| CPU ARM64 | `ghcr.io/<owner>/neuroncite:latest-cpu-arm64` | ARM64 | none |

Replace `<owner>` with the GitHub repository owner (e.g. `neuroncite`).

Versioned tags follow the pattern `:<version>-<variant>` (e.g. `:1.2.3-nvidia`).
The NVIDIA variant additionally receives the bare `:<version>` and `:latest` tags.

## Quick Start

### NVIDIA GPU

```bash
docker run --gpus all -p 3030:3030 \
  -v neuroncite-data:/data/Documents/NeuronCite \
  ghcr.io/<owner>/neuroncite:latest
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### AMD ROCm

```bash
docker run --device=/dev/kfd --device=/dev/dri \
  -p 3030:3030 \
  -v neuroncite-data:/data/Documents/NeuronCite \
  ghcr.io/<owner>/neuroncite:latest-rocm
```

### CPU only

```bash
docker run -p 3030:3030 \
  -v neuroncite-data:/data/Documents/NeuronCite \
  ghcr.io/<owner>/neuroncite:latest-cpu
```

Access the web UI at **http://localhost:3030** after the container starts.

## Docker Compose

Four profiles are provided for convenience:

```bash
# NVIDIA GPU
docker compose -f docker/docker-compose.yml --profile nvidia up -d

# AMD ROCm
docker compose -f docker/docker-compose.yml --profile rocm up -d

# CPU (x86_64)
docker compose -f docker/docker-compose.yml --profile cpu up -d

# CPU (ARM64)
docker compose -f docker/docker-compose.yml --profile cpu-arm64 up -d
```

Stop: `docker compose -f docker/docker-compose.yml --profile <profile> down`

## Building Locally

```bash
# NVIDIA (default)
docker build -f docker/Dockerfile -t neuroncite:nvidia .

# AMD ROCm
docker build -f docker/Dockerfile \
  --build-arg GPU=rocm \
  --build-arg RUNTIME_BASE=rocm/dev-ubuntu-22.04:6.4 \
  -t neuroncite:rocm .

# CPU only
docker build -f docker/Dockerfile \
  --build-arg GPU=cpu \
  --build-arg RUNTIME_BASE=ubuntu:22.04 \
  -t neuroncite:cpu .

# ARM64 CPU (cross-compilation via BuildKit)
docker buildx build --platform linux/arm64 -f docker/Dockerfile \
  --build-arg GPU=cpu \
  --build-arg RUNTIME_BASE=ubuntu:22.04 \
  -t neuroncite:cpu-arm64 .
```

## Environment Variables

These variables are pre-configured in the Dockerfile. Override them at runtime
if needed via `docker run -e VAR=value` or in your compose file.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOME` | `/data` | Application base directory. NeuronCite stores all data under `$HOME/Documents/NeuronCite/`. |
| `ORT_DYLIB_PATH` | `/usr/local/lib/libonnxruntime.so` | Path to the ONNX Runtime shared library loaded via dlopen by the `ort` crate. |
| `LD_LIBRARY_PATH` | `/usr/local/lib` | Linker search path for GPU execution provider libraries (CUDA or ROCm). |
| `OLLAMA_HOST` | `0.0.0.0:11434` | Bind address for the Ollama LLM server running inside the container. |
| `NVIDIA_VISIBLE_DEVICES` | `all` | NVIDIA Container Toolkit: which GPUs to expose. Set to a specific GPU index to limit access. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | NVIDIA Container Toolkit: required driver capabilities for ONNX Runtime inference. |

## CLI Arguments

The entrypoint starts `neuroncite web --bind 0.0.0.0 --port 3030`. Override via
the Docker command:

```bash
docker run --gpus all -p 8080:8080 \
  -v neuroncite-data:/data/Documents/NeuronCite \
  ghcr.io/<owner>/neuroncite:latest \
  /entrypoint.sh
```

To change only the port, modify the entrypoint or use a custom command override
in your compose file.

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `3030` | TCP port for the HTTP server. |
| `--bind` | `0.0.0.0` (in Docker) | Bind address. The Dockerfile overrides the application default of `127.0.0.1` to allow container-external access. |

## Persistent Data Volume

All application data is stored at `/data/Documents/NeuronCite/` inside the
container. Mount a Docker volume or host directory to persist data across
container restarts:

```
neuroncite-data:/data/Documents/NeuronCite
```

Contents of the data directory:

| Path | Description |
|------|-------------|
| `models/` | Embedding models downloaded from HuggingFace on first use. |
| `indexes/` | HNSW vector indexes and BM25 full-text indexes for indexed PDFs. |
| `runtime/pdfium/` | PDFium shared library, seeded from the image cache on first start. |

Ollama stores its models at `$HOME/.ollama/` (`/data/.ollama/`), which is also
inside the volume mount.

## Container Architecture

The container runs two processes managed by the entrypoint script:

1. **Ollama** (port 11434) -- local LLM inference server for citation verification
2. **NeuronCite** (port 3030) -- web server with embedded SolidJS frontend and REST API

`tini` runs as PID 1 for proper signal forwarding and zombie process reaping.
On `docker stop`, SIGTERM is forwarded to both processes for graceful shutdown.

## Health Check

The built-in health check probes `GET /api/v1/health` every 30 seconds with a
60-second start period. Monitor via:

```bash
docker inspect --format='{{.State.Health.Status}}' neuroncite
```

## Platform Support Matrix

| Architecture | NVIDIA CUDA | AMD ROCm | CPU |
|-------------|-------------|----------|-----|
| x86_64 (amd64) | yes | yes | yes |
| ARM64 (aarch64) | no | no | yes |

GPU acceleration on ARM64 is not available because Microsoft does not publish
ONNX Runtime GPU builds for ARM64 Linux.
