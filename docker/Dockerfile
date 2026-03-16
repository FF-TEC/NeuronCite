# =============================================================================
# NeuronCite Production Docker Image
# =============================================================================
# Multi-stage build producing a single container with:
#   - NeuronCite binary (Rust + embedded SolidJS frontend)
#   - ONNX Runtime 1.23.2 with GPU acceleration (CUDA, ROCm, or CPU-only)
#   - Ollama LLM server for citation verification
#   - Tesseract OCR with English and German language packs
#   - PDFium for multi-column PDF extraction
#
# GPU variants (selected via --build-arg GPU=nvidia|rocm|cpu):
#
#   NVIDIA (default, x86_64 only):
#     docker build -f docker/Dockerfile -t neuroncite:nvidia .
#     docker run --gpus all -p 3030:3030 -v neuroncite-data:/data/Documents/NeuronCite neuroncite:nvidia
#
#   AMD ROCm (x86_64 only):
#     docker build -f docker/Dockerfile --build-arg GPU=rocm --build-arg RUNTIME_BASE=rocm/dev-ubuntu-22.04:6.4 -t neuroncite:rocm .
#     docker run --device=/dev/kfd --device=/dev/dri -p 3030:3030 -v neuroncite-data:/data/Documents/NeuronCite neuroncite:rocm
#
#   CPU only (x86_64 and ARM64):
#     docker build -f docker/Dockerfile --build-arg GPU=cpu --build-arg RUNTIME_BASE=ubuntu:22.04 -t neuroncite:cpu .
#     docker run -p 3030:3030 -v neuroncite-data:/data/Documents/NeuronCite neuroncite:cpu
#
#   ARM64 CPU (via BuildKit cross-compilation or native ARM64 host):
#     docker buildx build --platform linux/arm64 -f docker/Dockerfile --build-arg GPU=cpu --build-arg RUNTIME_BASE=ubuntu:22.04 -t neuroncite:cpu-arm64 .
#
# Access: http://localhost:3030
# =============================================================================

# --- Build arguments for GPU variant and architecture selection ---
# RUNTIME_BASE: base image for the runtime stage (determines available GPU drivers)
# GPU: selects the ONNX Runtime variant to download (nvidia=CUDA, rocm=ROCm, cpu=CPU-only)
# TARGETARCH: injected by Docker BuildKit -- "amd64" or "arm64"
ARG RUNTIME_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG GPU=nvidia


# ---------------------------------------------------------------------------
# Stage 1: Frontend Build
# ---------------------------------------------------------------------------
# Compiles the SolidJS + TypeScript frontend into static assets. The output
# in dist/ is later embedded into the Rust binary via rust-embed.

FROM node:25-bookworm AS frontend

WORKDIR /frontend

# Copy dependency manifests first for layer caching. npm ci uses the lockfile
# for deterministic, reproducible installs.
COPY crates/neuroncite-web/frontend/package.json \
     crates/neuroncite-web/frontend/package-lock.json \
     ./

RUN npm ci

# Copy frontend source and build. Vite outputs to dist/ with content-hashed
# filenames for cache busting.
COPY crates/neuroncite-web/frontend/ ./
RUN npx vite build


# ---------------------------------------------------------------------------
# Stage 2: Rust Build
# ---------------------------------------------------------------------------
# Compiles the NeuronCite workspace into a single release binary. Uses a
# two-phase approach: first build dependencies with stub source files (cached
# across rebuilds), then compile the actual source code.

FROM rust:1.94-bookworm AS builder

# Build-time system dependencies for native crate compilation.
# pkg-config + libssl-dev: required by reqwest's optional openssl backend
#   (even though neuroncite uses rustls, some transitive deps may probe for it)
# build-essential: C/C++ compiler for native code in ort-sys and other crates
# build-essential, pkg-config, libssl-dev: standard build toolchain
# libwayland-dev: required by wayland-sys (transitive dep from rfd file dialog crate)
# libdbus-1-dev: required by libdbus-sys (transitive dep from ashpd/rfd for portal dialogs)
# libgtk-3-dev: required by gtk-sys, gdk-sys, pango-sys, atk-sys (transitive deps
#   from the gtk crate used by neuroncite-web for the Linux splash screen window)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    libwayland-dev \
    libdbus-1-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# --- Phase 1: Dependency compilation with stub source files ---
# Copy only Cargo manifests and lockfile. Creating empty lib.rs/main.rs stubs
# lets cargo resolve and compile all external dependencies without the actual
# source code. This layer is cached as long as Cargo.toml/Cargo.lock stay
# unchanged, saving 10-15 minutes on rebuilds.

COPY Cargo.toml Cargo.lock ./

# Copy all crate-level Cargo.toml files (preserving directory structure).
# Every workspace member listed in the root Cargo.toml must have its manifest
# present here, otherwise cargo fails with "failed to load manifest for
# workspace member".
COPY crates/neuroncite-core/Cargo.toml            crates/neuroncite-core/Cargo.toml
COPY crates/neuroncite-pdf/Cargo.toml              crates/neuroncite-pdf/Cargo.toml
COPY crates/neuroncite-chunk/Cargo.toml            crates/neuroncite-chunk/Cargo.toml
COPY crates/neuroncite-embed/Cargo.toml            crates/neuroncite-embed/Cargo.toml
COPY crates/neuroncite-store/Cargo.toml            crates/neuroncite-store/Cargo.toml
COPY crates/neuroncite-search/Cargo.toml           crates/neuroncite-search/Cargo.toml
COPY crates/neuroncite-api/Cargo.toml              crates/neuroncite-api/Cargo.toml
COPY crates/neuroncite-mcp/Cargo.toml              crates/neuroncite-mcp/Cargo.toml
COPY crates/neuroncite-annotate/Cargo.toml         crates/neuroncite-annotate/Cargo.toml
COPY crates/neuroncite-citation/Cargo.toml         crates/neuroncite-citation/Cargo.toml
COPY crates/neuroncite-pipeline/Cargo.toml          crates/neuroncite-pipeline/Cargo.toml
COPY crates/neuroncite-web/Cargo.toml              crates/neuroncite-web/Cargo.toml
COPY crates/neuroncite-html/Cargo.toml             crates/neuroncite-html/Cargo.toml
COPY crates/neuroncite-llm/Cargo.toml              crates/neuroncite-llm/Cargo.toml
COPY crates/neuroncite-testgen/Cargo.toml          crates/neuroncite-testgen/Cargo.toml
COPY crates/neuroncite/Cargo.toml                  crates/neuroncite/Cargo.toml

# Create stub source files so cargo can resolve the workspace dependency graph.
# Library crates need lib.rs, binary crates need main.rs. neuroncite-testgen is
# a binary crate (generates synthetic test PDFs) so it receives a main.rs stub.
RUN mkdir -p crates/neuroncite-core/src         && echo "" > crates/neuroncite-core/src/lib.rs \
    && mkdir -p crates/neuroncite-pdf/src        && echo "" > crates/neuroncite-pdf/src/lib.rs \
    && mkdir -p crates/neuroncite-chunk/src      && echo "" > crates/neuroncite-chunk/src/lib.rs \
    && mkdir -p crates/neuroncite-embed/src      && echo "" > crates/neuroncite-embed/src/lib.rs \
    && mkdir -p crates/neuroncite-store/src      && echo "" > crates/neuroncite-store/src/lib.rs \
    && mkdir -p crates/neuroncite-search/src     && echo "" > crates/neuroncite-search/src/lib.rs \
    && mkdir -p crates/neuroncite-api/src         && echo "" > crates/neuroncite-api/src/lib.rs \
    && mkdir -p crates/neuroncite-pipeline/src   && echo "" > crates/neuroncite-pipeline/src/lib.rs \
    && mkdir -p crates/neuroncite-mcp/src         && echo "" > crates/neuroncite-mcp/src/lib.rs \
    && mkdir -p crates/neuroncite-annotate/src    && echo "" > crates/neuroncite-annotate/src/lib.rs \
    && mkdir -p crates/neuroncite-citation/src    && echo "" > crates/neuroncite-citation/src/lib.rs \
    && mkdir -p crates/neuroncite-web/src         && echo "" > crates/neuroncite-web/src/lib.rs \
    && mkdir -p crates/neuroncite-html/src        && echo "" > crates/neuroncite-html/src/lib.rs \
    && mkdir -p crates/neuroncite-llm/src         && echo "" > crates/neuroncite-llm/src/lib.rs \
    && mkdir -p crates/neuroncite-testgen/src     && echo "fn main() {}" > crates/neuroncite-testgen/src/main.rs \
    && mkdir -p crates/neuroncite/src             && echo "fn main() {}" > crates/neuroncite/src/main.rs \
    && echo "fn main() {}" > crates/neuroncite/build.rs \
    && mkdir -p crates/neuroncite/benches \
    && echo "fn main() {}" > crates/neuroncite/benches/hnsw_query.rs \
    && echo "fn main() {}" > crates/neuroncite/benches/bm25_query.rs \
    && echo "fn main() {}" > crates/neuroncite/benches/embedding_cpu.rs \
    && echo "fn main() {}" > crates/neuroncite/benches/pdf_extraction.rs \
    && echo "fn main() {}" > crates/neuroncite/benches/hnsw_construction.rs

# rust-embed requires the frontend dist/ directory to exist at compile time.
# Create an empty directory with a placeholder so the stub build succeeds.
RUN mkdir -p crates/neuroncite-web/frontend/dist \
    && echo "" > crates/neuroncite-web/frontend/dist/index.html

# Build dependencies only. This step is cached until Cargo.toml or Cargo.lock
# change. The --no-default-features flag with explicit features matches the
# final build configuration.
RUN cargo build --release --no-default-features \
    --features backend-ort,web,pdfium,ocr,mcp \
    2>&1 || true

# --- Phase 2: Full source compilation ---
# Remove stub source files and invalidate cargo fingerprints for all workspace
# crates. Without this, cargo reuses the empty .rlib artifacts from Phase 1
# because its fingerprint cache does not detect the source file replacement
# across Docker layers. External dependency artifacts remain cached.

RUN find crates/ -name "*.rs" -delete \
    && rm -rf target/release/.fingerprint/neuroncite* \
    && rm -rf target/release/build/neuroncite* \
    && find target/release/deps -name 'neuroncite*' -delete 2>/dev/null || true \
    && find target/release/deps -name 'libneuroncite*' -delete 2>/dev/null || true

# Copy the actual source code for all crates
COPY crates/ crates/

# Copy the compiled frontend assets from Stage 1 into the location where
# rust-embed expects them (crates/neuroncite-web/frontend/dist/).
COPY --from=frontend /frontend/dist/ crates/neuroncite-web/frontend/dist/

# Final release build with all features except gui (no native window in container).
# The release profile applies LTO (thin), single codegen unit, and symbol stripping
# as configured in the workspace Cargo.toml.
RUN cargo build --release --no-default-features \
    --features backend-ort,web,pdfium,ocr,mcp


# ---------------------------------------------------------------------------
# Stage 3: Runtime
# ---------------------------------------------------------------------------
# Runtime image for GPU-accelerated ONNX inference. The base image is selected
# via the RUNTIME_BASE build arg:
#   - nvidia (default): nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#   - rocm:             rocm/dev-ubuntu-22.04:6.4
#   - cpu:              ubuntu:22.04
# Contains only the compiled binary, runtime dependencies, and Ollama.

FROM ${RUNTIME_BASE} AS runtime

# Prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Re-declare build args after FROM (Docker scoping rule: ARGs before FROM are
# not available after FROM unless re-declared).
ARG GPU=nvidia
ARG TARGETARCH=amd64

# Map Docker TARGETARCH values to the naming conventions used by ORT and pdfium:
#   Docker "amd64" -> ORT "x64", pdfium "x64"
#   Docker "arm64" -> ORT "aarch64", pdfium "arm64"
# GPU variants (nvidia, rocm) are only available for x86_64.
# ARM64 builds are forced to CPU-only regardless of the GPU arg.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "aarch64" > /tmp/ort_arch && echo "arm64" > /tmp/pdfium_arch; \
    else \
        echo "x64" > /tmp/ort_arch && echo "x64" > /tmp/pdfium_arch; \
    fi

# Runtime system dependencies:
# - curl, ca-certificates: HTTP downloads for embedding models from HuggingFace
# - tesseract-ocr + language packs: OCR fallback for scanned PDFs
# - libgomp1: OpenMP threading library required by ONNX Runtime
# - libwayland-client0, libdbus-1-3: runtime shared libraries linked by the rfd
#   (file dialog) crate via wayland-sys and libdbus-sys transitive dependencies
# - libgtk-3-0: runtime shared library for GTK3, linked by the gtk crate used
#   by neuroncite-web for the Linux splash screen window
# - zstd: required by the Ollama install script for archive extraction
# - tini: minimal init system for proper PID 1 signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    libgomp1 \
    libwayland-client0 \
    libdbus-1-3 \
    libgtk-3-0 \
    zstd \
    tini \
    && rm -rf /var/lib/apt/lists/*

# --- ONNX Runtime 1.23.2 ---
# The ort crate is compiled with the load-dynamic feature, which loads
# libonnxruntime.so at runtime via dlopen. ORT_DYLIB_PATH tells the ort crate
# where to find the shared library.
#
# The download URL and archive directory name differ per GPU variant:
#   nvidia: onnxruntime-linux-{arch}-gpu-{ver}.tgz     (CUDA + cuDNN, x64 only)
#   rocm:   onnxruntime-linux-x64-rocm-{ver}.tgz       (ROCm HIP, x64 only)
#   cpu:    onnxruntime-linux-{arch}-{ver}.tgz          (CPU-only, x64 and arm64)
#
# On ARM64, GPU variants are not available from Microsoft's releases.
# The build falls back to CPU-only regardless of the GPU arg.
ARG ORT_VERSION=1.23.2
RUN ORT_ARCH=$(cat /tmp/ort_arch) \
    && if [ "$TARGETARCH" = "arm64" ]; then \
        # ARM64: force CPU-only (no GPU ORT release available) \
        ORT_SLUG="onnxruntime-linux-aarch64-${ORT_VERSION}"; \
        ORT_DIR="${ORT_SLUG}"; \
    else \
        case "${GPU}" in \
            nvidia) \
                ORT_SLUG="onnxruntime-linux-x64-gpu-${ORT_VERSION}"; \
                ORT_DIR="${ORT_SLUG}" ;; \
            rocm) \
                ORT_SLUG="onnxruntime-linux-x64-rocm-${ORT_VERSION}"; \
                ORT_DIR="${ORT_SLUG}" ;; \
            cpu|*) \
                ORT_SLUG="onnxruntime-linux-x64-${ORT_VERSION}"; \
                ORT_DIR="${ORT_SLUG}" ;; \
        esac; \
    fi \
    && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_SLUG}.tgz" \
    | tar xzf - --strip-components=1 -C /usr/local/ "${ORT_DIR}/lib/" \
    && (ldconfig || true)

# --- PDFium (pre-cached for volume seeding) ---
# The application checks for libpdfium.so at $HOME/Documents/NeuronCite/runtime/pdfium/
# (deps.rs:146-149). Since this path is on a volume mount, the entrypoint
# copies the library from this cache location on first start.
# PDFIUM_ARCH is architecture-aware: "x64" for amd64, "arm64" for arm64.
RUN PDFIUM_ARCH=$(cat /tmp/pdfium_arch) \
    && mkdir -p /opt/neuroncite/cache/pdfium \
    && curl -fsSL "https://github.com/bblanchon/pdfium-binaries/releases/download/chromium/7699/pdfium-linux-${PDFIUM_ARCH}.tgz" \
    | tar xzf - -C /opt/neuroncite/cache/pdfium \
    && cp /opt/neuroncite/cache/pdfium/lib/libpdfium.so /opt/neuroncite/cache/pdfium/ 2>/dev/null \
    || cp /opt/neuroncite/cache/pdfium/bin/libpdfium.so /opt/neuroncite/cache/pdfium/ 2>/dev/null \
    || true

# --- Ollama ---
# Ollama provides a local LLM inference server used by NeuronCite's citation
# verification agent. The install script places the binary at /usr/local/bin/ollama.
# Models are downloaded on-demand by the user through the Web UI.
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- NeuronCite binary ---
COPY --from=builder /build/target/release/neuroncite /usr/local/bin/neuroncite

# --- Entrypoint script ---
# sed removes Windows carriage returns (\r) that cause "No such file or
# directory" errors when the shebang line contains a trailing \r on Linux.
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Clean up temporary arch files
RUN rm -f /tmp/ort_arch /tmp/pdfium_arch

# --- Non-root user ---
# Run both Ollama and NeuronCite as a non-root user for security.
# The data directory is created and owned by this user.
RUN groupadd -r neuroncite && useradd -r -g neuroncite -m -d /data neuroncite \
    && mkdir -p /data/Documents/NeuronCite \
    && chown -R neuroncite:neuroncite /data /opt/neuroncite

# --- Environment ---
# HOME=/data: The application resolves its base directory to $HOME/Documents/NeuronCite
#   (paths.rs:51). All persistent data (models, indexes, runtime) lives here.
# ORT_DYLIB_PATH: Tells the ort crate where to find libonnxruntime.so for load-dynamic.
# LD_LIBRARY_PATH: GPU provider .so files (libonnxruntime_providers_cuda.so or
#   libonnxruntime_providers_rocm.so) are co-located with the main ORT library
#   and need to be on the linker search path.
# NVIDIA_*: Required by the NVIDIA Container Toolkit for GPU passthrough.
#   These are harmless no-ops on non-NVIDIA images (ignored without --gpus flag).
ENV HOME=/data \
    ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/usr/local/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OLLAMA_HOST=0.0.0.0:11434

USER neuroncite
WORKDIR /data

EXPOSE 3030

# Health check probes the NeuronCite REST API. The /api/v1/health endpoint
# returns system status (GPU availability, loaded backend, dependency state)
# without requiring authentication.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:3030/api/v1/health || exit 1

# tini handles PID 1 responsibilities (signal forwarding, zombie reaping)
# so the entrypoint script does not need to implement them directly.
ENTRYPOINT ["tini", "--"]
CMD ["/entrypoint.sh"]
