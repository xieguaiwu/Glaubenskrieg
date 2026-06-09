# =============================================================================
# Glaubenskrieg CTM+GBDT Stock Prediction — Multi-Stage Docker Build
# =============================================================================
# Build from the parent directory so both Glaubenskrieg/ and Hoffnung/ are in
# the build context:
#
#   cd /path/to/ML
#   docker build -t glaubenskrieg -f Glaubenskrieg/Dockerfile .
#
# Runtime:
#   docker run --rm glaubenskrieg python scripts/train.py --config configs/default.yaml --device cpu
#   docker run --rm -v $(pwd)/data:/app/data:ro glaubenskrieg python scripts/infer.py --ctm-ckpt checkpoints/best.pt --data /app/data/stock_prices.csv
# =============================================================================

ARG TORCH_VERSION=2.1.0

LABEL description="Glaubenskrieg CTM+GBDT Stock Prediction"
LABEL version="0.1.0"

# =============================================================================
# Stage 1: Builder — compile Hoffnung C++ GBDT core
# =============================================================================

FROM python:3.12-slim AS builder

ARG TORCH_VERSION

# Install build toolchain (single layer, clean cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libomp-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (provides libtorch C++ via torch.utils.cmake_prefix_path)
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" pybind11

# Copy Hoffnung C++ source
COPY Hoffnung /app/Hoffnung

# Build GBDT shared library + Python bindings
RUN cd /app/Hoffnung \
    && mkdir -p build \
    && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTorch_DIR="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
    && make -j"$(nproc)"

# =============================================================================
# Stage 2: Runtime — slim Python image with Glaubenskrieg + Hoffnung
# =============================================================================

FROM python:3.12-slim AS runtime

ARG TORCH_VERSION

# Install OpenMP runtime (libgomp1) — NOT libomp-dev (dev headers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled GBDT artifacts from builder
COPY --from=builder /app/Hoffnung/build/gbdt_python*.so /app/Hoffnung/build/
COPY --from=builder /app/Hoffnung/build/libgbdt_core.so /app/Hoffnung/build/
COPY --from=builder /app/Hoffnung/build/python /app/Hoffnung/build/python

# Copy Glaubenskrieg Python source
COPY Glaubenskrieg /app

# Install Python runtime dependencies
RUN pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    numpy \
    pandas \
    pyyaml \
    scipy

# Python path so that src.* and gbdt.* are importable
ENV PYTHONPATH="/app:/app/Hoffnung/build:/app/Hoffnung/build/python"

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# ── Default command: show help ──
CMD ["python", "scripts/train.py", "--help"]

# ── Health check: verify torch + GBDT imports work ──
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD python -c "import torch; from gbdt import GBDT; print('HEALTHY')"
