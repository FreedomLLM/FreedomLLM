# Set base image
FROM nvidia/cuda:13.5.0-runtime-ubuntu26.04

# Add image metadata
LABEL org.opencontainers.image.title="FreedomLLM" \
      org.opencontainers.image.source="https://github.com/FreedomLLM/FreedomLLM" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        clang-21 \
        llvm-21 \
        lld-21 \
        libllvmlibc-21-dev \
        libc6-dev \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set C/C++ compiler to Clang
ENV CC=clang-21 \
    CXX=clang++-21

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR=/usr/local/bin \
        UV_NO_MODIFY_PATH=1 \
        sh

