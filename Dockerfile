# Python Image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    # build-essential is often required for compiling statistical libraries
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
# UV_LINK_MODE=copy: Ensures files are copied, not hardlinked
# UV_COMPILE_BYTECODE=1: Compiles python files for faster startup
# UV_PYTHON_DOWNLOADS=0: Tells uv to use the system python
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    # Add the virtual environment to PATH immediately
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install Dependencies
# We copy only the manifest files first to cache this layer
COPY pyproject.toml uv.lock ./

# Use distinct cache for uv to speed up re-builds
# --frozen: strict sync using lockfile
# --no-install-project: install dependencies only, not the project root yet
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy the rest of the project
COPY . .

# Sync the project
# This installs the project itself and any changes since the last layer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

