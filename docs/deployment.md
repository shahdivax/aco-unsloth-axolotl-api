# Deployment Guide

This guide provides instructions for deploying the Unsloth Training API.

## Prerequisites

*   **Operating System:** Linux (recommended for server environments).
*   **Python:** Version 3.8 or newer.
*   **GPU:** An NVIDIA GPU compatible with CUDA and Unsloth is **essential**. Ensure appropriate CUDA drivers are installed system-wide.
*   **Git:** For cloning the repository.
*   **pip:** For installing Python dependencies.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/A-C-O-A/aco-chef.git # Replace with the actual repository URL
    cd aco-chef            # Replace with the cloned directory name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install FastAPI, Uvicorn, Unsloth, PyTorch (with CUDA support if available), and other necessary packages.

## Running the API

While you can run the API directly with `python main.py` for development, for a more production-like setup, using an ASGI server like Uvicorn with a process manager like Gunicorn is recommended.

**1. Using Uvicorn (Standalone):**
   You can run Uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
   ```
   *   `main:app`: Points to the `app` instance in your `main.py` file.
   *   `--host 0.0.0.0`: Makes the API accessible externally.
   *   `--port 8000`: Specifies the port.
   *   `--workers 1`: For Unsloth and GPU-bound tasks, it's often best to run a single Uvicorn worker and scale by running multiple instances of the API behind a load balancer if needed, each on a different GPU or with careful GPU resource management. Training multiple models simultaneously on one GPU via multiple workers can lead to OOM errors.

**2. Using Gunicorn with Uvicorn Workers:**
   Gunicorn acts as a process manager for Uvicorn workers.
   ```bash
   gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:8000
   ```
   *   `-w 1`: Number of worker processes. Similar to the Uvicorn standalone point, **use 1 worker** if each API instance is managing one GPU for training. If you have multiple GPUs and want to run one API instance per GPU, you'd run multiple Gunicorn setups, each pinned to a specific GPU (see GPU Pinning section).
   *   `-k uvicorn.workers.UvicornWorker`: Specifies Uvicorn worker type.
   *   `main:app`: Path to your FastAPI app instance.
   *   `-b 0.0.0.0:8000`: Bind address and port.

**Important Note on Workers and GPU Resources:**
Unsloth model training is resource-intensive, particularly GPU memory. Running multiple training jobs simultaneously (which multiple workers could facilitate if requests come in fast) on a single GPU will likely lead to out-of-memory (OOM) errors.
The current API design with background tasks processes jobs one by one from an in-memory queue *per instance* of the API.
If you need to scale to multiple GPUs:
*   Run multiple independent instances of this API (e.g., in separate Docker containers or processes).
*   Each instance should be configured to use a specific GPU (see GPU Pinning).
*   Use a load balancer to distribute requests among these instances.

## Environment Variables

Consider setting these environment variables as needed:

*   **`CUDA_VISIBLE_DEVICES`**: To assign specific GPUs to the API instance. For example, to use only GPU 0:
    ```bash
    export CUDA_VISIBLE_DEVICES=0
    ```
    To use GPU 1:
    ```bash
    export CUDA_VISIBLE_DEVICES=1
    ```
    This should be set before starting the Gunicorn/Uvicorn process.
*   **`WANDB_API_KEY`**: If you are using Weights & Biases for logging, ensure this API key is available in the environment where the API runs.

## Hardware Requirements

*   **GPU:** An NVIDIA GPU with sufficient VRAM for the models you intend to train. For 4-bit quantized versions of models like Llama 7B or Qwen 4B, GPUs with at least 12-16GB VRAM are recommended, but larger models will need more. Check Unsloth's documentation for model-specific requirements.
*   **RAM:** Sufficient system RAM (e.g., 16GB or more, depending on dataset sizes and other processes).
*   **Storage:** Enough disk space for datasets, model weights, and saved trained adapters.

## (Optional) Dockerization

For easier deployment and portability, you can containerize the API using Docker.

**Example `Dockerfile`:**

```dockerfile
# Use a base image with Python and CUDA
# The exact CUDA version should match what your PyTorch/Unsloth expects.
# e.g., nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Set environment variables to ensure Python outputs directly to console
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install Python, pip, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

COPY requirements.txt .

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
# Install dependencies into the virtual environment
RUN . /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt

COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Activate the virtual environment and run the application
# Note: Ensure main:app is correct and accessible
CMD ["/bin/bash", "-c", ". /opt/venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1"]
```

**Build and Run the Docker Container:**

1.  **Build the image:**
    ```bash
    docker build -t unsloth-training-api .
    ```
2.  **Run the container:**
    ```bash
    docker run --gpus all -p 8000:8000 unsloth-training-api
    ```
    *   `--gpus all`: Makes all host GPUs available to the container. You can specify particular GPUs if needed.
    *   `-p 8000:8000`: Maps port 8000 on the host to port 8000 in the container.

This Dockerfile is a starting point and might need adjustments based on your specific CUDA/PyTorch versions and production needs.
