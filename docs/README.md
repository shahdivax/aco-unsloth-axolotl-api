# Unsloth Training API Documentation

Welcome to the documentation for the Unsloth Training API. This API allows you to programmatically trigger and manage the training of language models using the Unsloth library, optimized for speed and memory efficiency. It supports various training types, including Supervised Fine-Tuning (SFT) and preference tuning methods like GRPO (Group Rank Preference Optimization).

## Quick Start

1.  **Ensure Prerequisites:** You'll need Python 3.8+, an NVIDIA GPU with CUDA, and Git.
2.  **Clone & Install:**
    ```bash
    git clone https://github.com/A-C-O-A/aco-chef.git # Replace with the actual repository URL
    cd aco-chef            # Replace with the cloned directory name
    pip install -r requirements.txt
    ```
3.  **Run the API Locally:**
    ```bash
    python main.py
    ```
    The API will be available at `http://localhost:8000`.

## Documentation Sections

For detailed information, please refer to the following sections:

*   **[API Usage Guide](./usage.md):** Detailed explanation of API endpoints, request/response formats, and how to make calls.
*   **[Customization Guide](./customization.md):** Learn how to choose training types (SFT, GRPO), use different base models from Hugging Face, prepare datasets (via Hugging Face identifiers only), and tune hyperparameters for LoRA and the specific trainer.
*   **[Deployment Guide](./deployment.md):** Instructions for deploying the API in various environments, including using Gunicorn and Docker.

## Accessing Live API Docs

When the API is running (e.g., locally via `python main.py`), you can access the auto-generated interactive documentation:

*   **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
*   **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

These interfaces allow you to explore endpoints and even try them out directly from your browser.
