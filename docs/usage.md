# API Usage Guide

## Overview

This API provides an interface to trigger and monitor Unsloth model training. It supports different training methodologies like GRPO (Group Rank Preference Optimization) and SFT (Supervised Fine-Tuning). It allows specifying base models, datasets from Hugging Face, LoRA configurations, and training parameters.

## Running the API Locally

1.  **Prerequisites:**
    *   Python 3.8+
    *   GPU with CUDA drivers installed and configured.
    *   Git (for cloning the repository).
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/A-C-O-A/aco-chef.git
    cd aco-chef
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the API:**
    ```bash
    python main.py
    ```
    The API will typically be available at `http://localhost:8000`.

## Endpoints

### 1. Start Training: `POST /train`

This endpoint initiates a new model training job. The training process runs in the background.

**Request Body (`application/json`):**

*   `base_model_name` (string, default: `"unsloth/Qwen2-4B-Instruct-bnb-4bit"`): Identifier of the base model from Hugging Face (e.g., `"unsloth/Qwen2-0.5B-Instruct-bnb-4bit"`).
*   `training_type` (string, enum: `"grpo"`, `"sft"`): Specifies the training methodology.
*   `dataset_identifier` (string): The Hugging Face Hub dataset identifier (e.g., `"username/my_dataset_name"`).
    *   **For GRPO (`training_type: "grpo"`)**: This dataset must contain columns: `prompt`, `chosen`, `rejected`.
    *   **For SFT (`training_type: "sft"`)**: This dataset must contain columns: `instruction`, `output`. An `input` column is optional and will be used if present. The API's training worker will format these into a prompt suitable for instruction fine-tuning (e.g., Alpaca format).
*   `lora_config` (object): Configuration for LoRA (Low-Rank Adaptation).
    *   `r` (integer, e.g., `8`, `16`): LoRA rank.
    *   `lora_alpha` (integer, e.g., `16`, `32`): LoRA alpha.
    *   `target_modules` (array of strings, optional): Modules to apply LoRA to (e.g., `["q_proj", "k_proj", "v_proj"]`). Defaults are provided if omitted.
    *   Other parameters like `lora_dropout`, `bias` can be included. See Unsloth/PEFT documentation for more.
*   `trainer_config` (object): Configuration arguments for the selected trainer (`GRPOTrainer` or `SFTTrainer`). These are similar to Hugging Face `TrainingArguments`.
    *   `output_dir` (string): This field is required by the underlying Hugging Face/Unsloth Config objects. However, the API's training worker will **overwrite** this value. The actual output path for a job will be constructed as `"<output_dir_prefix>/<job_id>/"`. You must still provide a string value for `output_dir` in your request (e.g., `"./output"`), but it won't be used for path construction.
    *   `max_steps` (integer, optional): Total number of training steps.
    *   `num_train_epochs` (integer, optional): Total number of training epochs.
    *   `per_device_train_batch_size` (integer): Batch size per GPU.
    *   `learning_rate` (float, e.g., `5e-5`).
    *   `logging_steps` (integer): Log every X updates steps.
    *   Many other `TrainingArguments` can be specified here relevant to the chosen `training_type`.
*   `output_dir_prefix` (string, optional, default: `"training_output"`): Prefix for the main output directory where job-specific folders will be created. The API will create a unique job folder like `training_output/<job_id>/`.
*   `wandb_project` (string, optional): Name of the Weights & Biases project to log to. If omitted, W&B logging is disabled.

**Example Request (GRPO Training):**

```bash
curl -X POST "http://localhost:8000/train" \
-H "Content-Type: application/json" \
-d '{
  "base_model_name": "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
  "training_type": "grpo",
  "dataset_identifier": "trl-internal-testing/hh-rlhf-trl-style-test-prompt-chosen-rejected",
  "lora_config": {
    "r": 8,
    "lora_alpha": 16
  },
  "trainer_config": {
    "max_steps": 3,
    "per_device_train_batch_size": 1,
    "learning_rate": 5e-5,
    "logging_steps": 1,
    "output_dir": "./placeholder_grpo_output"
  },
  "output_dir_prefix": "my_grpo_trainings"
}'
```

**Example Request (SFT Training):**

```bash
curl -X POST "http://localhost:8000/train" \
-H "Content-Type: application/json" \
-d '{
  "base_model_name": "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
  "training_type": "sft",
  "dataset_identifier": "yahma/alpaca-cleaned",
  "lora_config": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  },
  "trainer_config": {
    "max_steps": 10,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "logging_steps": 1,
    "output_dir": "./placeholder_sft_output"
  },
  "output_dir_prefix": "my_sft_trainings"
}'
```
*(Note: For SFT with `yahma/alpaca-cleaned`, the dataset provides `instruction`, `input`, `output` columns, which the API worker formats internally.)*

**Success Response (`202 Accepted`):**

```json
{
  "job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "queued",
  "message": "Training job successfully queued.",
  "output_location": null
}
```

### 2. Get Job Status: `GET /status/{job_id}`

Retrieves the status of a specific training job.

**Path Parameters:**

*   `job_id` (string): The unique ID of the job, returned from the `/train` endpoint.

**Example Request:**

```bash
curl -X GET "http://localhost:8000/status/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

**Success Response (`200 OK`):**

The response shows the current state of the job.
Possible statuses: `"queued"`, `"starting"`, `"running"`, `"completed"`, `"failed"`.

```json
{
  "job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "completed",
  "message": "Training completed and model adapters saved.",
  "output_location": "my_grpo_trainings/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/final_lora_adapters", 
  "error_details": null
}
```
*(Note: The `output_location` path combines the `output_dir_prefix` and the `job_id`. The `final_lora_adapters` segment is the specific directory within the job's output folder where the trained LoRA adapters are saved by the worker.)*

**Error Response (`404 Not Found`):**

If the `job_id` does not exist.
```json
{
  "detail": "Job ID 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx' not found."
}
```

## Auto-Generated API Documentation

For an interactive API documentation (Swagger UI) and alternative ReDoc view, visit these URLs when the API is running:

*   **Swagger UI:** `http://localhost:8000/docs`
*   **ReDoc:** `http://localhost:8000/redoc`

This allows you to try out the API endpoints directly from your browser.
