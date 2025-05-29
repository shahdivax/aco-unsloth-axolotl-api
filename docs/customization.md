# Customization Guide

This guide provides tips on how to customize your training jobs by selecting different models, training types, preparing your own datasets, and adjusting hyperparameters.

## Choosing a Base Model (`base_model_name`)

The API allows you to specify any Unsloth-compatible model from the Hugging Face Hub using the `base_model_name` parameter in the `/train` endpoint.

**Key Considerations:**

*   **Unsloth Compatibility:** Unsloth optimizes models for faster training and lower memory usage. While many Hugging Face Transformer models can be used, Unsloth maintains its own forks or provides specific support for popular architectures (e.g., Llama, Mistral, Qwen). It's best to refer to [Unsloth's GitHub](https://github.com/unslothai/unsloth) or their Hugging Face organization page for lists of well-tested models.
*   **Model Size:** Larger models (e.g., 7B parameters vs. 0.5B parameters) generally offer better performance but require significantly more GPU memory and longer training times.
*   **Quantization:** The API, following Unsloth's common practice, loads models in 4-bit by default (`load_in_4bit = True` in the `training_worker.py`). This greatly reduces memory usage. Ensure the model you choose has 4-bit compatible variants (e.g., BNB-quantized versions).
*   **Instruction-Tuned vs. Base Models:** For tasks requiring following instructions or chat-like behavior, instruction-tuned or chat models (often with "Instruct" or "Chat" in their names) are usually preferred over base pre-trained models, especially as a starting point for further fine-tuning.

**Example:**
To use a different Unsloth Qwen2 model:
`"base_model_name": "unsloth/Qwen2-1.5B-Instruct-bnb-4bit"`

## Choosing a Training Type (`training_type`)

The API supports different training methodologies via the `training_type` parameter. Choose the one that best fits your goal:

*   **`"grpo"` (Group Rank Preference Optimization):**
    *   **Use Case:** For preference tuning. This method refines a model based on direct feedback about which responses are "chosen" (better) and which are "rejected" (worse) for a given prompt.
    *   **Why:** Ideal for aligning models to specific styles, desired behaviors (e.g., helpfulness, harmlessness), or improving response quality based on human or model-generated preferences.
    *   **Base Model Note:** Often applied to an already instruction-tuned model to further steer its behavior.

*   **`"sft"` (Supervised Fine-Tuning):**
    *   **Use Case:** For instruction tuning or teaching the model new knowledge/tasks in a structured format.
    *   **Why:** Use this to adapt a base model (or an already instruction-tuned model) to follow instructions, perform specific tasks (like summarization, translation, question answering in a certain style), or learn from a corpus of instruction-response pairs.
    *   **Data Format:** The model learns from examples of instructions and desired outputs.

**Guidance:**
*   If you have a pre-trained model and want to teach it to follow instructions, start with **SFT**.
*   If you have an instruction-tuned model and want to refine its responses based on preferences (e.g., make it more concise, more detailed, or adhere to a specific persona), use **GRPO**.
*   You might even use SFT first, then GRPO, as a multi-stage fine-tuning process.

## Preparing Datasets (`dataset_identifier`)

All datasets must be hosted on the Hugging Face Hub and specified using the `dataset_identifier` parameter. The API currently does not support local dataset paths or direct JSON uploads.

**Crucial Column Requirements by Training Type:**

The `training_worker.py` script expects specific column names in your Hugging Face dataset based on the chosen `training_type`.

**1. For GRPO Training (`training_type: "grpo"`):**
Your dataset **must** contain the following three columns:
*   `prompt`: The input, query, or context provided to the model.
*   `chosen`: The preferred or higher-quality response to the `prompt`.
*   `rejected`: The less preferred or lower-quality response to the `prompt`.

**Example:**
If your dataset `my_org/my_grpo_data` on Hugging Face has these columns, you'd use:
`"dataset_identifier": "my_org/my_grpo_data"`

**2. For SFT Training (`training_type: "sft"`):**
Your dataset **must** contain the following columns:
*   `instruction`: The instruction or question for the model.
*   `output`: The desired response the model should generate.
*   `input` (optional): Additional context or input for the instruction. If this column is not present or is empty for some rows, it will be omitted from the formatted prompt.

**SFT Data Formatting (Handled by API):**
You provide the dataset with `instruction`, `output`, and optional `input` columns. The API's training worker will then internally format these into a single structured prompt string for each example (e.g., using an Alpaca-like template) before tokenization and training. This means you don't need to pre-format your SFT dataset into a single text field.

**Example:**
If your dataset `my_org/my_sft_data` on Hugging Face has `instruction`, `output`, and (optionally) `input` columns:
`"dataset_identifier": "my_org/my_sft_data"`
*(A common public example is `yahma/alpaca-cleaned`, which has these fields.)*

**General Dataset Considerations:**
*   **Column Mapping:** If your dataset uses different names for these required columns (e.g., "question" instead of "prompt"), you **must preprocess your dataset on the Hugging Face Hub first** to rename these columns. The current API version does not support automatic column remapping during the training job.
*   **Finding Datasets:** Explore the [Hugging Face Hub Datasets](https://huggingface.co/datasets). Many datasets are available for tasks like DPO (Direct Preference Optimization, similar structure to GRPO) and instruction tuning (SFT).

## Hyperparameter Tuning

Adjusting LoRA and trainer hyperparameters can significantly impact model performance and training efficiency.

**1. LoRA Configuration (`lora_config`):**

*   `r` (rank): A smaller `r` means fewer trainable parameters and faster training but might underfit. Common values: 8, 16, 32, 64.
*   `lora_alpha`: Scaling factor, often set to `r` or `2*r`.
*   `target_modules`: List of model layers to apply LoRA to. Unsloth often defaults to all linear layers. Specifying these can fine-tune performance. The training worker uses `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` as default if this field is not provided.
*   `lora_dropout`: Dropout probability for LoRA layers.

**2. Trainer Configuration (`trainer_config`):**

This object accepts parameters similar to Hugging Face's `TrainingArguments` and will be passed to the Unsloth `GRPOConfig` or `SFTConfig` based on the selected `training_type`. Refer to the [Hugging Face TrainingArguments documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) and [Unsloth's documentation/examples](https://github.com/unslothai/unsloth) for a full list.

*   **Common Arguments:**
    *   `learning_rate`: Crucial for training stability and performance. Typical values for fine-tuning are small (e.g., `1e-5` to `5e-4`).
    *   `per_device_train_batch_size`: Affects memory usage and training speed. Adjust based on GPU memory.
    *   `gradient_accumulation_steps`: Effectively increases batch size without increasing memory, at the cost of slower updates.
    *   `num_train_epochs` or `max_steps`: Control how long the training runs.
    *   `warmup_ratio` or `warmup_steps`: Helps stabilize training at the beginning.
    *   `logging_steps`: How often to log training progress.
    *   `output_dir`: (Required by config objects, but API overwrites it based on `output_dir_prefix` and `job_id`. See Usage Guide.)
    *   `max_seq_length`: (Defaulted by the worker if not in `trainer_config`. Ensure your model and data are suitable). Maximum sequence length for tokenization.
*   **Trainer-Specific Arguments:**
    *   Some arguments in `trainer_config` might be specific to `GRPOConfig` (e.g., `beta` for GRPO/DPO) or `SFTConfig` (e.g., `packing`, `dataset_num_proc`).
    *   The API handles basic necessary setup like `dataset_text_field` for SFT internally. For advanced, trainer-specific settings, consult Unsloth's documentation.

**General Tips for Tuning:**

*   **Start Small:** Begin with a small number of epochs/steps and a small batch size to ensure the training pipeline works.
*   **Monitor Training:** If using W&B (`wandb_project`), monitor loss curves and other metrics.
*   **Iterate:** Hyperparameter tuning is often an iterative process.
*   **Refer to Unsloth Examples:** The [Unsloth GitHub repository](https://github.com/unslothai/unsloth) contains many example notebooks that showcase effective hyperparameter settings for various models and tasks. These are excellent starting points.
