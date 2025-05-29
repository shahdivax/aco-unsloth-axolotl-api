from pydantic import BaseModel # Removed field_validator as it's no longer used
from typing import Optional, List, Dict # List is no longer used for dataset_json_payload
from enum import Enum

class TrainingType(str, Enum):
    GRPO = "grpo"
    SFT = "sft"

# DatasetSourceType is no longer needed
# class DatasetSourceType(str, Enum):
#     HUGGINGFACE = "huggingface"
#     JSON_UPLOAD = "json_upload"

class TrainingRequest(BaseModel):
    base_model_name: str = "unsloth/Qwen2-4B-Instruct-bnb-4bit"
    training_type: TrainingType  # Added
    dataset_identifier: str      # Changed from Optional[str] to str; this is now the primary way to specify a dataset.
                                 # Assumes dataset is pre-formatted and available on Hugging Face Hub or locally accessible by path.
    trainer_config: Dict         # Renamed from grpo_config
    lora_config: Dict            # Kept as is
    output_dir_prefix: Optional[str] = "training_output"
    wandb_project: Optional[str] = None

    # The old validator 'validate_dataset_source' is removed as:
    # 1. dataset_source_type is removed.
    # 2. dataset_json_payload is removed.
    # 3. dataset_identifier is now always required, so Pydantic's default check suffices.

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    output_location: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    output_location: Optional[str] = None
    error_details: Optional[str] = None
