import asyncio
import os
import torch
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig
import wandb
import logging
import models # Assuming models.py is in the same directory
from models import TrainingType # Added TrainingType import

# Logging setup will be inherited from main.py
logger = logging.getLogger(__name__)

async def run_training_job(job_id: str, request: models.TrainingRequest, job_status_storage: dict):
    logger.info(f"Starting training job {job_id} for type: {request.training_type.value}")
    job_status_storage[job_id] = {
        "status": "starting", 
        "output_location": None, 
        "error_details": None, 
        "message": f"Training process initiated for type: {request.training_type.value}."
    }
    await asyncio.sleep(1) 

    original_wandb_project = os.environ.get("WANDB_PROJECT")
    original_wandb_disabled = os.environ.get("WANDB_DISABLED")

    model = None # Initialize model to None for broader scope access if needed in finally
    tokenizer = None # Initialize tokenizer

    try:
        # 1. Model Loading (Common)
        logger.info(f"[{job_id}] Loading base model: {request.base_model_name}")
        # Get max_seq_length from trainer_config or default to 2048 (used for both model and LoRA)
        # Both GRPOConfig and SFTConfig have max_seq_length.
        # We take it from the user's trainer_config for consistency.
        max_seq_length_from_config = request.trainer_config.get('max_seq_length', 2048)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=request.base_model_name,
            max_seq_length=max_seq_length_from_config,
            dtype=None,
            load_in_4bit=True,
        )
        logger.info(f"[{job_id}] Base model {request.base_model_name} loaded successfully.")
        await asyncio.sleep(1)

        # 2. PEFT (LoRA) Configuration (Common)
        logger.info(f"[{job_id}] Configuring LoRA with params: {request.lora_config}")
        lora_params = request.lora_config
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_params.get('r', 16),
            target_modules=lora_params.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_alpha=lora_params.get('lora_alpha', 16),
            lora_dropout=lora_params.get('lora_dropout', 0),
            bias=lora_params.get('bias', "none"),
            use_gradient_checkpointing=lora_params.get('use_gradient_checkpointing', True), # SFT default is True, GRPO might also benefit
            random_state=lora_params.get('random_state', 3407),
        )
        logger.info(f"[{job_id}] LoRA configured successfully for the model.")
        await asyncio.sleep(1)

        # 3. Dataset Loading (Common part)
        logger.info(f"[{job_id}] Loading dataset from identifier: {request.dataset_identifier}")
        try:
            # Assuming "train" split. This could be made configurable.
            dataset = load_dataset(request.dataset_identifier, split="train")
            logger.info(f"[{job_id}] Dataset '{request.dataset_identifier}' loaded. Columns: {dataset.column_names}")
        except Exception as e:
            logger.error(f"[{job_id}] Failed to load dataset '{request.dataset_identifier}': {e}", exc_info=True)
            job_status_storage[job_id].update({
                "status": "failed", 
                "message": f"Failed to load dataset: {str(e)}",
                "error_details": str(e)
            })
            return
        
        job_status_storage[job_id]["message"] = "Dataset loaded, preparing for training..."
        await asyncio.sleep(1)

        trainer = None
        training_args = None # To hold specific config (GRPOConfig or SFTConfig)

        # 4. Conditional Logic for Training Type
        output_directory = os.path.join(request.output_dir_prefix or "training_output", job_id)
        os.makedirs(output_directory, exist_ok=True)
        
        trainer_args_dict = request.trainer_config.copy()
        trainer_args_dict["output_dir"] = output_directory # Common for both trainer types
        
        # Ensure max_seq_length is in trainer_args_dict for the Config objects
        if 'max_seq_length' not in trainer_args_dict:
            trainer_args_dict['max_seq_length'] = max_seq_length_from_config
        
        # Default learning_rate if not specified (common for both)
        if 'learning_rate' not in trainer_args_dict:
            trainer_args_dict['learning_rate'] = 5e-5 # A common default

        if request.training_type == TrainingType.GRPO:
            logger.info(f"[{job_id}] Configuring for GRPO training.")
            required_cols = ["prompt", "chosen", "rejected"]
            if not all(col in dataset.column_names for col in required_cols):
                missing_cols_msg = f"Dataset for GRPO missing required columns. Expected: {required_cols}, Got: {dataset.column_names}"
                logger.error(f"[{job_id}] {missing_cols_msg}")
                job_status_storage[job_id].update({"status": "failed", "message": missing_cols_msg, "error_details": missing_cols_msg})
                return

            # GRPO specific defaults if any (currently none identified beyond common ones)
            training_args = GRPOConfig(**trainer_args_dict)
            trainer = GRPOTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=training_args)
            logger.info(f"[{job_id}] GRPOTrainer initialized.")

        elif request.training_type == TrainingType.SFT:
            logger.info(f"[{job_id}] Configuring for SFT training.")
            # For SFT, common column names are 'instruction', 'input', 'output' or just 'text'
            # The formatting function will create 'formatted_text'
            # We must check for the source columns for the formatting function.
            required_sft_source_cols = ["instruction", "output"] # 'input' is optional
            if not all(col in dataset.column_names for col in required_sft_source_cols):
                missing_cols_msg = f"Dataset for SFT (for formatting) missing required source columns. Expected: {required_sft_source_cols}, Got: {dataset.column_names}"
                logger.error(f"[{job_id}] {missing_cols_msg}")
                job_status_storage[job_id].update({"status": "failed", "message": missing_cols_msg, "error_details": missing_cols_msg})
                return

            # Nested formatting function for SFT
            def format_sft_dataset(examples):
                PROMPT_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}"
                INPUT_TEMPLATE = "### Input:\n{input}"
                RESPONSE_TEMPLATE = "### Response:\n{output}"
                
                texts = []
                for i in range(len(examples["instruction"])):
                    text_parts = [PROMPT_TEMPLATE]
                    text_parts.append(INSTRUCTION_TEMPLATE.format(instruction=examples["instruction"][i]))
                    if "input" in examples and examples["input"][i] is not None and str(examples["input"][i]).strip() != "":
                        text_parts.append(INPUT_TEMPLATE.format(input=examples["input"][i]))
                    text_parts.append(RESPONSE_TEMPLATE.format(output=examples["output"][i]))
                    texts.append("\n\n".join(text_parts) + tokenizer.eos_token) # tokenizer is in closure
                return {"formatted_text": texts}

            logger.info(f"[{job_id}] Mapping dataset for SFT training...")
            # Keep __index_level_0__ if it exists, remove other original columns
            cols_to_remove = [name for name in dataset.column_names if name != "__index_level_0__"]
            processed_dataset = dataset.map(
                format_sft_dataset, 
                batched=True, 
                remove_columns=cols_to_remove
            )
            logger.info(f"[{job_id}] SFT dataset formatted. New columns: {processed_dataset.column_names}")
            
            # Set dataset_text_field for SFTTrainer
            trainer_args_dict["dataset_text_field"] = "formatted_text" 
            
            # Ensure all required SFTConfig parameters are properly set with valid defaults
            # Fix potential None value issues that could cause comparison errors
            sft_defaults = {
                'max_length': max_seq_length_from_config,  # Ensure this is set and not None
                'per_device_train_batch_size': trainer_args_dict.get('per_device_train_batch_size', 1),
                'gradient_accumulation_steps': trainer_args_dict.get('gradient_accumulation_steps', 4),
                'max_steps': trainer_args_dict.get('max_steps', 10),
                'logging_steps': trainer_args_dict.get('logging_steps', 1),
                'learning_rate': trainer_args_dict.get('learning_rate', 2e-4),
                'warmup_ratio': trainer_args_dict.get('warmup_ratio', 0.03),
                'lr_scheduler_type': trainer_args_dict.get('lr_scheduler_type', 'constant'),
                'save_strategy': trainer_args_dict.get('save_strategy', 'steps'),
                'save_steps': trainer_args_dict.get('save_steps', 0.5),
                'packing': trainer_args_dict.get('packing', False),
                'dataset_text_field': 'formatted_text'
            }
            
            # Update trainer_args_dict with defaults for any missing values
            for key, default_value in sft_defaults.items():
                if key not in trainer_args_dict or trainer_args_dict[key] is None:
                    trainer_args_dict[key] = default_value

            # Additional safety check - ensure no None values in critical parameters
            logger.info(f"[{job_id}] Final SFT config validation before creating SFTConfig...")
            for key, value in trainer_args_dict.items():
                if value is None and key in ['max_length', 'max_steps', 'per_device_train_batch_size', 
                                           'gradient_accumulation_steps', 'learning_rate', 'logging_steps']:
                    raise ValueError(f"Critical SFT parameter '{key}' is still None after defaults applied!")
                logger.info(f"[{job_id}] SFT Config - {key}: {value} (type: {type(value)})")

            # Additional check for required numeric parameters
            required_numeric_params = {
                'max_length': (int, 1, 8192),
                'max_steps': (int, 1, 100000),
                'per_device_train_batch_size': (int, 1, 64),
                'gradient_accumulation_steps': (int, 1, 128),
                'learning_rate': (float, 1e-6, 1e-1),
                'logging_steps': (int, 1, 1000)
            }
            
            for param_name, (expected_type, min_val, max_val) in required_numeric_params.items():
                if param_name in trainer_args_dict:
                    value = trainer_args_dict[param_name]
                    if not isinstance(value, expected_type):
                        try:
                            # Try to convert to expected type
                            trainer_args_dict[param_name] = expected_type(value)
                            logger.info(f"[{job_id}] Converted {param_name} from {type(value)} to {expected_type}")
                        except (TypeError, ValueError) as e:
                            raise ValueError(f"Parameter '{param_name}' cannot be converted to {expected_type}: {value}")
                    
                    # Check value range
                    converted_value = trainer_args_dict[param_name]
                    if not (min_val <= converted_value <= max_val):
                        logger.warning(f"[{job_id}] Parameter '{param_name}' value {converted_value} is outside recommended range [{min_val}, {max_val}]")

            logger.info(f"[{job_id}] All SFT parameters validated successfully.")

            training_args = SFTConfig(**trainer_args_dict)
            trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=processed_dataset, args=training_args)
            logger.info(f"[{job_id}] SFTTrainer initialized.")
        
        else:
            unknown_type_msg = f"Unknown training type: {request.training_type}"
            logger.error(f"[{job_id}] {unknown_type_msg}")
            job_status_storage[job_id].update({"status": "failed", "message": unknown_type_msg, "error_details": unknown_type_msg})
            return

        job_status_storage[job_id]["message"] = "Trainer initialized, preparing for training run."
        await asyncio.sleep(1)

        # 5. WandB Setup (Common)
        if request.wandb_project:
            os.environ["WANDB_PROJECT"] = request.wandb_project
            if "WANDB_DISABLED" in os.environ:
                del os.environ["WANDB_DISABLED"]
            logger.info(f"[{job_id}] Logging to W&B project: {request.wandb_project}")
        else:
            # Properly disable W&B to avoid interactive prompts
            os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_MODE"] = "disabled"
            os.environ["WANDB_SILENT"] = "true"
            logger.info(f"[{job_id}] W&B logging disabled.")
        
        job_status_storage[job_id]["message"] = "WandB configured, starting training..."
        await asyncio.sleep(1)
        
        # Log combined training args
        # The training_args object (GRPOConfig or SFTConfig) has all computed args.
        logger.info(f"[{job_id}] Training arguments for {request.training_type.value}: {training_args.to_dict()}")


        # 6. Training (Common)
        logger.info(f"[{job_id}] Starting model training (trainer.train())...")
        job_status_storage[job_id]["status"] = "running"
        job_status_storage[job_id]["message"] = "Model training in progress."
        
        # Add detailed validation before training
        try:
            logger.info(f"[{job_id}] Validating training configuration...")
            
            # Validate training_args configuration
            config_dict = training_args.to_dict()
            logger.info(f"[{job_id}] Training config validation - checking for None values...")
            
            critical_params = ['max_length', 'max_steps', 'per_device_train_batch_size', 
                             'gradient_accumulation_steps', 'learning_rate', 'logging_steps']
            
            for param in critical_params:
                if param in config_dict:
                    value = config_dict[param]
                    if value is None:
                        raise ValueError(f"Critical parameter '{param}' is None. This will cause comparison errors.")
                    logger.info(f"[{job_id}] ✓ {param}: {value} (type: {type(value)})")
                else:
                    logger.warning(f"[{job_id}] Parameter '{param}' not found in config")
            
            # Validate model and tokenizer
            if model is None:
                raise ValueError("Model is None before training")
            if tokenizer is None:
                raise ValueError("Tokenizer is None before training")
                
            # Validate dataset
            if request.training_type == TrainingType.SFT:
                if len(processed_dataset) == 0:
                    raise ValueError("Processed dataset is empty")
                logger.info(f"[{job_id}] Dataset validation: {len(processed_dataset)} samples, columns: {processed_dataset.column_names}")
            else:
                if len(dataset) == 0:
                    raise ValueError("Dataset is empty")
                logger.info(f"[{job_id}] Dataset validation: {len(dataset)} samples, columns: {dataset.column_names}")
            
            logger.info(f"[{job_id}] All validations passed. Starting trainer.train()...")
            
        except Exception as validation_error:
            logger.error(f"[{job_id}] Validation failed: {validation_error}")
            job_status_storage[job_id].update({
                "status": "failed", 
                "error_details": f"Validation error: {str(validation_error)}", 
                "message": f"Pre-training validation failed: {str(validation_error)}"
            })
            return
        
        # Attempt training with enhanced error handling
        try:
            trainer.train() 
            logger.info(f"[{job_id}] Model training (trainer.train()) finished successfully.")
            
        except Exception as training_error:
            # Catch and provide detailed error information
            error_msg = str(training_error)
            error_type = type(training_error).__name__
            
            logger.error(f"[{job_id}] Training failed with {error_type}: {error_msg}")
            logger.error(f"[{job_id}] Full traceback:", exc_info=True)
            
            # Check for specific error patterns
            if "'<' not supported between instances of 'NoneType' and 'int'" in error_msg:
                detailed_error = (
                    f"NoneType comparison error detected. This typically occurs when:\n"
                    f"1. A required parameter is None instead of a valid number\n"
                    f"2. max_length, max_steps, or batch_size parameters are None\n"
                    f"3. Learning rate or other numeric parameters are not properly set\n"
                    f"Current config: {config_dict}\n"
                    f"Original error: {error_msg}"
                )
                logger.error(f"[{job_id}] Detailed analysis: {detailed_error}")
                
                job_status_storage[job_id].update({
                    "status": "failed", 
                    "error_details": detailed_error, 
                    "message": f"Training failed due to NoneType comparison error - check parameter configuration"
                })
            else:
                job_status_storage[job_id].update({
                    "status": "failed", 
                    "error_details": f"{error_type}: {error_msg}", 
                    "message": f"Training failed due to: {error_msg}"
                })
            return
        
        logger.info(f"[{job_id}] Training completed successfully.")
        job_status_storage[job_id]["message"] = "Training completed."
        await asyncio.sleep(1)

        # 7. Save Model & Update Status - Success (Common)
        logger.info(f"[{job_id}] Saving model adapters post-training...")
        final_output_path = os.path.join(output_directory, "final_lora_adapters") # output_directory is already job-specific
        
        # Save adapters using the PEFT model's method
        model.save_pretrained(final_output_path) 
        tokenizer.save_pretrained(final_output_path) # Save tokenizer too
        logger.info(f"[{job_id}] Model adapters saved to {final_output_path}")
        
        job_status_storage[job_id] = {
            "status": "completed",
            "output_location": final_output_path,
            "error_details": None,
            "message": "Training completed and model adapters saved."
        }

    except Exception as e:
        logger.error(f"Error in training job {job_id}: {e}", exc_info=True)
        job_status_storage[job_id].update({ # Use update to preserve existing fields if any partial failure happened
            "status": "failed", 
            "error_details": str(e), 
            "message": f"Training failed due to: {str(e)}"
        })
        # No explicit return needed here as finally will execute and function will end.
        # However, adding one for clarity as per previous patterns might be considered if required.

    finally:
        # Cleanup W&B env vars
        if original_wandb_project is None:
            if "WANDB_PROJECT" in os.environ: del os.environ["WANDB_PROJECT"]
        else:
            os.environ["WANDB_PROJECT"] = original_wandb_project

        if original_wandb_disabled is None:
            if "WANDB_DISABLED" in os.environ: del os.environ["WANDB_DISABLED"]
        else:
            os.environ["WANDB_DISABLED"] = original_wandb_disabled
        
        # Clean up model and tokenizer to free GPU memory
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if 'trainer' in locals() and trainer is not None: # trainer might not be initialized if error occurs early
            del trainer
        
        torch.cuda.empty_cache() # Attempt to clear CUDA cache
        logger.info(f"[{job_id}] CUDA cache cleared.")
        
        logger.info(f"Finished processing job {job_id}. Final status: {job_status_storage.get(job_id, {}).get('status')}")
