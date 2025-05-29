import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Dict
import models # Assuming models.py is in the same directory
import training_worker # Assuming training_worker.py is in the same directory
import logging
import os
import threading
import time

# Ngrok setup for Kaggle environments
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("pyngrok not available. Ngrok tunneling will be disabled.")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI App Initialization
app = FastAPI(
    title="Unsloth Training API",
    description="An API to trigger and monitor Unsloth model training jobs.",
    version="0.1.0",
)

# In-memory Job Status Storage
# This will store job status. The structure for each job entry will be dictated by
# training_worker.run_training_job and the models.JobStatusResponse.
# Example: {"job_id_1": {"status": "running", "output_location": None, "error_details": None, "message": "Training in progress."}}
job_statuses: Dict[str, Dict] = {}

def setup_ngrok_tunnel(port: int = 8000):
    """
    Set up ngrok tunnel for Kaggle environments.
    Returns the public URL if successful, None otherwise.
    """
    if not NGROK_AVAILABLE:
        logger.warning("pyngrok is not available. Skipping ngrok setup.")
        return None
    
    try:
        # Check if running in Kaggle environment
        is_kaggle = os.path.exists('/kaggle/working')
        
        if is_kaggle:
            logger.info("Kaggle environment detected. Setting up ngrok tunnel...")
            
            # Kill any existing ngrok processes
            try:
                ngrok.kill()
            except:
                pass
            
            # Create tunnel
            public_url = ngrok.connect(port, proto="http")
            logger.info(f"🌐 Ngrok tunnel established!")
            logger.info(f"📡 Public URL: {public_url}")
            logger.info(f"🔗 API Documentation: {public_url}/docs")
            logger.info(f"📊 API Health Check: {public_url}/")
            
            return str(public_url)
        else:
            logger.info("Not running in Kaggle environment. Ngrok tunnel not needed.")
            return None
            
    except Exception as e:
        logger.error(f"Failed to setup ngrok tunnel: {e}")
        return None

def start_server_with_ngrok():
    """
    Start the FastAPI server with ngrok tunnel support.
    """
    import uvicorn
    
    port = 8000
    host = "0.0.0.0"
    
    # Setup ngrok tunnel if available and in Kaggle
    public_url = setup_ngrok_tunnel(port)
    
    if public_url:
        print("\n" + "="*60)
        print("🚀 UNSLOTH TRAINING API READY!")
        print("="*60)
        print(f"📡 Public URL: {public_url}")
        print(f"🔗 API Docs: {public_url}/docs")
        print(f"📊 Health Check: {public_url}/")
        print("="*60)
        print("\n🔥 Example GRPO Training Request:")
        print(f"""
curl -X POST "{public_url}/train" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "base_model_name": "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
    "training_type": "grpo",
    "dataset_identifier": "trl-internal-testing/hh-rlhf-trl-style-test-prompt-chosen-rejected",
    "lora_config": {{
      "r": 8,
      "lora_alpha": 16,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }},
    "trainer_config": {{
      "max_steps": 10,
      "per_device_train_batch_size": 2,
      "gradient_accumulation_steps": 4,
      "warmup_steps": 1,
      "learning_rate": 2e-4,
      "logging_steps": 1,
      "optim": "adamw_8bit",
      "weight_decay": 0.01,
      "lr_scheduler_type": "linear",
      "seed": 42
    }}
  }}'
""")
        print("\n📝 Check job status:")
        print(f'curl "{public_url}/status/{{job_id}}"')
        print("\n" + "="*60)
    else:
        print(f"\n🌐 Server running locally at: http://{host}:{port}")
        print(f"📚 API Documentation: http://{host}:{port}/docs")
    
    # Start the server
    uvicorn.run(app, host=host, port=port, log_level="info")

@app.post("/train", response_model=models.TrainingResponse, status_code=202)
async def train_model(request: models.TrainingRequest, background_tasks: BackgroundTasks):
    """
    Accepts a training request, queues the training job, and returns a job ID.
    """
    job_id = str(uuid.uuid4())
    logger.info(f"Received training request {job_id}. Payload: {request.model_dump_json(indent=2)}")

    # Initialize job status
    job_statuses[job_id] = {
        "status": "queued",
        "message": "Training job queued.",
        "output_location": None,
        "error_details": None
    }
    
    # Add the training job to background tasks
    # The training_worker.run_training_job will update job_statuses directly.
    background_tasks.add_task(training_worker.run_training_job, job_id, request, job_statuses)
    
    logger.info(f"Training job {job_id} has been queued.")
    
    return models.TrainingResponse(
        job_id=job_id,
        status="queued",
        message="Training job successfully queued.",
        output_location=None # Output location is not known at queue time
    )

@app.get("/status/{job_id}", response_model=models.JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Retrieves the status of a specific training job.
    """
    logger.info(f"Received status request for job ID: {job_id}")
    
    status_info = job_statuses.get(job_id)
    
    if not status_info:
        logger.warning(f"Job ID {job_id} not found in job_statuses.")
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    
    logger.info(f"Returning status for job ID {job_id}: {status_info}")
    
    # Ensure all fields from JobStatusResponse are present, even if None
    return models.JobStatusResponse(
        job_id=job_id,
        status=status_info.get("status", "unknown"),
        message=status_info.get("message"),
        output_location=status_info.get("output_location"),
        error_details=status_info.get("error_details")
    )

@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint for basic API health check.
    """
    return {"message": "Unsloth Training API is running."}

# To run this application (after installing requirements.txt):
# uvicorn main:app --reload
#
# Example TrainingRequest for testing with curl or httpie:
#
# Example for GRPO Training:
# {
#   "base_model_name": "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
#   "training_type": "grpo",
#   "dataset_identifier": "trl-internal-testing/hh-rlhf-trl-style-test-prompt-chosen-rejected",
#   "lora_config": {
#     "r": 8,
#     "lora_alpha": 16,
#     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
#   },
#   "trainer_config": {
#     "max_steps": 10,
#     "per_device_train_batch_size": 2,
#     "gradient_accumulation_steps": 4,
#     "warmup_steps": 1,
#     "learning_rate": 2e-4,
#     "logging_steps": 1,
#     "optim": "adamw_8bit",
#     "weight_decay": 0.01,
#     "lr_scheduler_type": "linear",
#     "seed": 42,
#     "output_dir": "./outputs_grpo" 
#   }
# }
# Note: The dataset 'trl-internal-testing/hh-rlhf-trl-style-test-prompt-chosen-rejected'
# is small and has 'prompt', 'chosen', 'rejected' columns suitable for GRPO.
# The 'output_dir' in trainer_config is technically required by HuggingFace/Unsloth configs,
# but the API worker will overwrite it based on 'output_dir_prefix' and job_id.
#
# Example for SFT Training:
# {
#   "base_model_name": "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
#   "training_type": "sft",
#   "dataset_identifier": "yahma/alpaca-cleaned",
#   "lora_config": {
#     "r": 8,
#     "lora_alpha": 16
#   },
#   "trainer_config": {
#     "max_steps": 10,
#     "per_device_train_batch_size": 2,
#     "gradient_accumulation_steps": 4,
#     "logging_steps": 1,
#     "learning_rate": 2e-4,
#     "output_dir": "./outputs_sft"
#   }
# }
# Note: For 'yahma/alpaca-cleaned', the SFT formatting function in the worker expects
# 'instruction', 'input', 'output' columns. This dataset has them.

if __name__ == "__main__":
    start_server_with_ngrok()
