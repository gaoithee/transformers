import os

from handcoded_tokenizer import STLTokenizer
from configuration import STLConfig
from modeling_stldec import STLForCausalLM
from utils import CustomDataset

import argparse
import json
import logging
import math
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from torch.utils.data import Dataset
import ast
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers import AutoConfig, AutoModel
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = get_logger(__name__)

# Extend `AutoClasses` to support the custom model
AutoConfig.register("STLdec", STLConfig)
AutoModel.register(STLConfig, STLForCausalLM)

# Initialize the model with random weights and the desired architecture
config = STLConfig()
model = AutoModel.from_config(config)
tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')

##########################################################################

# Upload training configuration file 
with open('training_config.json', 'r') as f:
    args = json.load(f)

# Read from `training_config.json` and define the corresponding variables
dataset_name = args['dataset_name']
train_file = args['train_file']
validation_file = args['validation_file']
output_dir = args['output_dir']
model_name_or_path = args['model_name_or_path']
tokenizer_name = args['tokenizer_name']
block_size = args['block_size']
batch_size = args['batch_size']
gradient_accumulation_steps = args['gradient_accumulation_steps']
num_train_epochs = args['num_train_epochs']
learning_rate = args['learning_rate']
weight_decay = args['weight_decay']
seed = args['seed']
with_tracking = args['with_tracking']
hub_model_id = args['hub_model_id']
push_to_hub = args['push_to_hub']
trust_remote_code = args['trust_remote_code']
overwrite_cache = args['overwrite_cache']
per_device_train_batch_size = args['per_device_train_batch_size']
per_device_eval_batch_size = args['per_device_eval_batch_size']
checkpointing_steps = args['checkpointing_steps']
resume_from_checkpoint = args['resume_from_checkpoint']
lr_scheduler_type = args['lr_scheduler_type']
num_warmup_steps = args['num_warmup_steps']
max_train_steps = args['max_train_steps']

private_hub_token = "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"
##########################################################################


# Initialize the accelerator. In this example, we let the accelerator manage device placement for us.
# If tracking is enabled, the accelerator will also automatically initialize the tracking system and will 
# pick up any available trackers in the environment by default.
accelerator_log_kwargs = {}

# Check if tracking is enabled and configure the tracking settings accordingly
if args["with_tracking"]:
    # Set the project directory for tracking logs (can specify a different tracker such as TensorBoard, etc.)
    accelerator_log_kwargs["project_dir"] = output_dir

# Create an accelerator instance with gradient accumulation steps and other tracking configurations
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, **accelerator_log_kwargs)

# Configure logging for debugging purposes. This will log detailed information for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # Set logging level to INFO for detailed logs
)

# Adjust the verbosity of logs based on whether the current process is the main one.
# Main process will show more detailed logs, while others will suppress less important logs.
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()  # Show warnings for dataset loading
    transformers.utils.logging.set_verbosity_info()  # Show detailed logs for transformers
else:
    datasets.utils.logging.set_verbosity_error()  # Show only errors for non-main processes
    transformers.utils.logging.set_verbosity_error()  # Show only errors for non-main processes

# If a specific seed is provided, set the random seed to ensure reproducibility of results.
if seed is not None:
    set_seed(seed)

# Handle the repository creation if the current process is the main process.
if accelerator.is_main_process:
    # Check if model should be pushed to the Hugging Face Hub
    if push_to_hub:
        # Retrieve or infer the repository name for the model
        repo_name = hub_model_id
        if repo_name is None:
            # If no repository ID is provided, use the output directory name as the repo name
            repo_name = Path(output_dir).absolute().name
        # Create the repository on the Hugging Face Hub and retrieve the repository ID
        api = HfApi()
        repo_id = api.create_repo(repo_name, exist_ok=True, token=private_hub_token).repo_id
        # Write entries to .gitignore to avoid saving unnecessary files (e.g., checkpoints)
        with open(os.path.join(output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    # If not pushing to the hub, ensure the output directory exists
    elif output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

# Wait for all processes to synchronize before proceeding with the next steps.
accelerator.wait_for_everyone()

# If a dataset name is provided, load it from the Hugging Face Hub.
# Note: this is not used!
if dataset_name is not None:
    # Download and load the dataset from the hub with the specified configuration
    raw_datasets = load_dataset(
        args.dataset_name, args["dataset_config_name"], trust_remote_code=args["trust_remote_code"]
    )
    # If the dataset doesn't include a validation split, create one using the training data.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args["dataset_name"],
            args["dataset_config_name"],
            trust_remote_code=args["trust_remote_code"],
        )
        raw_datasets["train"] = load_dataset(
            args["dataset_name"],
            args["dataset_config_name"],
            trust_remote_code=args["trust_remote_code"],
        )

else:
    # If no dataset name is provided, load the data from local files instead.
    data_files = {}
    dataset_args = {}
    # Add the training file to the data_files dictionary if provided
    if train_file is not None:
        data_files["train"] = train_file
        extension = train_file.split(".")[-1]
    # Add the validation file to the data_files dictionary if provided
    if validation_file is not None:
        data_files["validation"] = validation_file
        extension = validation_file.split(".")[-1]
    # Handle the case where the dataset extension is ".txt"
    if extension == "txt":
        extension = "text"

    # Load the dataset based on the extension (e.g., 'csv', 'json', 'text', etc.)
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)


# Set the device to GPU if available, otherwise default to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the training and evaluation datasets, moving them to the specified device (GPU or CPU)
train_dataset = CustomDataset(raw_datasets['train'], device=device)
eval_dataset = CustomDataset(raw_datasets['validation'], device=device)

# Create DataLoaders for training and evaluation.
# The DataLoader is responsible for batching the data, shuffling it, and using the specified collate function.
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True,  # Shuffle the data to ensure randomness in batches
    collate_fn=default_data_collator,  # The function used to combine individual data samples into a batch
    batch_size=args["batch_size"]  # Set the batch size as defined in the arguments
)

eval_dataloader = DataLoader(
    eval_dataset, 
    shuffle=True,  # Shuffle for evaluation as well (typically not needed but can be done for randomness)
    collate_fn=default_data_collator,  # Same collate function as training
    batch_size=args["batch_size"]  # Set the batch size for evaluation
)

# Optimizer setup:
# Split model parameters into two groups:
# 1. Parameters that will have weight decay (e.g., all except biases and layer norm weights).
# 2. Parameters that won't have weight decay (biases and layer norm weights).
no_decay = ["bias", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args["weight_decay"],  # Apply weight decay here
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,  # No weight decay for biases and layer norm weights
    },
]

# Initialize the AdamW optimizer with the grouped parameters, learning rate, and beta values.
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters, 
    lr=args["learning_rate"],  # Set the learning rate
    betas=(0.9, 0.99)  # Set the beta values for the Adam optimizer (momentum and variance decay)
)

# Scheduler setup:
# Calculate the number of updates per epoch considering gradient accumulation.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])

# If `max_train_steps` is not provided, we calculate it based on the number of epochs and updates per epoch.
if args["max_train_steps"] is None:
    args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch
    overrode_max_train_steps = True

# Initialize the learning rate scheduler (e.g., linear scheduler).
lr_scheduler = get_scheduler(
    name=args["lr_scheduler_type"],  # The type of learning rate scheduler (e.g., 'linear')
    optimizer=optimizer,  # The optimizer that the scheduler will adjust
    num_warmup_steps=args["num_warmup_steps"] * accelerator.num_processes,  # Number of warmup steps, scaled by the number of processes
    num_training_steps=args["max_train_steps"]
    if overrode_max_train_steps
    else args["max_train_steps"] * accelerator.num_processes,  # Total number of training steps, scaled if needed
)

# Prepare everything for distributed training using the Accelerator.
# This prepares the model, optimizer, dataloaders, and scheduler for efficient training across multiple devices.
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

# Recalculate the total number of steps for training after preparing the DataLoader.
# This is important because the DataLoader size might have changed after `accelerator.prepare`.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])

# If we overrode `max_train_steps`, we also recalculate the number of epochs based on the new total steps.
if overrode_max_train_steps:
    args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch
    # Recalculate the number of epochs required to meet the new total training steps
    args["num_train_epochs"] = math.ceil(args["max_train_steps"] / num_update_steps_per_epoch)

# Set up the checkpointing frequency (how often to save model checkpoints).
checkpointing_steps = args["checkpointing_steps"]
if checkpointing_steps is not None and checkpointing_steps.isdigit():
    checkpointing_steps = int(checkpointing_steps)

# Initialize tracking if enabled.
# The tracking system (e.g., TensorBoard, Weights & Biases) will log the configuration and metrics.
if args["with_tracking"]:
    # Store the experiment configuration and initialize the trackers.
    experiment_config = args
    # TensorBoard cannot log Enums directly, so we use the raw value for the scheduler type.
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
    
    # Initialize the tracker to log the experiment with the given configuration.
    accelerator.init_trackers("clm_no_trainer", experiment_config)



# Calculate the total batch size considering devices and gradient accumulation
total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

# Log training setup details
logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
logger.info(f"  Max optimization steps = {max_train_steps}")

# Initialize progress bar for tracking steps
completed_steps = args.get("completed_steps", 0)  # Default to 0 if not provided
starting_epoch = args.get("starting_epoch", 0)  # Default to 0 if not provided
updated_max_train_steps = max_train_steps - completed_steps
progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process) # Disable progress bar for non-main processes


# Check if we are resuming from a checkpoint
if resume_from_checkpoint:
    # Check if the checkpoint path is provided and valid
    if resume_from_checkpoint is not None or resume_from_checkpoint != "":
        checkpoint_path = resume_from_checkpoint
        print("Checkpoint path provided!")
        path = os.path.basename(resume_from_checkpoint)
    else:
        # If no path is provided, choose the most recent checkpoint
        dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        dirs.sort(key=os.path.getctime)  # Sort directories by modification time
        path = dirs[-1]  # Choose the most recent directory
        checkpoint_path = path
        path = os.path.basename(checkpoint_path)

    # Log that we are resuming from a checkpoint
    accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path)  # Load the checkpoint's state
    training_difference = os.path.splitext(path)[0]  # Extract the training checkpoint identifier
    
    logger.info(f"Starting epoch = {starting_epoch}, resume step = {completed_steps}")

    # If the checkpoint was saved at an epoch, handle accordingly
    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1  # Increment the epoch
        resume_step = None  # No specific step, resume by epoch
        completed_steps = starting_epoch * num_update_steps_per_epoch  # Calculate completed steps
    else:
        # If the checkpoint was saved at a step, calculate based on that
        resume_step = completed_steps  # Example step (should be extracted from the checkpoint filename)
        # starting_epoch = resume_step // len(train_dataloader)  # Calculate epoch from step
        # completed_steps = resume_step // gradient_accumulation_steps  # Adjust for gradient accumulation
        resume_step -= starting_epoch * len(train_dataloader)  # Adjust the remaining steps in the current epoch

# Main training loop, running for each epoch
for epoch in range(starting_epoch, num_train_epochs):
    model.train()
    
    # Track loss if necessary
    if args["with_tracking"]:
        total_loss = 0

    # If resuming from a checkpoint, skip the first few batches based on resume_step
    if args["resume_from_checkpoint"] and epoch == starting_epoch and resume_step is not None:
        logger.info("Resuming training from the specified step")
        active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)  # Skip batches up to resume_step
    else:
        active_dataloader = train_dataloader  # Regular dataloader if not resuming

    total_steps = num_train_epochs * len(active_dataloader)  # Total steps for the epoch
    logger.info(f"Total expected steps: {total_steps}")

    # Loop through batches in the dataloader
    for step, batch in enumerate(active_dataloader):
        with accelerator.accumulate(model):  # Handle gradient accumulation
            outputs = model(**batch)  # Forward pass
            loss = outputs.loss  # Calculate loss

            # Track the loss if required
            if with_tracking:
                total_loss += loss.detach().float()

            accelerator.backward(loss)  # Backpropagation
            optimizer.step()  # Optimization step
            lr_scheduler.step()  # Update learning rate
            optimizer.zero_grad()  # Clear gradients

            logger.info(f"  Loss = {loss}, epoch = {epoch}, step = {step + resume_step}")  # Log the loss

        # Update progress bar if gradients are synchronized
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
        
        # Save a checkpoint every specified number of steps
        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                output_dir = f"step_{completed_steps}"  # Define checkpoint folder based on steps
                if output_dir is not None:
                    output_dir = os.path.join(output_dir, output_dir)  # Include the output directory path
                accelerator.save_state(output_dir)  # Save model state

        # Stop training if the max steps are reached
        if completed_steps >= updated_max_train_steps:
            break

        # ALSO saves at the end of each epoch!
        output_dir = f"epoch_{epoch}"
        if output_dir is not None:
            output_dir = os.path.join(output_dir, output_dir)
            accelerator.save_state(output_dir)
