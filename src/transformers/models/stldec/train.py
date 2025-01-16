from handcoded_tokenizer import STLTokenizer
from configuration import STLConfig
from modeling_stldec import STLForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM

import argparse
import json
import logging
import math
import os
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
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = get_logger(__name__)

# Extend `AutoClasses` to support the custom model
AutoConfig.register("STLdec", STLConfig)
AutoModelForCausalLM.register(STLConfig, STLForCausalLM)

# Initialize the model with random weights and the desired architecture
config = STLConfig()
model = AutoModelForCausalLM.from_config(config)
tokenizer = STLTokenizer('tokenizer.json')

args = {
    'dataset_name': None,  # or a custom dataset path
    'train_file': 'train_set.csv',
    'validation_file': 'validation_set.csv',
    'output_dir': './output_test',
    'model_name_or_path': 'STLForCausalLM',
    'tokenizer_name': 'STLTokenizer',
    'block_size': 500,
    'batch_size': 16,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 3,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'num_warmup_steps': 0,
    'max_train_steps': None,
    'seed': 42,
    'with_tracking': True,
    'hub_model_id': None,
    'push_to_hub': False,
    'trust_remote_code': False,
    'overwrite_cache': False,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'checkpointing_steps': 'epoch',  
    'resume_from_checkpoint': None,
    'lr_scheduler_type': 'linear',  
    'num_warmup_steps': 100,   
    'max_train_steps': 1000, 
    'lr': 0.01                  
}


# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
# If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
# in the environment
accelerator_log_kwargs = {}

if args["with_tracking"]:
    # accelerator_log_kwargs["log_with"] = args["report_to"]
    accelerator_log_kwargs["project_dir"] = args["output_dir"]

accelerator = Accelerator(gradient_accumulation_steps=args["gradient_accumulation_steps"], **accelerator_log_kwargs)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if accelerator.is_local_main_process:
  datasets.utils.logging.set_verbosity_warning()
  transformers.utils.logging.set_verbosity_info()
else:
  datasets.utils.logging.set_verbosity_error()
  transformers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args["seed"] is not None:
    set_seed(args["seed"])

# Handle the repository creation
if accelerator.is_main_process:
  if args["push_to_hub"]:
    # Retrieve of infer repo_name
    repo_name = args["hub_model_id"]
    if repo_name is None:
      repo_name = Path(args["output_dir"]).absolute().name
      # Create repo and retrieve repo_id
      api = HfApi()
      repo_id = api.create_repo(repo_name, exist_ok=True, token=args["hub_token"]).repo_id
      with open(os.path.join(args["output_dir"], ".gitignore"), "w+") as gitignore:
        if "step_*" not in gitignore:
          gitignore.write("step_*\n")
        if "epoch_*" not in gitignore:
          gitignore.write("epoch_*\n")
  elif args["output_dir"] is not None:
    os.makedirs(args["output_dir"], exist_ok=True)

accelerator.wait_for_everyone()


if args["dataset_name"] is not None:
  # Downloading and loading a dataset from the hub.
  raw_datasets = load_dataset(
      args.dataset_name, args["dataset_config_name"], trust_remote_code=args["trust_remote_code"]
      )
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
    data_files = {}
    dataset_args = {}
    if args["train_file"] is not None:
      data_files["train"] = args["train_file"]
      extension = args["train_file"].split(".")[-1]
    if args["validation_file"] is not None:
      data_files["validation"] = args["validation_file"]
      extension = args["validation_file"].split(".")[-1]
    if extension == "txt":
      extension = "text"

    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)


# Create a `CustomDataset` class to format properly the input data wrt the 
# `input_ids`, `labels` and `attention_mask` attributes
class CustomDataset(Dataset):
    def __init__(self, df, device='cpu'):
        self.df = df
        self.device = device  

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Start from `Encoded_Formula`
        encoded_formula = self.df['Encoded_Formula'][idx]
        encoded_formula = ast.literal_eval(encoded_formula.strip())
        
        input_ids = encoded_formula[:-1]  # Tutti tranne l'ultimo
        labels = encoded_formula[1:]     # Tutti tranne il primo

        attention_mask = [0 if token == '1' else 1 for token in input_ids]
        # if 1 (i.e. tokenized `pad`), then neglect that token

        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = CustomDataset(raw_datasets['train'], device=device)
eval_dataset = CustomDataset(raw_datasets['validation'], device=device)


# DataLoaders creation:
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args["batch_size"]
)

eval_dataloader = DataLoader(
    eval_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args["batch_size"]
)

# Optimizer
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args["weight_decay"],
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], betas=(0.9, 0.99))


# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])
if args["max_train_steps"] is None:
  args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch
  overrode_max_train_steps = True
lr_scheduler = get_scheduler(
    name=args["lr_scheduler_type"],
    optimizer=optimizer,
    num_warmup_steps=args["num_warmup_steps"] * accelerator.num_processes,
    num_training_steps=args["max_train_steps"]
    if overrode_max_train_steps
    else args["max_train_steps"] * accelerator.num_processes,
)

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])
if overrode_max_train_steps:
  args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch
  # Afterwards we recalculate our number of training epochs
  args["num_train_epochs"] = math.ceil(args["max_train_steps"] / num_update_steps_per_epoch)

# Figure out how many steps we should save the Accelerator states
checkpointing_steps = args["checkpointing_steps"]
if checkpointing_steps is not None and checkpointing_steps.isdigit():
  checkpointing_steps = int(checkpointing_steps)

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if args["with_tracking"]:
  experiment_config = args
  # TensorBoard cannot log Enums, need the raw value
  experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
  accelerator.init_trackers("clm_no_trainer", experiment_config)


# Train!
total_batch_size = args["per_device_train_batch_size"] * accelerator.num_processes * args["gradient_accumulation_steps"]

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")

num_train_epochs = args["num_train_epochs"]
per_device_train_batch_size = args["per_device_train_batch_size"]
gradient_acc_steps = args["gradient_accumulation_steps"]
max_train_steps = args["max_train_steps"]

logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {gradient_acc_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")

# Only show the progress bar once on each machine.
progress_bar = tqdm(range(args["max_train_steps"]), disable=not accelerator.is_local_main_process) # i 1000 sono questi!
completed_steps = 0
starting_epoch = 0
# Potentially load in the weights and states from a previous save
if args["resume_from_checkpoint"]:
    print("questo non dovrebbe succedere")
    if args["resume_from_checkpoint"] is not None or args["resume_from_checkpoint"] != "":
        checkpoint_path = args["resume_from_checkpoint"]
        path = os.path.basename(args["resume_from_checkpoint"])
    else:
      # Get the most recent checkpoint
      dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
      dirs.sort(key=os.path.getctime)
      path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
      checkpoint_path = path
      path = os.path.basename(checkpoint_path)

    accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path)
    # Extract `epoch_{i}` or `step_{i}`
    training_difference = os.path.splitext(path)[0]

    if "epoch" in training_difference:
      starting_epoch = int(training_difference.replace("epoch_", "")) + 1
      resume_step = None
      completed_steps = starting_epoch * num_update_steps_per_epoch
    
    else:
      # need to multiply `gradient_accumulation_steps` to reflect real steps
      resume_step = int(training_difference.replace("step_", "")) * args["gradient_accumulation_steps"]
      starting_epoch = resume_step // len(train_dataloader)
      completed_steps = resume_step // args["gradient_accumulation_steps"]
      resume_step -= starting_epoch * len(train_dataloader)

for epoch in range(starting_epoch, args["num_train_epochs"]):
  model.train()
  if args["with_tracking"]:
    total_loss = 0
  if args["resume_from_checkpoint"] and epoch == starting_epoch and resume_step is not None:
    active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
  else:
    active_dataloader = train_dataloader

  for step, batch in enumerate(active_dataloader):
    with accelerator.accumulate(model):
      outputs = model(**batch)
      loss = outputs.loss
      # print(loss)
      if args["with_tracking"]:
        total_loss += loss.detach().float()
      accelerator.backward(loss)
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
        
      # Checks if the accelerator has performed an optimization step behind the scenes
    if accelerator.sync_gradients:
        progress_bar.update(1)
        completed_steps += 1
    
    if isinstance(checkpointing_steps, int):
        if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
            output_dir = f"step_{completed_steps}"
            if args["output_dir"] is not None:
                output_dir = os.path.join(args["output_dir"], output_dir)
                accelerator.save_state(output_dir)
    if completed_steps >= args["max_train_steps"]:
        break


model.eval()
losses = []
for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        outputs = model(**batch)   
    loss = outputs.loss
    losses.append(accelerator.gather_for_metrics(loss.repeat(args["per_device_eval_batch_size"])))

losses = torch.cat(losses)
try:
    eval_loss = torch.mean(losses)
    perplexity = math.exp(eval_loss)
except OverflowError:
    perplexity = float("inf")


logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

if args["with_tracking"]:
    accelerator.log(
        {
            "perplexity": perplexity,
            "eval_loss": eval_loss,
            "train_loss": total_loss.item() / len(train_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        },
        step=completed_steps,
    )

if args["checkpointing_steps"] == "epoch":
    output_dir = f"epoch_{epoch}"
    if args["output_dir"] is not None:
        output_dir = os.path.join(args["output_dir"], output_dir)
        accelerator.save_state(output_dir)














