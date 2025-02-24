import numpy as np
from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import ast
import pandas as pd
from handcoded_tokenizer import STLTokenizer
from configuration import STLConfig
from modeling_stldec import STLForCausalLM
from encoder import STLEncoder

from transformers import AutoConfig, AutoModelForCausalLM 


##################################################################

model_path = "balanced/step_500"
optimizer_path = "balanced/step_500/optimizer.bin"
scheduler_path = "balanced/step_500/scheduler.bin"

##################################################################

AutoConfig.register("STLdec", STLConfig)
AutoModelForCausalLM.register(STLConfig, STLForCausalLM)

config = STLConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(model_path, config = config).to(device)  # Sposta il modello sulla device
tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')
encoder = STLEncoder(embed_dim=1024, anchor_filename='anchor_set_1024_dim.pickle')

accelerator = Accelerator()

optimizer = torch.load(optimizer_path)
scheduler = torch.load(scheduler_path)
optimizer = accelerator.prepare(optimizer)
scheduler = accelerator.prepare(scheduler)

eval_df = pd.read_csv("datasets/test_balanced_validation_set.csv")

##################################################################

formulae_dataset = []

for idx in range(len(eval_df)):
    embedding = eval(eval_df["Embedding"][idx])
    embedding = torch.tensor(embedding, dtype=torch.long).to(device)
    tok_formula = eval(eval_df["Encoded_Formula"][idx])


    with torch.no_grad():
        generated_ids = model.generate(
            encoder_hidden_states=embedding,  # Usa gli ID tokenizzati
            pad_token_id=model.config.pad_token_id,  # ID del token di padding, se presente
            bos_token_id=model.config.bos_token_id,
            max_new_tokens = 500
        )

    generated_text = tokenizer.decode(generated_ids[0][2:-2].tolist())
    gold_formula = eval_df["Formula"][idx]

    formulae_dataset.append({
        "Gold Formula": gold_formula,
        "Generated Formula": generated_text
    })
    
eval_df = pd.DataFrame(formulae_dataset)
eval_df.to_csv('balanced_500_supertest.csv')