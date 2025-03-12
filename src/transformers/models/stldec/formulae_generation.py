import os
import numpy as np
import torch
from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import ast
import pandas as pd
from handcoded_tokenizer import STLTokenizer
from configuration import STLConfig
from modeling_stldec import STLForCausalLM
from encoder import STLEncoder
from transformers import AutoConfig, AutoModelForCausalLM 

# Percorsi di base
model_dir = "balanced/"
eval_df = pd.read_csv("datasets/test_balanced_validation_set.csv")

# Inizializzazione di STL
AutoConfig.register("STLdec", STLConfig)
AutoModelForCausalLM.register(STLConfig, STLForCausalLM)

config = STLConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer ed encoder
tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')
encoder = STLEncoder(embed_dim=1024, anchor_filename='anchor_set_1024_dim.pickle')

accelerator = Accelerator()

# Trova tutti i checkpoint step_*
step_folders = [f for f in os.listdir(model_dir) if f.startswith('step_')]

# Prepara il dataset per la generazione
formulae_dataset = []

for idx in range(len(eval_df)):
    embedding = eval(eval_df["Embedding"][idx])
    embedding = torch.tensor(embedding, dtype=torch.long).to(device)
    tok_formula = eval(eval_df["Encoded_Formula"][idx])
    gold_formula = eval_df["Formula"][idx]
    
    # Genera la lista di "generated_text" per ogni step_*
    generated_texts = {"Gold Formula": gold_formula}

    for step_folder in step_folders:
        # Carica il modello, l'optimizer e lo scheduler per ogni step_*
        model_path = os.path.join(model_dir, step_folder)
        optimizer_path = os.path.join(model_path, 'optimizer.bin')
        scheduler_path = os.path.join(model_path, 'scheduler.bin')
        
        # Carica il modello
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config).to(device)
        
        # Carica l'ottimizzatore e lo scheduler
        optimizer = torch.load(optimizer_path)
        scheduler = torch.load(scheduler_path)
        optimizer = accelerator.prepare(optimizer)
        scheduler = accelerator.prepare(scheduler)

        # Genera il testo
        with torch.no_grad():
            generated_ids = model.generate(
                encoder_hidden_states=embedding,  
                pad_token_id=model.config.pad_token_id,  
                bos_token_id=model.config.bos_token_id,
                max_new_tokens=500
            )

        generated_text = tokenizer.decode(generated_ids[0][2:-2].tolist())
        generated_texts[step_folder] = generated_text

    formulae_dataset.append(generated_texts)

# Crea il dataframe finale
final_df = pd.DataFrame(formulae_dataset)

# Salva il dataframe come CSV
final_df.to_csv('balanced_all_steps.csv', index=False)

