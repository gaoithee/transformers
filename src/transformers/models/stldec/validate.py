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

model_path = "tf_output_test_16batch/step_32500"
optimizer_path = "tf_output_test_16batch/step_32500/optimizer.bin"
scheduler_path = "tf_output_test_16batch/step_32500/scheduler.bin"

# model_path = "step_20000/step_20000"
# optimizer_path = "step_20000/step_20000/optimizer.bin"
# scheduler_path = "step_20000/step_20000/scheduler.bin"

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

eval_df = pd.read_csv("datasets/validation_set.csv")

##################################################################

subset_eval_df = eval_df[:200]
formulae_dataset = []

for idx in range(len(subset_eval_df)):
    embedding_str = subset_eval_df["Embedding"][idx]
    tensor_str_cleaned = embedding_str[len("tensor("):-1]
    numbers_str = tensor_str_cleaned.strip("[]")
    numbers = list(map(float, numbers_str.split(", ")))
    encoder_hidden_states = torch.tensor(numbers).to(device)
    encoder_hidden_states = torch.tensor(encoder_hidden_states, device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(
            encoder_hidden_states=encoder_hidden_states,  # Usa gli ID tokenizzati
            pad_token_id=model.config.pad_token_id,  # ID del token di padding, se presente
            bos_token_id=model.config.bos_token_id,
            max_new_tokens = 500
        )

    generated_text = tokenizer.decode(generated_ids[0][2:-2].tolist())
    gold_formula = subset_eval_df["Formula"][idx]

    formulae_dataset.append({
        "Gold Formula": gold_formula,
        "Generated Formula": generated_text
    })
    
eval_df = pd.DataFrame(formulae_dataset)
eval_df.to_csv('step_20000.csv')
eval_df = pd.read_csv('step_20000.csv')
encoder = STLEncoder(embed_dim=1024, anchor_filename='anchor_set_1024_dim.pickle')

gold_embeddings = encoder.compute_embeddings(eval_df["Gold Formula"].tolist())
generated_embeddings = encoder.compute_embeddings(eval_df["Generated Formula"].tolist())

eval_df['Embedding Gold Formula'] = gold_embeddings.tolist()
eval_df['Embedding Generated Formula'] = generated_embeddings.tolist()

euclidean_distance = []

for idx in range(len(eval_df)):
    gold = torch.tensor(eval_df["Embedding Gold Formula"][idx])
    generated = torch.tensor(eval_df["Embedding Generated Formula"][idx])
    euclidean_distance.append(torch.dist(gold, generated))

print(f"Mean euclidean distance: {np.mean(euclidean_distance)}")

eval_df.to_csv('step_32500_formulae.csv')

