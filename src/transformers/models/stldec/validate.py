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

model_path = "../../../../../../../../../leonardo_scratch/fast/IscrC_IRA-LLMs/balanced_@/step_12000"
optimizer_path = "../../../../../../../../../leonardo_scratch/fast/IscrC_IRA-LLMs/balanced_@/step_12000/optimizer.bin"
scheduler_path = "../../../../../../../../../leonardo_scratch/fast/IscrC_IRA-LLMs/balanced_@/step_12000/scheduler.bin"

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

eval_df = pd.read_pickle("datasets/new_balanced_validation_set.pkl")

##################################################################

eval_df = eval_df.head(200)

formulae_dataset = []

for idx in range(len(eval_df)):
    embedding = eval_df["Embedding"][idx]
    encoder_hidden_states = torch.tensor(embedding, dtype = torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(
            encoder_hidden_states=encoder_hidden_states,  # Usa gli ID tokenizzati
            pad_token_id=model.config.pad_token_id,  # ID del token di padding, se presente
            bos_token_id=model.config.bos_token_id,
            forced_eos_token_id = config.forced_eos_token_id,
            max_new_tokens = 500
        )
    # print(generated_ids[0])
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    # print(generated_text)
    generated_text = generated_text[3:-2]
    # print(generated_text)
    formulae_dataset.append(generated_text)

encoder = STLEncoder(embed_dim=1024, anchor_filename='anchor_set_1024_dim.pickle')

generated_embeddings = encoder.compute_embeddings(formulae_dataset)
gold_embeddings = encoder.compute_embeddings(eval_df["Formula"])




# eval_df.head()

# gold_embeddings = encoder.compute_embeddings(eval_df["Gold Formula"])
# generated_embeddings = encoder.compute_embeddings(eval_df["Generated Formula"])

eval_df['Embedding Gold Formula'] = gold_embeddings.tolist()
eval_df['Embedding Generated Formula'] = generated_embeddings.tolist()

euclidean_distance = []

for idx in range(len(eval_df)):
     gold = torch.tensor(eval_df["Embedding Gold Formula"][idx])
     generated = torch.tensor(eval_df["Embedding Generated Formula"][idx])
     euclidean_distance.append(torch.dist(gold, generated))

print(f"Mean euclidean distance: {np.mean(euclidean_distance)}")

# eval_df.to_csv('balanced/step_7000_formulae.csv')

