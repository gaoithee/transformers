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

eval_df = pd.read_pickle("datasets/depth_8_formulae.pkl")
eval_df = eval_df.head(500)
gold_formulae = eval_df['Formula'] 

steps = ['old_datasets/step_24000']

formulae_dataset = []

for i in steps:
    model_path = f"../../../../../../../../../leonardo_scratch/fast/IscrC_IRA-LLMs/{i}"
    optimizer_path = f"../../../../../../../../../leonardo_scratch/fast/IscrC_IRA-LLMs/{i}/optimizer.bin"
    scheduler_path = f"../../../../../../../../../leonardo_scratch/fast/IscrC_IRA-LLMs/{i}/scheduler.bin"

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
    
    
##################################################################
    print(i)
    generated_formulae = []

    for idx in range(len(eval_df)):
        embedding = eval(eval_df["Embedding"][idx])
        encoder_hidden_states = torch.tensor(embedding, dtype=torch.float32).to(device)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            generated_ids = model.generate(
                encoder_hidden_states=encoder_hidden_states,  # Usa gli ID tokenizzati
                pad_token_id=model.config.pad_token_id,  # ID del token di padding, se presente
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.forced_eos_token_id,
                max_new_tokens = 500
            )

        generated_text = tokenizer.decode(generated_ids[0].tolist())
        generated_text = generated_text[3:-2]
        generated_formulae.append(generated_text)
        # print(generated_text)
    # print(i)
    formulae_dataset.append(generated_formulae)

eval_df = pd.DataFrame(formulae_dataset).transpose()

eval_df['gold formula'] = gold_formulae
# print('ci sono in teoria')
eval_df.to_csv('depth8/old1024.csv', index=False)
