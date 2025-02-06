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
import torch.nn.functional as F

eval_df = pd.read_csv('predicted_gold_formulae.csv')
encoder = STLEncoder(embed_dim=1024, anchor_filename='anchor_set_1024_dim.pickle')

gold_embeddings = encoder.compute_embeddings(eval_df["Gold Formula"])
generated_embeddings = encoder.compute_embeddings(eval_df["Generated Formula"])

eval_df['Embedding Gold Formula'] = gold_embeddings.tolist()
eval_df['Embedding Generated Formula'] = generated_embeddings.tolist()

euclidean_distance = []
cosine_distance = []

for idx in range(len(eval_df)):
    gold = torch.tensor(eval_df["Embedding Gold Formula"][idx])
    generated = torch.tensor(eval_df["Embedding Generated Formula"][idx])

    euclidean_distance.append(torch.dist(gold, generated))
#    cosine_distance.append(1-F.cosine_similarity(gold, generated))

print(f"Mean euclidean distance: {np.mean(euclidean_distance)}")
print("Mean cosine distance: {np.mean(cosine_distance)}")


eval_df.to_csv('updated_predicted_gold_formulae.csv')


