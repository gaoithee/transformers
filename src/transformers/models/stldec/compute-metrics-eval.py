import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

df = pd.read_csv('updated_predicted_gold_formulae.csv')
euclidean_distance = []
cosine_distance = []

for idx in range(len(df)):
    gold = torch.tensor(df["Gold Formula Embedding"][idx])
    generated = torch.tensor(df["Generated Formula Embedding"][idx])

    euclidean_distance.append(torch.dist(gold, generated))
    cosine_distance.append(1-F.cosine_similarity(gold, generated))

print("Mean euclidean distance: {np.mean(euclidean_distance)}")
print("Mean cosine distance: {np.mean(cosine_distance)}")
