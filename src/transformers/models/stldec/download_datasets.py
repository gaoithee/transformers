import pandas as pd

splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
test_df = pd.read_csv("hf://datasets/saracandu/stldecoding/" + splits["test"])
validation_df = pd.read_csv("hf://datasets/saracandu/stldecoding/" + splits["validation"])
train_df = pd.read_csv("hf://datasets/saracandu/stldecoding/" + splits["train"])

test_df.to_csv('datasets/test_set.csv')
train_df.to_csv('datasets/train_set.csv')
validation_df.to_csv('dataset/validation_set.csv')
