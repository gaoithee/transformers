import pandas as pd
from handcoded_tokenizer import STLTokenizer

tokenizer = STLTokenizer('tokenizer_files/tokenizer.json', model_max_length = 500)
df = pd.read_csv('datasets/balanced_validation_set.csv')

encodings = []

for i in range(len(df)):
    encoded = tokenizer.encode(df['Formula'][i])
    encodings.append(tokenizer.postpad_sequence(encoded, pad_token_id = 1))

df['Encoded_Formula'] = encodings

df.to_csv('datasets/test_balanced_validation_set.csv')

