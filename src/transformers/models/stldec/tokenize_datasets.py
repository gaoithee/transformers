import pandas as pd
from handcoded_tokenizer import STLTokenizer

tokenizer = STLTokenizer('tokenizer_files/tokenizer.json', model_max_length = 500)
pad_token_id = 1
df = pd.read_pickle('datasets/easysk_train_set.pkl')

encodings = []

for i in range(len(df)):
    encoded = tokenizer.encode(df['Formula'][i])
    encodings.append(tokenizer.postpad_sequence(encoded, pad_token_id))

df['Encoded_Formula'] = encodings

df.to_pickle('datasets/easysk_train_set.pkl')

