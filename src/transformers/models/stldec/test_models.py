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

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("STLdec", STLConfig)
AutoModelForCausalLM.register(STLConfig, STLForCausalLM)

config = STLConfig()

model_path = "tf_output_test_16batch/step_5400"
# Carica il modello e spostalo sulla device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(model_path, config = config).to(device)  # Sposta il modello sulla device
tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')

# Inizializza l'Accelerator
accelerator = Accelerator()

# Definisci i percorsi
optimizer_path = "tf_output_test_16batch/step_5400/optimizer.bin"
scheduler_path = "tf_output_test_16batch/step_5400/scheduler.bin"

# Carica lo stato dell'ottimizzatore e dello scheduler, se necessario
# Questi passi dipendono dalla libreria che stai usando, di seguito un esempio generico
optimizer = torch.load(optimizer_path)
scheduler = torch.load(scheduler_path)

optimizer = accelerator.prepare(optimizer)
scheduler = accelerator.prepare(scheduler)

test_df = pd.read_csv("datasets/test_set.csv")

# Estrai un esempio a caso dal test_df
example_idx = 0  # Puoi scegliere un altro indice o usare random per un campione casuale
# encoded_formula = ast.literal_eval(test_df['Embeddings'][example_idx].strip())  # Decodifica la lista di token

# Converti il vettore di token in un tensor
# encoder_hidden_states = torch.tensor(encoded_formula, dtype=torch.long).unsqueeze(0).to(model.device)  # Aggiungi la dimensione b

embedding_str = test_df["Embedding"][example_idx]
tensor_str_cleaned = embedding_str[len("tensor("):-1]
numbers_str = tensor_str_cleaned.strip("[]")
numbers = list(map(float, numbers_str.split(", ")))
encoder_hidden_states = torch.tensor(numbers).to(device)
encoder_hidden_states = torch.tensor(encoder_hidden_states, device=device).unsqueeze(0).unsqueeze(0)


# Genera la sequenza autoregressiva
with torch.no_grad():
    generated_ids = model.generate(
        encoder_hidden_states=encoder_hidden_states,  # Usa gli ID tokenizzati
        pad_token_id=model.config.pad_token_id,  # ID del token di padding, se presente
        bos_token_id=model.config.bos_token_id,
        max_new_tokens = 500
    )

print("Generated IDs:")
print(generated_ids)

# Decodifica e visualizza il testo generato
generated_text = tokenizer.decode(generated_ids[0].tolist())
print("Generated Text:")
print(generated_text)

