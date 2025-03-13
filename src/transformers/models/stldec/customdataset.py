import pandas as pd
import ast
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, device='cpu'):
        """
        Initializes the dataset by storing the DataFrame and setting the device.
        
        Args:
        - df: A pandas DataFrame containing the data (e.g., `Encoded_Formula`, `Embedding`).
        - device: The device ('cpu' or 'cuda') where the tensors will be moved for processing.
        """
        self.df = df
        self.device = device
        transformed_data = []

        for idx in range(len(self.df)):

            # Extract the encoded formula (tokenized input sequence) from the DataFrame
            encoded_formula = self.df['Encoded_Formula'][idx]
            # Convert the string representation of a list back to a Python list using ast.literal_eval
            encoded_formula = ast.literal_eval(encoded_formula.strip())

            # Extract the precomputed formula embedding (hidden states) from the DataFrame
            formula_embedding = self.df['Embedding'][idx]

            # Clean the string and convert it back to a tensor
            formula_embedding = formula_embedding.replace("tensor(", "").rstrip(")")
            formula_embedding = eval(formula_embedding)

            # Define the input_ids by excluding the last token (shifted tokens for prediction)
            input_ids = encoded_formula[:-1]  # All tokens except the last
            # Define the labels by excluding the first token (shifted tokens for teacher forcing)
            labels = encoded_formula[1:]     # All tokens except the first

            # Create the attention mask to indicate which tokens should be attended to.
            # Tokens equal to '1' (typically padding tokens) will be masked (set to 0), 
            # and the rest will be visible (set to 1).
            attention_mask = [0 if token == 1 else 1 for token in input_ids]

            # Convert `input_ids`, `labels`, and `attention_mask` to tensors and move them to the desired device (e.g., GPU or CPU)
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

            # Convert the formula embedding (list of hidden states) to a tensor and move it to the device
            encoder_hidden_states = torch.tensor(formula_embedding, dtype=torch.float32).to(self.device)

            # Store the transformed data in a dictionary
            transformed_data.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'encoder_hidden_states': encoder_hidden_states
            })

        # Convert the transformed data into a DataFrame (now with tensors)
        self.df = pd.DataFrame(transformed_data)

    def __len__(self):
        """
        Returns the length of the dataset, i.e., the number of examples in the DataFrame.
        
        Returns:
        - Length of the DataFrame (number of samples).
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a specific example from the dataset, processes it, and formats it 
        into the required structure for the model (e.g., `input_ids`, `labels`, `attention_mask`).
        
        Args:
        - idx: Index of the example to retrieve.
        
        Returns:
        - A dictionary containing the formatted input data, including:
            - `input_ids`: The tokenized input sequence (excluding the last token).
            - `labels`: The tokenized target sequence (excluding the first token).
            - `attention_mask`: A mask indicating which tokens should be attended to.
            - `encoder_hidden_states`: Embedding for each formula (precomputed, used as hidden states).
        """
        
        # Return the formatted data as a dictionary, which the model can use directly for training or evaluation
        return {
            'input_ids': self.df['input_ids'][idx],
            'labels': self.df['labels'][idx],
            'attention_mask': self.df['attention_mask'][idx],
            'encoder_hidden_states': self.df['encoder_hidden_states'][idx]
        }
