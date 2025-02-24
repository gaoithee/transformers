import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def load_json(path: str) -> Union[Dict, List]:
    """
    Load a JSON file from the given path.

    Args:
        path (str): The path to the JSON file to be loaded.
    
    Returns:
        Union[Dict, List]: The parsed content of the JSON file, which could be a dictionary or a list.
    """
    with open(path, "r") as f:
        return json.load(f)


class STLTokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer class that extends `PreTrainedTokenizer` to handle a specific vocabulary and tokenization process.

    This tokenizer can load a vocabulary from a JSON file, tokenize text, convert tokens to IDs, 
    and handle padding and special tokens.
    """

    def __init__(self, vocab_path: str, unk_token: str = "unk", pad_token: str = "pad", 
                 bos_token: str = "/s", eos_token: str = "s", model_max_length = 512):
        """
        Initializes the STLTokenizer with a given vocabulary and special tokens.

        Args:
            vocab_path (str): The path to the JSON file containing the vocabulary.
            unk_token (str, optional): The token used for unknown words. Defaults to "unk".
            pad_token (str, optional): The token used for padding. Defaults to "pad".
            bos_token (str, optional): The token used for the beginning of a sequence. Defaults to "/s".
            eos_token (str, optional): The token used for the end of a sequence. Defaults to "s".
        """
        self.vocab = load_json(vocab_path)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.model_max_length = model_max_length
        self.id_to_token = {v: k for k, v in self.vocab.items()}  # Reverse mapping

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        return len(self.vocab)

    def prepad_sequence(self, sequence, undo = False):
        """
        Replaces spaces in the input sequence with a specified padding token.

        Args:
            sequence (str): The input sequence.
            undo (bool): If True, replace the padding token with spaces. Defaults to False, which pads the spaces.

        Returns:
            str: The preprocessed sequence with spaces or padding tokens replaced.
        """
        if undo:
            return sequence.replace(f'{self.pad_token}', ' ')
        else:
            return sequence.replace(' ', f'{self.pad_token}')

    def add_bos_eos(self, sequence: str) -> str:
        """
        Aggiunge i token BOS all'inizio e EOS alla fine della sequenza.

        Args:
            sequence (str): La sequenza di input.

        Returns:
            str: La sequenza con i token BOS ed EOS.
        """
        return f'{self.bos_token} {sequence} {self.eos_token}'

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into a list of tokens.

        The method preprocesses the input text by replacing spaces with padding tokens and then tries to 
        find the longest possible match for each substring in the vocabulary.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[str]: A list of tokens representing the tokenized text.
        """
        text = self.add_bos_eos(text)
        text = self.prepad_sequence(text)
        
        tokens = []
        i = 0
        while i < len(text):
            best_match = None
            for j in range(len(text), i, -1):  # Try matching substrings of decreasing length
                subtoken = text[i:j]
                if subtoken in self.vocab:
                    best_match = subtoken
                    break
            if best_match:
                tokens.append(best_match)
                i += len(best_match)
            else:
                tokens.append(self.unk_token)
                i += 1
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Converts a list of tokens into a list of token IDs.

        Args:
            tokens (List[str]): A list of tokens to be converted into IDs.

        Returns:
            List[int]: A list of corresponding token IDs.
        """
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Converts a list of token IDs into a list of tokens.

        Args:
            ids (List[int]): A list of token IDs to be converted into tokens.

        Returns:
            List[str]: A list of corresponding tokens.
        """
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def encode(self, sequence: str) -> List[int]:
        """
        Encodes a string sequence into a list of token IDs.
    
        This method tokenizes the input sequence using the `tokenize` method, 
        and then converts the resulting tokens into their corresponding token IDs 
        using the `convert_tokens_to_ids` method.
    
        Args:
            sequence (str): The input sequence (text) to be encoded.
    
        Returns:
            List[int]: A list of token IDs corresponding to the input sequence.
        """
        splitted_sequence = self.tokenize(sequence)
        return self.convert_tokens_to_ids(splitted_sequence)

    def postpad_sequence(self, sequence, pad_token_id):
       """
       Fills the sequence up to max_length padding elements 
       """        
       num_extra_elements = self.model_max_length - len(sequence) -1
       if num_extra_elements > 0:
           sequence.extend([pad_token_id] * num_extra_elements)
       return sequence

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs into a string of text.

        The method converts the IDs to tokens and joins them to form a string. 
        It also restores the original spaces or padding tokens if `undo` is True.

        Args:
            token_ids (List[int]): A list of token IDs to be decoded.
            skip_special_tokens (bool, optional): Whether to skip special tokens during decoding. Defaults to False.

        Returns:
            str: The decoded string.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        decoded = "".join(tokens)
        return self.prepad_sequence(decoded, undo=True)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary to a file. 
        Useful only when the vocabulary has to be retrieved and is not given 
        (thus this is not the case: here to further improvements with sentencepiece).

        This method saves the vocabulary to a JSON file in the specified directory. 

        Args:
            save_directory (str): The directory where the vocabulary file will be saved.
            filename_prefix (Optional[str]): An optional prefix for the filename.

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.
        """
        vocab_file = f"{save_directory}/{filename_prefix + '-' if filename_prefix else ''}vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        return (vocab_file,)

    def get_vocab(self) -> dict:
        """
        Retrieves the vocabulary used by the tokenizer.

        Returns:
            dict: The vocabulary as a dictionary.
        """
        return self.vocab


# EXAMPLE OF USAGE 

# sequence = "( not ( x_1 <= 0.2988 ) until[11,21] x_0 <= -0.7941 )"
# tokenizer = STLTokenizer('tokenizer_files/tokenizer.json')
# token_ids = tokenizer.encode(sequence)
# decoded_sequence = tokenizer.decode(token_ids)

# print("Original sequence: ", sequence)
# print("Encoded sequence: ", token_ids)
# print("Decoded sequence: ", decoded_sequence)


