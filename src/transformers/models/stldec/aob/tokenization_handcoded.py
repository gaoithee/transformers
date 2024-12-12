import re
from typing import List

class TemporalLogicTokenizer:
    def __init__(self):
        # Definizione dei pattern principali
        self.token_pattern = re.compile(
            r'(\band\b|\bor\b|\bnot\b|'  # Operatori logici (`and`, `or`, `not`)
            r'\balways\b|\beventually\b|\buntil\b|'  # Operatori temporali (`until`, `always`, `eventually`)
            r'x_\d+|'  # Variabili (prima del parsing dettagliato)
            r'-?\d+\.\d+|-?\d+|'  # Numeri (interi o decimali con segno), ovvero TIENI `-` come token
            r'\[|\]|\(|\)|,|inf)'  # Simboli speciali (`[`, `]`, `(`, `)`, `,`, `inf`)
        )
    
    def split_number(self, number: str) -> List[str]:
        """
        Divide un numero in token individuali (es. 2.0375 -> ['2', '.', '0', '3', '7', '5']).
        """
        return list(number)  # Ogni carattere diventa un token separato
    
    def split_variable(self, variable: str) -> List[str]:
        """
        Divide una variabile come x_0 in token individuali (es. x_0 -> ['x', '_', '0']).
        """
        return list(variable)  # Ogni carattere diventa un token separato
    
    def tokenize(self, text: str) -> List[str]:
        # Cerca tutti i token nella stringa
        tokens = self.token_pattern.findall(text)
        # Espandi numeri e variabili nei loro singoli caratteri
        expanded_tokens = []
        for token in tokens:
            if re.match(r'-?\d+\.\d+|-?\d+', token):  # Riconosce i numeri
                expanded_tokens.extend(self.split_number(token))
            elif re.match(r'x_\d+', token):  # Riconosce le variabili
                expanded_tokens.extend(self.split_variable(token))
            else:
                expanded_tokens.append(token)
        return expanded_tokens
    

# example of usage: 

    # test = '( ( ( always ( not ( x_0 >= 2.0375 ) ) until[0,10] ( x_1 <= 0.1976 until[2,20] eventually[12,16] ( x_1 >= 0.8872 ) ) ) until[3,inf] eventually[11,inf] ( x_0 >= 0.8886 ) ) and x_1 >= 1.8647 )'
    # tokenizer = TemporalLogicTokenizer()
    # tokens = tokenizer.tokenize(test)