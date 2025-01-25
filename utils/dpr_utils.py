import re
import unicodedata
import random
import numpy as np

import torch

from utils.dpr_tokenizers import SimpleTokenizer

def set_seed(seed=42): 
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def normalize_document(document):
    document = re.sub(r"[’‘“”]", "'", document)
    document = re.sub(r"\n", " ", document)
    return re.sub(r'^\s*"?|"?\s*$', '', document)

def normalize_query(question):
    return re.sub(r"[’‘“”]", "'", question)

def normalize(text):
    """
    Normalization Form Decomposed.
    Decomposes characters into their base form and combining marks.
    """
    return unicodedata.normalize('NFD', text) 

def has_answer(answers, doc):
    tokenizer = SimpleTokenizer()
    
    doc = tokenizer.tokenize(normalize(doc)).words(uncased=True)
    for answer in answers:
        answer = tokenizer.tokenize(normalize(answer)).words(uncased=True)
        
        # Check if the tokenized answer appears in the tokenized document
        for i in range(0, len(doc) - len(answer) + 1): 
            if answer == doc[i : i+len(answer)]:
                return True
    return False