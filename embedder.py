import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('utils')

from torch.utils.data import DataLoader

from utils.dpr_utils import normalize_query
from settings        import CHUNK_SIZE, SHARD_NUMS, ENCODING_BS

def load_documents(file_path):
    chunks = pd.read_csv(file_path, sep='\t', usecols=['text', 'title'], chunksize=CHUNK_SIZE)
    wiki   = []
    
    for chunk in tqdm(chunks, total=SHARD_NUMS, desc='Loading documents ...'):
        chunk['text'] = chunk['text'].str.strip('"')
        wiki.extend(chunk[['title', 'text']].values.tolist())
    return wiki

def load_queries(file_path):
    data             = pd.read_csv(file_path, sep='\t')
    queries, answers = [], []
    
    for _, row in tqdm(data.iterrows(), total=len(data), desc='Loading queries ...'):
        queries.append(normalize_query(row.iloc[0]))
        answers.append(eval(row.iloc[1]))
    return queries, answers

def embed_documents(documents, encoder):
    titles     = [str(x[0]) for x in documents]
    texts      = [str(x[1]) for x in documents]
    dataloader = DataLoader(list(zip(titles, texts)), batch_size=ENCODING_BS, shuffle=False, num_workers=4, pin_memory=True)
    embeddings = []

    for batch in tqdm(dataloader, desc='Embedding documents ...'):
        title_batch, text_batch = batch
        batch_embeddings = encoder.encode(title_batch, text_batch, is_doc=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def embed_queries(queries, encoder):
    dataloader = DataLoader(queries, batch_size=ENCODING_BS, shuffle=False, num_workers=4, pin_memory=True)
    embeddings = []
    
    for batch in tqdm(dataloader, desc='Embedding queries ...'):
        batch_embeddings = encoder.encode(batch, is_doc=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)