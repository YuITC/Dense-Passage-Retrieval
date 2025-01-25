import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('utils')

from utils.dpr_utils   import set_seed
from utils.dpr_encoder import PretrainedDualEncoder
from embedder          import load_queries, embed_documents, embed_queries
from settings          import PRETRAINED_DOC_ENCODER, PRETRAINED_QUERY_ENCODER, DEVICE, NGPU, WIKI_PATH, TEST_SET, CHUNK_SIZE, SHARD_NUMS

if __name__ == '__main__':
    set_seed(42)
    os.makedirs('dpr-embeddings/documents/', exist_ok=True)
    os.makedirs('dpr-embeddings/queries/'  , exist_ok=True)
    
    dualEncoder = PretrainedDualEncoder(
        pretrained_doc_encoder=PRETRAINED_DOC_ENCODER,
        pretrained_query_encoder=PRETRAINED_QUERY_ENCODER,
        device=DEVICE,
        ngpu=NGPU
    )
 
    chunks     = pd.read_csv(WIKI_PATH, sep='\t', usecols=['text', 'title'], chunksize=CHUNK_SIZE)
    queries, _ = load_queries(TEST_SET)
        
    # Use the commented code below in case doesn't have enough RAM
    # start, end = 0, 4
    for i, chunk in enumerate(tqdm(chunks, total=SHARD_NUMS, desc='Loading documents ...')):
        # if i < start: continue
        # elif i > end: break
        
        chunk['text']  = chunk['text'].str.strip('"')
        shard          = chunk[['title', 'text']].values.tolist()
        doc_embeddings = embed_documents(shard, dualEncoder)
        np.save(f"dpr-embeddings/documents/wiki_shard_{i}.npy", doc_embeddings)

    query_embeddings = embed_queries(queries, dualEncoder)
    np.save('dpr-embeddings/queries/NQ_queries.npy', query_embeddings)