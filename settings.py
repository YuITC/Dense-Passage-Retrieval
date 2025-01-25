import math
import torch

# Pretrained models
PRETRAINED_DOC_ENCODER   = 'dpr-trained-dualencoder/doc_encoder'
PRETRAINED_QUERY_ENCODER = 'dpr-trained-dualencoder/query_encoder'

# Retrieval document: Wikipedia
WIKI_PATH = 'dpr-dataset/downloads/data/wikipedia_split/psgs_w100.tsv'
WIKI_DOCS = 21_015_324

# Dataset: Natural Question
TEST_SET = 'dpr-dataset/downloads/data/retriever/qas/nq-test.csv'

# Encoding and data processing parameters
ENCODING_BS = 256
CHUNK_SIZE  = 10**6
SHARD_NUMS  = math.ceil(WIKI_DOCS / CHUNK_SIZE)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NGPU   = torch.cuda.device_count()