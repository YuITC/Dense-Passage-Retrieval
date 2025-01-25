import json
import argparse
import numpy as np

import transformers
transformers.logging.set_verbosity_error()

from utils.dpr_encoder import PretrainedDualEncoder
from utils.dpr_utils   import set_seed, has_answer
from embedder          import load_documents, load_queries, embed_queries
from retriever         import retrieval, visualize
from settings          import NGPU, WIKI_PATH, TEST_SET, PRETRAINED_DOC_ENCODER, PRETRAINED_QUERY_ENCODER, DEVICE

def retrieving_result(save_path, queries, answers, retr_res, wiki, num_docs=10):
    with open('demo/result_' + save_path + '.txt', 'w', encoding='utf-8') as f:
        for q_idx, (query, answer, retr_SI) in enumerate(zip(queries, answers, retr_res), start=1):
            f.write('=' * 100 + '\n')
            f.write(f"Query {q_idx}: {query}\n")
            f.write(f"Answer: {answer}\n")
            f.write('=' * 100 + '\n')
            f.write('\nRetrieval results:\n\n')

            for rank, (score, idx) in enumerate(retr_SI[:num_docs], start=1):
                title, text = wiki[idx]
                has_ans = 'May contain answer' if has_answer(answer, text) else 'May not contain answer'
                f.write(f"Rank [{rank}] - Score [{score:.2f}] - Title [{title}] [{has_ans}]\n")
                f.write(f"{text}\n\n")
            
if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_source', choices=['demo', 'dataset'], default='dataset')
    parser.add_argument('--demo_file')
    args = parser.parse_args()
    
    dualEncoder = PretrainedDualEncoder(
        pretrained_doc_encoder=PRETRAINED_DOC_ENCODER, 
        pretrained_query_encoder=PRETRAINED_QUERY_ENCODER, 
        device=DEVICE,
        ngpu=NGPU
    )
    wiki = load_documents(WIKI_PATH)
    
    if args.query_source == 'demo':
        with open(args.demo_file, 'r', encoding='utf-8') as f:
            demo = json.load(f)
        queries = [item['query']  for item in demo]
        answers = [item['answer'] for item in demo]
        query_embeddings = embed_queries(queries, dualEncoder)
        
        top_k     = 10
        save_path = 'demo_source'
        
    elif args.query_source == 'dataset':
        queries, answers = load_queries(TEST_SET)
        query_embeddings = np.load('dpr-embeddings/queries/NQ_queries.npy')
        
        top_k     = 100
        save_path = 'dataset_source'
        
    topK_ac, retr_res = retrieval(query_embeddings, answers, wiki, k=top_k)
    visualize(save_path, topK_ac, k=top_k, save_fig=True)
    retrieving_result(save_path, queries, answers, retr_res, wiki, num_docs=10)