import heapq
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import faiss
import transformers
transformers.logging.set_verbosity_error()

from utils.dpr_utils import has_answer
from settings        import NGPU, CHUNK_SIZE, SHARD_NUMS

def retrieval_by_shard(i, query_embeddings, k):
    co        = faiss.GpuMultipleClonerOptions()
    co.shard  = True
    index_cpu = faiss.IndexFlatIP(768)
    index_gpu = faiss.index_cpu_to_gpus_list(index_cpu, co=co, ngpu=NGPU)

    shard_data = np.lib.format.open_memmap(f"dpr-embeddings/documents/wiki_shard_{i}.npy", mode='r')
    index_gpu.add(shard_data)
    del shard_data

    S, I       = index_gpu.search(query_embeddings, k)
    global_ids = I + (CHUNK_SIZE * i)
    shard_res  = [[(S[q, r], global_ids[q, r]) for r in range(k)] for q in range(query_embeddings.shape[0])]

    return shard_res

def retrieval(query_embeddings, answers, wiki, k=100):
    q_num = query_embeddings.shape[0]
    retr  = [[] for _ in range(q_num)]

    # Retrieval by shard
    for i in tqdm(range(SHARD_NUMS), desc=f'Retrieving shard ...'):
        retr_shard = retrieval_by_shard(i, query_embeddings, k)
        for q in range(q_num):
            retr[q].extend(retr_shard[q])

    # Combine results
    retr_res = []
    for q in range(q_num):
        q_topK = heapq.nlargest(k, retr[q], key=lambda x: x[0])
        retr_res.append(q_topK)
    
    # Top-k accuracy
    hit_per_k = [0] * k
    for answer, retr_SI in tqdm(zip(answers, retr_res), total=q_num, desc='Retrieval hits per query ...'):
        docs      = [wiki[doc_id][1] for _, doc_id in retr_SI]
        hit       = [has_answer(answer, doc) for doc in docs]
        first_hit = next((i for i, x in enumerate(hit) if x), None)
        if first_hit is not None:
            hit_per_k[first_hit:] = [h+1 for h in hit_per_k[first_hit:]] # If answer appears at top-k, it also appears at top-h (h > k)
    topK_ac = [x/q_num for x in hit_per_k]
    
    return topK_ac, retr_res
    
def visualize(save_path, topK_ac, k=100, save_fig=False):
    topK_ac = [0] + topK_ac # range: [0, 100]
    
    sns.set_style('whitegrid')
    sns.set_context('talk', rc={'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    _, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(range(0, k + 1), topK_ac, linestyle='-', marker='o', markevery=10, color='royalblue', label='Hit Rate / Accuracy')
    ax.fill_between(range(0, k + 1), topK_ac, color='royalblue', alpha=0.2)

    for x, y in zip(range(0, k + 1), topK_ac):
        if x == 1 or x % 10 == 0:
            ax.text(x-0.01, y+0.01, f"{y:.3f}", fontsize=12, fontweight='bold', color='black', ha='center', va='bottom')

    ax.set_title('Top-k Retrieval Accuracy', pad=15, fontweight='bold')
    ax.set_xlabel('k (Top-k)', fontweight='bold')
    ax.set_ylabel('Accuracy / Hit Rate (%)', fontweight='bold')
    ax.set_xticks(range(0, k + 1, 10))
    ax.grid(alpha=0.5)
    ax.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig('demo/topK_' + save_path + '.png', dpi=300, bbox_inches='tight')
    plt.show()