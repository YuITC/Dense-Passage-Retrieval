# Dense-Passage-Retrieval

[Dense Passage Retrieval (DPR)](https://arxiv.org/pdf/2004.04906.pdf) is a highly regarded technique that underpins retrieval-augmented Large Language Models (LLMs). While the original DPR repository is an excellent resource for academic research, it offers a multitude of configurable options that may be challenging for beginners to navigate.

This repository presents a streamlined implementation of the DPR model using the Natural Questions dataset. It is designed to facilitate a clear and straightforward understanding of DPR without compromising on essential details. Additionally, a pre-trained DPR model is provided for immediate experimentation.


## 1. Requirements

**NOTE**: This repository is compatible with Python 3.10.

Install the appropriate **PyTorch and FAISS** version based on your CUDA version. Visit the [PyTorch Previous Versions page](https://pytorch.org/get-started/previous-versions/) and [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for detailed instructions.
```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
```

Install the remaining **Python libraries**:
```bash
pip install transformers numpy pandas seaborn matplotlib tqdm wget
```


## 2. Installation

Clone this repository and navigate to its directory:
```bash
git clone https://github.com/YuITC/Dense-Passage-Retrieval.git
cd Dense-Passage-Retrieval
```


## 3. Download Data

This repository utilizes the following datasets:
- **English Wikipedia Dump (Dec. 20, 2018)**: Serves as the source documents for answering questions.
- **Natural Questions (NQ) Dataset**: Provides the question-answer pairs for training and evaluation.

Execute the following commands to download the necessary data:
```bash
python utils/download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir dpr-dataset
python utils/download_data.py --resource data.retriever.nq --output_dir dpr-dataset
python utils/download_data.py --resource data.retriever.qas.nq --output_dir dpr-dataset
```


## 4. Usage

This repository provides pre-trained **query encoder** and **document encoder** models, available [here](https://drive.google.com/drive/folders/1QAqzFNhUWBg5pCyAC-zQLGkWTStvPH5Q?usp=sharing).

**Embedding phase**: Generate embeddings for queries and documents.
```bash
python embedder_main.py
```

**Retrieving phase**: Retrieve relevant documents based on queries.
```bash
# For retrieving on the dataset
python retriever_main.py --query_source dataset

# For retrieving on the demo queries
python retriever_main.py --query_source demo --demo_file demo/demo_actual_queries.json
```


## 5. Results

**Retrieval Performance**:
The following table compares the retrieval performance of the replicated DPR model against the reported results in the original paper on the Natural Questions (NQ) dataset:
|          | Top-20 | Top-100 |
|:--------:|:------:|:-------:|
| [Paper](https://arxiv.org/pdf/2004.04906.pdf) |  78.4  |   85.4  |
|   Ours   |  79.1  |   86  |

**Computational Costs**:
All experiments were conducted on an 2xT4 16GB GPU setup (powered by Kaggle).
|          | generate embedding |   build & search index  |
|:--------:|:------------------:|:--------------:|
| Duration |      15h        |       7m 27s      |  


## 6. References
[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) by VVladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.

[facebookresearch/DPR](https://github.com/facebookresearch/DPR) – Official GitHub repository for DPR by Facebook Research.

[Hannibal046/nanoDPR](https://github.com/Hannibal046/nanoDPR) – A lightweight implementation of DPR.

[YouTube Tutorial by @IRwithPUGGY](https://www.youtube.com/watch?v=TbObc9s8Gzw) – An instructional video explaining DPR concepts.


## 7. License
This project is licensed under the [Apache License](LICENSE).


## 8. Contact
For any questions or feedback, please open an issue or contact [lehuuphuoc2502yuitc@gmail.com](lehuuphuoc2502yuitc@gmail.com).