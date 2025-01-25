import torch
from torch.amp import autocast

import transformers
from transformers import BertTokenizerFast, BertModel
transformers.logging.set_verbosity_error()

class PretrainedDualEncoder(torch.nn.Module):
    def __init__(self, pretrained_doc_encoder: str, pretrained_query_encoder: str, device: torch.device, ngpu: int):
        super().__init__()

        # Setup device
        self.device = device
        self.ngpu   = ngpu
        
        # Setup encoder
        self.doc_tokenizer = BertTokenizerFast.from_pretrained(pretrained_doc_encoder)
        self.doc_encoder   = BertModel.from_pretrained(pretrained_doc_encoder, add_pooling_layer=False).to(self.device)
        self.doc_encoder.eval()

        self.query_tokenizer = BertTokenizerFast.from_pretrained(pretrained_query_encoder)
        self.query_encoder   = BertModel.from_pretrained(pretrained_query_encoder, add_pooling_layer=False).to(self.device)
        self.query_encoder.eval()

        # Multi-GPU support
        if self.ngpu > 1:
            self.doc_encoder   = torch.nn.DataParallel(self.doc_encoder)
            self.query_encoder = torch.nn.DataParallel(self.query_encoder)

    def encode(self, *args, is_doc=True):
        tokenizer = self.doc_tokenizer if is_doc else self.query_tokenizer
        encoder   = self.doc_encoder   if is_doc else self.query_encoder
        
        model_input = tokenizer(args[0], args[1], max_length=256, padding='max_length', truncation=True, return_tensors='pt') if is_doc else \
                      tokenizer(args[0]         , max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        model_input = model_input.to(self.device)

        with torch.no_grad(), autocast(device_type='cuda', enabled=True):
            model_output = encoder(**model_input)
            embedding    = model_output.last_hidden_state[:, 0, :] # CLS token embeddings

        return embedding.cpu().numpy()