# Step 5: The Top-Level Brain Container (models/gpt_model.py)
# Finally, open models/gpt_model.py. This is the core master module that orchestrates the entire model from input IDs to output probability distributions (logits).
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.embeddings import GPTEmbeddings
from models.transformer_block import Block

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            embeddings = GPTEmbeddings(config),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying optimization
        self.transformer.embeddings.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size limit is {self.config.block_size}"

        # 1. Forward embeddings layer
        x = self.transformer.embeddings(idx)
        
        # 2. Forward stack of transformer decoder blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # 3. Output layer and cross-entropy loss computation
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return logits, loss