# Step 1: Foundational Embeddings (models/embeddings.py)
# Open up models/embeddings.py. This file handles converting your token IDs into vectors and adding spatial awareness via positional encodings.
import torch
import torch.nn as nn

class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Token embedding lookup table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional embedding lookup table
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        
        # Generation of positions array [0, 1, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Add token vectors and position vectors together
        tok_emb = self.wte(idx) # shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # shape (t, n_embd)
        
        return self.drop(tok_emb + pos_emb)