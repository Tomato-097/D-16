# Step 4: The Transformer Block (models/transformer_block.py)
# Open models/transformer_block.py. This layer glues LayerNorm, your custom attention mechanism, and feed-forward layers together using residual connections.
import torch
import torch.nn as nn
from models.attention import CausalSelfAttention
from models.feedforward import FeedForward

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # Pre-layer normalization with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x