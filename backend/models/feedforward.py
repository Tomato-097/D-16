# Step 3: The Feed-Forward Layer (models/feedforward.py)
# Open models/feedforward.py. This represents the multilayer perceptron (MLP) applied right after attention inside the transformer block.
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)