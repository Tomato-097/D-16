from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256       # Paragraph context size
    vocab_size: int = 32000     # Khmer tokenizer vocabulary size
    n_layer: int = 8            # Keeping your deep 8-layer architecture
    n_head: int = 8             # 384 / 8 = 48 dimension per head
    n_embd: int = 384           # Embedding dimension
    dropout: float = 0.1        # Overfitting protection
    learning_rate: float = 6e-4 # CRUCIAL: The gas pedal to smash the 6.6 loss barrier
    batch_size: int = 32        # Your hard T4 GPU VRAM limit to prevent crashes