import os
import sys
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torch.nn import functional as F
from tokenizers import Tokenizer

# Append current directory to system paths so internal architecture imports resolve cleanly
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from models.gpt_model import GPT

# ---------------------------------------------------------------------------
# Path and Hardware Initializations
# ---------------------------------------------------------------------------
MODEL_DIR = CURRENT_DIR / "model"
# Targets backend/model/my-transformer/checkpoints/ckpt.pt
LOCAL_MODEL_PATH = MODEL_DIR / "my-transformer/checkpoints/ckpt.pt"
TOKENIZER_PATH = CURRENT_DIR / "tokenizer/khmer_tokenizer.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model Bootstrapping Core Execution
# ---------------------------------------------------------------------------
def load_khmer_gpt_pipeline():
    """Validates local filesystem layouts, sets up structures, and loads state maps."""
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing custom checkpoint file at: {LOCAL_MODEL_PATH.resolve()}")
    
    # 1. Instantiating Custom Tokenizer Map File
    print(f"-> Loading custom Khmer Tokenizer from: {TOKENIZER_PATH.name}")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Missing tokenizer mapping dictionary file at: {TOKENIZER_PATH.resolve()}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # 2. Extracting configurations and weight structures from raw .pt asset
    print(f"-> Parsing model metadata maps from: {LOCAL_MODEL_PATH.name}")
    checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    
    # 3. Generating custom network layout empty chassis
    print("-> Initializing GPT structure layer nodes...")
    model = GPT(config).to(device)
    
    # 4. Injecting state metrics directly into layer matrices
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Khmer GPT Infrastructure Successfully Initialized!")
    
    return model, tokenizer, config

# Bootstrap pipeline once at service startup
model, tokenizer, model_config = load_khmer_gpt_pipeline()

# ---------------------------------------------------------------------------
# FastAPI Gateway Application Interface
# ---------------------------------------------------------------------------
app = FastAPI(title="Khmer GPT Production API Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=150)
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)
    top_k: int = Field(default=50, ge=1)

class GenerateResponse(BaseModel):
    generated_text: str

@app.get("/")
def health():
    return {"status": "online", "engine": "Khmer GPT Block Stacking Core", "device": str(device)}

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        # Step 1: Vectorize incoming prompt text using your custom local tokenizer layout
        start_ids = tokenizer.encode(request.prompt.strip()).ids
        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Step 2: Autoregressive processing generation loop derived from your terminal script
        with torch.no_grad():
            for _ in range(request.max_new_tokens):
                # Ensure input length conforms strictly to the block size limit of the attention matrix
                x_cond = x if x.size(1) <= model_config.block_size else x[:, -model_config.block_size:]
                
                logits, _ = model(x_cond)
                logits = logits[:, -1, :] / request.temperature
                
                if request.top_k is not None:
                    v, _ = torch.torch.topk(logits, min(request.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                    
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, idx_next), dim=1)

        # Step 3: Parse internal prediction arrays back into custom string format
        generated_ids = x[0].tolist()
        output_text = tokenizer.decode(generated_ids)
        
        return GenerateResponse(generated_text=output_text)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Failure: {str(exc)}")  