# Khmer GPT - Text Generation Web Application

A lightweight, self-contained web interface and production-ready API for running inference on a custom GPT model trained from scratch. The system architecture processes Khmer script text prompt encodings locally through your custom 8-layer attention matrix stack.

---

## 🏗️ System Architecture & Workflow

The application is split into two independent modules communicating via JSON REST endpoints:

1. **Frontend UI (`/frontend`)**: A pure HTML/CSS/JS user interface serving an interactive text completion layout. It dispatches prompt inputs over a local fetch network stream and renders generative updates.
2. **Backend Engine (`/backend`)**: A robust FastAPI microservice layer that dynamically builds your custom neural model schema (`GPT`), populates the network structural nodes with saved checkpoint weight arrays (`ckpt.pt`), and runs autoregressive token prediction passes locally.

---

## 📂 Project Directory Layout

Ensure your structural workspace folder alignment matches this distribution exactly for automated tracking parameters to resolve:

```text
MY_TRANSFORMER/
├── backend/
│   ├── config/
│   │   └── config_loader.py       # Configuration parser logic
│   ├── models/
│   │   ├── attention.py           # Causal Self-Attention block metrics
│   │   ├── embeddings.py          # Token & Position embedding lookups
│   │   ├── feedforward.py         # Deep linear expansion/GELU layers
│   │   ├── gpt_model.py           # Master structural network wrapper
│   │   └── transformer_block.py   # Pre-LN residual container block
│   ├── tokenizer/
│   │   └── khmer_tokenizer.json   # Custom Hugging Face tokenizer file
│   ├── model/
│   │   └── my-transformer/
│   │       └── checkpoints/
│   │           └── ckpt.pt        # Raw model weights state dictionary
│   ├── main.py                    # FastAPI server & inference pipeline
│   └── requirements.txt           # Python dependency manifests
├── frontend/
│   ├── index.html                 # Text generation interface
│   ├── script.js                  # Fetch API proxy event pipeline
│   └── style.css                  # Typography & structural layout styles
└── README.md