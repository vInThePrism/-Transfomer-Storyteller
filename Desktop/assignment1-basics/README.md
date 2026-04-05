# 🤖 MiniGPT — Transformer LM from Scratch

A clean, from-scratch implementation of a GPT-style autoregressive Transformer language model in PyTorch, trained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset. Built as CS336 Assignment 1.

---

## ✨ Features

- **Custom everything** — `Linear`, `Embedding`, `RMSNorm`, `SwiGLU`, `RoPE`, `MultiHeadAttention`, `TransformerBlock`, `TransformerLM` all implemented from scratch (no `nn.Linear` shortcuts)
- **Flash Attention** — automatically uses `scaled_dot_product_attention` when available for GPU speedup
- **Rotary Positional Embedding (RoPE)** — lazily-cached sin/cos tables, applied per head
- **SwiGLU FFN** — gated activation with `d_ff = ⌈8/3 · d_model⌉` rounded to 64
- **Custom AdamW** — hand-written optimizer with bias correction and decoupled weight decay
- **Cosine LR schedule** — linear warmup + cosine annealing
- **Gradient clipping** — global norm clipping
- **Memory-mapped dataset** — efficient streaming of large binary token files via `np.memmap`
- **Checkpoint system** — saves model + optimizer state per epoch, fully resumable
- **Sampling utilities** — temperature, top-k, top-p (nucleus) sampling with special token suppression
- **Ablation suite** — three architectural variants for systematic comparison

---

## 🔬 Ablation Studies

| Variant | Change vs. Baseline |
|---|---|
| `SiLU` | Replaces SwiGLU with plain SiLU (single projection, no gating) |
| `post_Norm` | Applies RMSNorm *after* the residual add (post-norm) instead of before |
| `not_RMSNorm` | Removes all RMSNorm from residual blocks entirely |

---

## 🛠 Tech Stack

| Component | Library |
|---|---|
| Deep learning | PyTorch |
| Tokenization | HuggingFace `tokenizers` / `transformers` |
| Data pipeline | NumPy memmap |
| Visualization | Matplotlib |
| System monitoring | psutil |

---

## 📦 Installation

```bash
# Clone the repo
git clone <repo-url>
cd assignment1-basics

# Install dependencies
pip install torch transformers tokenizers numpy matplotlib psutil tqdm
```

---

## 🚀 Quick Start

### 1. Prepare training data

Download [TinyStoriesV2-GPT4-train.txt](https://huggingface.co/datasets/roneneldan/TinyStories) and place it in the project root, then:

```bash
python get_train_data.py
# Outputs: data.bin (int32 token binary, ~800k samples)
```

### 2. Train the model

```bash
python train.py \
  --epochs 5 \
  --batch_size 16 \
  --context_length 128 \
  --d_model 256 \
  --num_heads 8 \
  --num_layers 6 \
  --lr 3e-4 \
  --data_path data.bin
```

Checkpoints are saved to `ckpt/epoch_N.pt`. PPL curves saved as `ckpt/train_ppl.png` and `ckpt/val_ppl.png`.

### 3. Generate text

```bash
python model.py
# Uses ckpt/epoch_5.pt to generate from prompt "Lily like play computer games"
```

### 4. Run ablations

```bash
# From the Assignment1_Ablations/ directory:
cd Assignment1_Ablations

python SiLU_train.py       # SiLU FFN ablation
python post_Norm_train.py  # Post-norm ablation
python not_RMSNorm_train.py # No-norm ablation
```

---

## 📁 Project Structure

```
assignment1-basics/
│
├── model.py                  # Full model definition + sampling + CustomAdamW
├── train.py                  # Training loop, dataset, LR schedule, checkpointing
├── get_train_data.py         # Tokenize TinyStories → data.bin
│
├── bpe_tokenizer/
│   └── tokenizer.json        # BPE tokenizer (vocab ~50k)
│
├── ckpt/                     # Saved checkpoints (epoch_N.pt) + PPL plots
│
└── Assignment1_Ablations/
    ├── tokenizer.json        # Tokenizer for ablation runs
    ├── data.in               # Pre-tokenized binary data for ablations
    │
    ├── SiLU_model.py         # Baseline with SiLU FFN
    ├── SiLU_train.py
    │
    ├── post_Norm_model.py    # Post-norm variant
    ├── post_Norm_train.py
    │
    ├── not_RMSNorm_model.py  # No-norm variant
    └── not_RMSNorm_train.py
```

---

## 🏗 Architecture

```
Input tokens (B, T)
      │
  Embedding (vocab_size → d_model)
      │
  ┌───┴─────────────────────────┐
  │   TransformerBlock × N      │
  │   ┌──────────────────────┐  │
  │   │ RMSNorm              │  │
  │   │ MultiHeadAttention   │  │  ← RoPE on Q, K
  │   │ + residual           │  │
  │   │ RMSNorm              │  │
  │   │ SwiGLU FFN           │  │
  │   │ + residual           │  │
  │   └──────────────────────┘  │
  └─────────────────────────────┘
      │
  RMSNorm (final)
      │
  LM Head (d_model → vocab_size)
      │
  Logits (B, T, vocab_size)
```

**Default config:** `d_model=256`, `num_heads=8`, `num_layers=6`, `context_length=128`, `vocab_size=50257` → ~10M parameters

---

## 📊 Screenshots

> Training perplexity curve (baseline model, 5 epochs on TinyStories):

![Training PPL](ckpt/train_ppl.png)

> Validation perplexity curve:

![Validation PPL](ckpt/val_ppl.png)

---

## ⚙️ Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 256 | Hidden dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 6 | Transformer blocks |
| `context_length` | 128 | Max sequence length |
| `lr` | 3e-4 | Peak learning rate |
| `min_lr` | 3e-5 | Minimum LR (end of cosine decay) |
| `weight_decay` | 0.1 | AdamW weight decay |
| `batch_size` | 16 | Batch size |
| Warmup | 10% of total steps | Linear warmup fraction |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
