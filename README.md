# Transformer Translation from Scratch — German → English

## Project Overview

This project implements a German-to-English Transformer translation system from scratch in PyTorch. The goal was to understand the architecture deeply and improve it through controlled experiments rather than rely on a pretrained translation model.

The project progressed from a simple word-level tokenizer to SentencePiece BPE, then tuned architecture, optimization, training duration, and decoding. The final system uses PyTorch, SentencePiece BPE, OPUS Books German-English data, SacreBLEU, greedy decoding, and beam search.

## Dataset

Dataset: **Helsinki-NLP/opus_books (de-en)**

- Total sentence pairs: ~51,467
- Training pairs: ~46,320
- Validation pairs: ~5,147
- Split: 90/10 with a fixed random seed

The BPE tokenizer was trained only on the training split.

## Initial Word-Level Pipeline

The first version used a simple regex tokenizer and separate German/English vocabularies.

- German vocabulary: ~26,953
- English vocabulary: ~17,627
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- Embeddings: learned from scratch with `nn.Embedding`

Pipeline:

```text
Raw text
→ word tokenizer
→ vocabulary lookup
→ token IDs
→ learned embeddings
→ Transformer
```

This baseline was useful for learning the architecture, but rare or unseen words could become `<unk>`.

## Transformer Architecture

The model was implemented from scratch using:

- learned source and target embeddings
- sinusoidal positional encoding
- scaled dot-product attention
- multi-head attention
- encoder blocks
- masked decoder self-attention
- encoder-decoder cross-attention
- position-wise feed-forward networks
- residual connections and layer normalization
- final vocabulary projection

Scaled dot-product attention follows:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

## Training Mechanics

For every batch:

```text
forward pass
→ loss
→ backward pass
→ gradients
→ optimizer.step()
→ model weights updated
```

With ~46,320 training examples and batch size 32, there are about **1,448 optimizer updates per epoch**. Over 20 epochs, that is roughly **28,960 updates**.

The same parameter tensors are reused across batches, but their values change after each optimizer step.

## Experiment Methodology

The project used controlled experiments: whenever practical, only one variable was changed at a time.

Main metrics:

- validation loss
- perplexity
- greedy BLEU
- beam BLEU
- training time

## Architecture Experiments

| Experiment | Val Loss | PPL | Greedy BLEU | Beam BLEU |
|---|---:|---:|---:|---:|
| d128, h4, enc3/dec3 | ~3.922 | ~50.52 | ~2.55 | ~2.79 |
| d128, h8, enc3/dec3 | ~3.915 | ~50.13 | ~2.94 | ~3.00 |
| d128, h8, enc4/dec4 | ~3.877 | ~48.26 | ~2.56 | ~3.27 |
| d256, h8, enc4/dec4 | ~3.908 | ~49.82 | ~1.50 | ~2.31 |
| d256, h8, enc4/dec4, ff1024 | ~4.316 | ~74.91 | ~0.41 | ~0.41 |

Key finding: **bigger was not automatically better**. The small dataset favored the more compact `d_model=128`, `d_ff=512` architecture.

## Initialization and LR Schedule

### Xavier + fixed LR

- Best epoch: 1
- Val loss: ~7.057
- PPL: ~1161
- Greedy BLEU: ~0.02
- Beam BLEU: ~0.02

### Xavier + Transformer schedule

- Val loss: ~4.095
- PPL: ~60.03
- Greedy BLEU: ~0.92
- Beam BLEU: ~1.61

This showed that initialization and optimization recipe interact strongly. Xavier was not inherently bad; it simply did not work well with the fixed learning-rate setup used here.

## Label Smoothing

With label smoothing enabled:

- Val loss: ~4.678
- PPL: ~107.52
- Greedy BLEU: ~3.07
- Beam BLEU: ~2.75

Because label smoothing changes the loss definition, its loss/perplexity are not directly comparable to unsmoothed runs.

## Dropout Experiments

| Dropout | Val Loss | PPL | Greedy BLEU | Beam BLEU |
|---:|---:|---:|---:|---:|
| 0.20 | 3.969 | 52.95 | 2.25 | 2.21 |
| 0.10 | ~3.877 | ~48.26 | ~2.56 | ~3.27 |
| 0.05 | 3.864 | 47.67 | 2.35 | 2.79 |

Final choice: **0.10**.

## Batch Size

Batch 64 reduced training time but was slightly worse in quality.

| Batch | Val Loss | PPL | Greedy BLEU | Beam BLEU |
|---:|---:|---:|---:|---:|
| 32 | ~3.877 | ~48.26 | ~2.56 | ~3.27 |
| 64 | 3.896 | 49.20 | 2.50 | 3.18 |

Final choice: **32**.

## Learning-Rate Sweep

| LR | Val Loss | PPL | Greedy BLEU | Beam BLEU |
|---:|---:|---:|---:|---:|
| 0.00100 | ~3.8767 | ~48.26 | ~2.56 | ~3.27 |
| 0.00075 | 3.8718 | 48.03 | **3.33** | 2.93 |
| 0.00050 | **3.8692** | **47.90** | 2.87 | 2.99 |
| 0.00025 | 3.9514 | 52.01 | 2.52 | 2.83 |

Final choice: **0.00075** as the best balance for later experiments.

## Weight Tying

Sharing the target embedding matrix with the output projection hurt performance.

- Val loss: ~4.149
- PPL: ~63.36
- Greedy BLEU: ~2.25
- Beam BLEU: ~2.63

Final setting: `weight_tying=False`.

## Adam Parameter Experiment

Transformer-paper-style Adam settings (`beta2=0.98`, `eps=1e-9`) were tested with the constant LR recipe.

Result:

- Val loss: 3.896
- PPL: 49.22
- Greedy BLEU: 2.56
- Beam BLEU: 2.17

The default Adam-style settings worked better:

```text
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
```

## Training Duration

| Epochs | Val Loss | PPL | Greedy BLEU | Beam BLEU |
|---:|---:|---:|---:|---:|
| 10 | ~3.872 | ~48.03 | ~3.33 | ~2.93 |
| 15 | 3.8290 | 46.02 | 3.62 | 3.34 |
| 20 | **3.8029** | **44.83** | **3.70** | 3.01 |

The model was still improving after 10 epochs, so 20 epochs became the final training duration.

# Moving to SentencePiece BPE

The major preprocessing improvement was replacing word-level tokenization with **SentencePiece BPE** using one shared German-English vocabulary.

Configuration:

```text
model_type = bpe
pad_id = 0
unk_id = 1
bos_id = 2
eos_id = 3
```

New pipeline:

```text
German + English training text
→ SentencePiece BPE
→ shared subword vocabulary
→ BPE IDs
→ learned embeddings
→ Transformer
```

The embeddings were still learned from scratch. Only tokenization changed.

## BPE Vocabulary Experiments

| Tokenization | Val Loss | PPL | Greedy BLEU | Beam BLEU |
|---|---:|---:|---:|---:|
| Word-level | 3.8029 | 44.83 | 3.70 | 3.01 |
| BPE 16k | 3.9598 | 52.45 | 3.7043 | 4.3067 |
| **BPE 8k** | 3.6729 | 39.37 | **3.9295** | **4.7039** |
| BPE 4k | **3.3964** | **29.86** | 3.1479 | 3.8509 |

Key finding: **8k BPE was the best balance**. The 4k tokenizer produced lower token-level loss/perplexity but worse final translation BLEU, likely because of excessive subword fragmentation.

Final tokenizer choice: **SentencePiece BPE, vocabulary size 8000**.

## Persistent Tokenizer Management

The training pipeline checks Google Drive for an existing tokenizer:

```text
found → reuse tokenizer
not found → train once → save to Drive → reuse later
```

This prevents retraining the tokenizer for every model experiment and keeps tokenization consistent.

# Decoding Experiments

Training and inference were separated so decoding could be tuned without retraining.

## Maximum Generation Length

For the best 8k model:

```text
MAX_LEN 60 → Beam BLEU ≈ 4.7039
MAX_LEN 80 → Beam BLEU ≈ 4.7524
```

Final choice: **80**.

## Length Penalty

| Alpha | Beam BLEU |
|---:|---:|
| 0.6 | 4.7524 |
| 0.8 | 4.7401 |
| **1.0** | **4.8005** |
| 1.2 | 4.6832 |

Final choice: **alpha = 1.0**.

## Beam Size

| Beam Size | Beam BLEU |
|---:|---:|
| 2 | 4.5087 |
| **4** | **4.8005** |
| 8 | 4.2332 |

Final choice: **beam size 4**.

# Final Best Configuration

```text
Tokenizer: SentencePiece BPE
BPE vocabulary: 8000

d_model: 128
heads: 8
encoder layers: 4
decoder layers: 4
d_ff: 512

dropout: 0.1
batch size: 32
learning rate: 0.00075
epochs: 20

initialization: PyTorch default
label smoothing: 0.0
weight tying: False

Adam beta1: 0.9
Adam beta2: 0.999
Adam eps: 1e-8

MAX_LEN: 80
beam size: 4
length penalty alpha: 1.0
```

Best observed Beam BLEU:

```text
4.8005
```

# Evaluation Notebook

`evaluate.ipynb` is used only for evaluation and inference:

```text
load tokenizer
→ load trained checkpoint
→ recreate model
→ load validation split
→ greedy BLEU
→ beam BLEU
→ decoding experiments
→ translation examples
→ qualitative error analysis
```

`main_train.py` is used for training experiments.

# Qualitative Error Analysis

A manual review of 20 validation examples produced:

## Meaning Preservation

```text
Yes      1
Partial  9
No      10
```

- 5% fully preserved meaning
- 45% partially preserved meaning
- 50% showed substantial semantic loss

## Fluency

```text
Mostly 17
Yes     3
```

The model generally produced readable English.

## Repetition

```text
Yes 8
No 12
```

Approximately 40% of reviewed outputs showed some repetition.

## Too Generic

```text
Yes 14
No   6
```

Approximately 70% contained overly generic phrasing.

## Main Error Types

```text
Semantic drift           12
Repetition                2
Repetition / Omission     2
Omission                  2
Good translation          1
Omission / Repetition     1
```

The dominant failure mode was **semantic drift**.

## Main Qualitative Finding

> The model learned target-language fluency more successfully than source-language semantic faithfulness.

Many outputs sounded like plausible literary English but did not preserve the original German meaning accurately. Beam search improved BLEU but could not fully solve semantic drift.

# What Worked

- 8 attention heads instead of 4
- 4 encoder/decoder layers instead of 3
- compact `d_model=128`, `d_ff=512`
- learning rate around 0.00075
- training longer than 10 epochs
- SentencePiece BPE instead of word-level tokenization
- BPE vocabulary size 8000
- `MAX_LEN=80`
- beam size 4
- length penalty alpha 1.0

# What Did Not Work

- `d_model=256`
- `d_ff=1024`
- Xavier + constant LR
- weight tying
- Transformer-style Adam values with constant LR
- batch size 64 for best quality
- dropout 0.20
- BPE vocabulary 4k for BLEU
- beam size 8
- length penalty 1.2

These failed experiments were valuable because they demonstrated that improvements depend strongly on dataset size and the full training recipe.

# Limitations

The final model has several important limitations:

1. **Small parallel corpus** — ~46k training sentence pairs is tiny compared with modern machine-translation datasets.
2. **Limited model capacity** — the final network is intentionally compact for experimentation.
3. **Literary domain** — OPUS Books encourages literary target-language patterns, including generic phrasing.
4. **Semantic drift** — the main qualitative limitation is loss of source meaning.
5. **Development BLEU subset** — experiments used a limited validation subset for fast iteration rather than a full production benchmark.
6. **No pretrained representations** — all embeddings and Transformer parameters were learned from scratch.
7. **No large-scale distributed training** — scaling beyond this experiment was intentionally left as future work.

# Future Work

Potential next steps include:

- training on much larger German-English parallel corpora
- evaluating on the complete held-out validation/test set
- using larger model capacity after increasing data size
- adding COMET or chrF alongside BLEU
- investigating repetition penalties or constrained decoding
- batched beam search
- cloud deployment of the Dockerized API
- ONNX/TorchScript optimization where appropriate
- W&B or MLflow experiment tracking

# Deployment

The trained Transformer has now been converted into a local deployable service using **FastAPI and Docker**.

Training benefits strongly from GPU because it requires forward pass, backward pass, gradients, and optimizer states. Inference is much lighter and can run on CPU for low-volume usage.

## Inference Optimization

The original beam-search inference called the full Transformer repeatedly while generating the target sentence. This meant the same German source sentence was encoded again at every decoding step.

For deployment, the Transformer was refactored to expose separate `encode()` and `decode()` methods.

```text
German sentence
→ encoder once
→ save encoder output
→ decoder step 1
→ decoder step 2
→ decoder step 3
→ ...
```

The source representation is therefore reused during autoregressive generation. This improves inference efficiency without changing the trained weights or requiring retraining.

## FastAPI

The model is served through a FastAPI application.

Main endpoint:

```text
POST /translate
```

Example request:

```json
{
  "text": "Es war ganz unmöglich, an diesem Tage einen Spaziergang zu machen."
}
```

Response structure:

```json
{
  "source": "Es war ganz unmöglich, an diesem Tage einen Spaziergang zu machen.",
  "translation": "..."
}
```

The checkpoint and tokenizer are loaded once when the API starts and reused across requests.

FastAPI's interactive API documentation is available at:

```text
/docs
```

when the service is running.

## Docker

The FastAPI service was containerized with Docker so the application can run in a consistent environment.

The Docker image contains the application runtime and dependencies such as:

- Python
- PyTorch
- FastAPI
- Uvicorn
- SentencePiece
- NumPy
- Transformer model code
- inference code

The trained checkpoint and tokenizer are kept as separate model artifacts and mounted into the container instead of being tied to a machine-specific path.


# Project Structure

```text
transformer-translation/
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── model_service.py
├── config.py
├── main_train.py
├── test_deployment.py
├── data/
│   ├── data_utils.py
│   └── bpe_tokenizers.py
├── model/
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
├── inference/
│   ├── greedy.py
│   ├── beam.py
│   └── translate.py
├── experiments/
│   ├── logger.py
│   └── plotting.py
├── notebooks/
│   └── evaluate.ipynb
├── checkpoints/
├── tokenizers/
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

# Final Conclusion

A complete German-to-English Transformer translation system was implemented from scratch and improved through systematic experiments.

The strongest final setup used an 8k shared SentencePiece BPE vocabulary, a compact 4-layer encoder/decoder Transformer, and tuned beam-search decoding. The best observed Beam BLEU was approximately **4.8005**.

The project was then extended beyond model experimentation into deployment: inference was optimized by reusing the encoder output, the model was exposed through FastAPI, and the service was successfully containerized and tested locally with Docker.

The primary modeling limitation remains semantic faithfulness: the model learned readable target-language patterns more effectively than precise German-to-English semantic alignment. The logical modeling improvement would therefore be substantially more and more diverse parallel training data, while the next engineering step would be optional cloud hosting of the Dockerized API.
