# zerokaraLLM

Colab implementation of https://github.com/oreilly-japan/deep-learning-from-scratch-6

## Colab notebooks

The notebooks are organized by chapter and tuned for a single Google Colab T4 where GPU work is involved. Chapters 8 and 9 are intentionally excluded for now because their larger-scale model/training exercises are less suitable for a simple single-T4 notebook workflow.

| Chapter | Topic | Colab |
|---|---|---|
| 1 | Character/byte tokenizer, BPE, special tokens, pre-tokenization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch01_tokenizer.ipynb) |
| 2 | Attention, causal mask, multi-head attention, Transformer, GPT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch02_transformer_gpt.ipynb) |
| 3 | Pretraining, generation, SFT, GRPO core objective | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch03_pretrain_sft_grpo.ipynb) |
| 4 | BPE optimization, cache, chunking, parallel preprocessing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch04_tokenizer_optimization.ipynb) |
| 5 | RoPE, SwiGLU, RMSNorm, modern attention, KV cache | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch05_modern_llm_blocks.ipynb) |
| 6 | AdamW, LR schedule, BF16, clipping, validation, DPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch06_training_optimization_dpo.ipynb) |
| 7 | Larger BPE training/encoding and tokenizer scaling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch07_tokenizer_scaling.ipynb) |

## Execution policy

- Chapters 1, 4 and 7 are mainly CPU/tokenizer exercises.
- Chapters 2, 3, 5 and 6 detect CUDA automatically and are sized for a Colab T4.
- Long training runs from the book are shortened into smoke tests so that forward, backward, optimization and loss behavior can be checked interactively.
- Increase model size, sequence length or training steps only after the small configuration runs correctly.
