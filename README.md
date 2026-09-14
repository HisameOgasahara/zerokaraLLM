# zerokaraLLM

Colab notebooks for the official code of [oreilly-japan/deep-learning-from-scratch-6](https://github.com/oreilly-japan/deep-learning-from-scratch-6).

## Policy

These notebooks do **not** replace the book code with simplified toy implementations.

- Chapters 1–7 use the official upstream source files without omitting or rewriting them.
- The upstream revision is pinned to commit `c9b6e2ed531b08dd9f451a091a34e9645148e2e2`.
- Each notebook is organized as **show the complete original source → run the original source** for every file in that chapter.
- Model architecture, algorithms, optimizer logic, training procedures, tokenizer logic, and intermediate examples are not compressed.
- Chapters 8 and 9 are intentionally excluded for now.
- If a T4-specific reduced run is added later, it should be an additional optional cell. The original implementation remains unchanged.

## Colab notebooks

| Chapter | Included upstream files | Colab |
|---|---:|---|
| 1 — Tokenizer | 9 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch01_tokenizer.ipynb) |
| 2 — Attention and GPT | 9 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch02_transformer_gpt.ipynb) |
| 3 — Pretraining, SFT, Chat, GRPO | 6 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch03_pretrain_sft_grpo.ipynb) |
| 4 — BPE Optimization | 6 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch04_tokenizer_optimization.ipynb) |
| 5 — Modern LLM Blocks | 6 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch05_modern_llm_blocks.ipynb) |
| 6 — Training Optimization and DPO | 9 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch06_training_optimization_dpo.ipynb) |
| 7 — Tokenizer Scaling | 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch07_tokenizer_scaling.ipynb) |

## Notes for Colab T4

The notebooks request a GPU runtime, but the original chapter code is preserved as-is. CPU-oriented tokenizer chapters naturally do most work on the CPU. Longer training scripts can take substantial runtime; they are not silently shortened in these notebooks.
