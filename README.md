# zerokaraLLM

Colab notebooks for `oreilly-japan/deep-learning-from-scratch-6`.

The notebooks for Chapters 1-7 contain the complete upstream chapter implementations. Source files are reorganized into readable notebook cells without dropping algorithms or replacing them with toy versions. Shared `codebot` / `storybot` / `webbot` modules needed by a chapter are also exposed in visible cells.

Pinned upstream commit: `c9b6e2ed531b08dd9f451a091a34e9645148e2e2`

Chapters 8 and 9 are intentionally excluded for now.

| Chapter | Topic | Colab |
|---|---|---|
| 1 | Tokenizer from characters to BPE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch01_tokenizer.ipynb) |
| 2 | Attention, Transformer and GPT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch02_transformer_gpt.ipynb) |
| 3 | Pretraining, generation, SFT and GRPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch03_pretrain_sft_grpo.ipynb) |
| 4 | BPE optimization and parallel tokenization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch04_tokenizer_optimization.ipynb) |
| 5 | RoPE, SwiGLU, RMSNorm and KV cache | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch05_modern_llm_blocks.ipynb) |
| 6 | Training optimization and DPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch06_training_optimization_dpo.ipynb) |
| 7 | Tokenizer scaling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch07_tokenizer_scaling.ipynb) |

## Policy

- Preserve upstream architecture, algorithms, dimensions, and default training hyperparameters.
- Split `.py` files into notebook cells only for readability; generation fails if concatenation differs from the original source.
- Keep shared local modules visible instead of hiding important implementation behind an external import.
- Use the upstream checkout only for data/assets and the original package layout required by chapter scripts.
- Do not introduce reduced/toy defaults merely to make a notebook finish quickly on T4.
