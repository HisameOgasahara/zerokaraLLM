# zerokaraLLM

『밑바닥부터 시작하는 딥러닝 6』의 공식 일본어 코드 저장소를 Google Colab에서 학습하기 좋게 재구성한 노트북 모음입니다.

기준 upstream commit: `c9b6e2ed531b08dd9f451a091a34e9645148e2e2`

## 구성 원칙

- 1~7장의 공식 `.py` 구현을 빠뜨리지 않습니다.
- 모델 구조, 알고리즘, 기본 하이퍼파라미터를 임의로 토이 버전으로 축소하지 않습니다.
- 함수·클래스·실행부를 노트북 셀에 맞게 나눠 가독성을 높입니다.
- 장에서 사용하는 공통 `codebot` / `storybot` 구현도 노트북 안에서 직접 확인할 수 있습니다.
- 자연어 설명과 코드 주석은 한국어로 제공합니다. 기술 용어와 변수명은 원래 표기를 유지합니다.
- 8~9장은 현재 단일 Colab T4 실습 범위에서 제외합니다.

## Colab 노트북

| 장 | 주제 | Colab |
|---|---|---|
| 1장 | 문자에서 BPE까지 토크나이저 만들기 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch01_tokenizer.ipynb) |
| 2장 | Attention에서 Transformer와 GPT까지 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch02_transformer_gpt.ipynb) |
| 3장 | 사전학습, 생성, SFT, GRPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch03_pretrain_sft_grpo.ipynb) |
| 4장 | BPE 최적화와 병렬 토크나이징 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch04_tokenizer_optimization.ipynb) |
| 5장 | RoPE, SwiGLU, RMSNorm, KV Cache | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch05_modern_llm_blocks.ipynb) |
| 6장 | 학습 최적화와 DPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch06_training_optimization_dpo.ipynb) |
| 7장 | 토크나이저 스케일링 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/blob/main/notebooks/ch07_tokenizer_scaling.ipynb) |
