# Multimodal UNcommonsense: From Odd to Ordinary and Ordinary to Odd (EMNLP 2025 finding)

**Authors:**
**[Yejin hand*](https://github.com/ozzaney)** ,
**[Saejin Kim*](https://0110tpwls.github.io/)**,
[Dongjun Min](https://github.com/mdj1214),
[Youngjae Yu](https://yj-yu.github.io/home/),

[[`Dataset`](https://huggingface.co/datasets/Steamout/MUN)]

This repository contains a set of Python scripts for generating datasets, conducting multimodal inference, and evaluating various language-vision models. The tools are designed to streamline the process of creating, refining, and comparing rationales and situations derived from image-text pairs.

---

## Features

### Dataset Generation
- Scripts for generating dataset entries combining captions, rationales, and situations.
- Ensures outputs are in a consistent, parseable format.
- Supports zero-shot and few-shot rationalization modes.

### Multimodal Inference
- Leverages vision-language models including:
  - **InternVL2.5**
  - **Phi3.5V**
  - **Phi4Vmm**
  - **Qwen2VL**
  - **Qwen2.5VL**
  - **Llava-OneVision**

### Evaluation Framework
- Scripts for comparing and ranking models based on their generated outputs.
- Combines text and image retrieval for more robust evaluation.

### Model Integration
- Supports efficient embedding generation and retrieval using models like OpenCLIP and HuggingFace.

---

## Requirements

Install all dependencies using the provided `env.yaml` file:

```bash
conda env create -f env.yaml
conda activate multimodal-env
```

## Usage

### Dataset Generation
To generate a dataset with rationales and situations, use:

```bash
python dataset_generation.py --data_file [INPUT_FILE]
```
### Multimodal Inference
Run inference using specific vision-language models:

```bash
python vlm_explanation_generation.py --model [MODEL_NAME] --data_type [DATA_TYPE] --mode [zero_shot|n_shot] --sample_mode [random|retrieval]
```
### Model Comparison
Evaluate and rank models with:

```bash
python hllm_comparision.py
```

## Scripts Overview
1. dataset_generation.py
Generates datasets containing captions, rationales, and situations using OpenAI GPT models.
Outputs structured data for downstream tasks.
2. gpt_explanation_generation.py
Generates explanations for image-situation pairs in zero-shot or few-shot modes.
Converts AVIF images to PNG if needed.
3. hllm_comparision.py
Compares generated rationales from two models and produces rankings.
4. vlm_explanation_generation.py
Integrates multiple vision-language models for rationalization and situation inference.
5. ensemble_retriever.py
Implements ensemble image-text retrieval for improved dataset generation and evaluation.


