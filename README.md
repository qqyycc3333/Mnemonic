# Mnemonic

Official implementation of **“Mnemonic: Selecting Tuning Gradients to Mitigate Catastrophic Forgetting in LLMs.”**

Mnemonic is a lightweight gradient-selection plugin for parameter-efficient fine-tuning. It assigns lower retention probabilities to large gradients and models gradient distributions hierarchically across attention groups, helping reduce parameter drift while preserving downstream-task adaptation.

## Highlights

- Rehearsal-free and end-to-end trainable
- Compatible with LoRA and DoRA
- No additional inference overhead
- Less than 1% additional training time in our experiments
- Evaluated on Llama 3 and Qwen 3 with mathematical and coding tasks

Compared with vanilla fine-tuning, Mnemonic improves general capabilities by up to **1.22%** and target-task performance by up to **1.99%**.

## Repository Structure

```text
Mnemonic/
├── LLaMA-Factory/          # Training framework
├── config/                 # Training and evaluation configurations
├── data/                   # Dataset files and registrations
├── scripts/                # Launch scripts for training and evaluation
├── lm-evaluation-harness/  # General-capability evaluation
├── evaluate/               # Additional evaluation utilities
└── model/                  # Model download scripts
```

## Quick Start

```bash
git clone https://github.com/qqyycc3333/Mnemonic.git
cd Mnemonic

pip install -r requirements.txt
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

Download the model:

```bash
python model/download.py
```

Before training, update the model path and `output_dir` in the corresponding configuration or launch script.

Run the MetaMathQA experiment:

```bash
bash scripts/metamathqa/llama3-8b/run_sft.sh
```

Run the CodeTester experiment:

```bash
bash scripts/codetester/llama3-8b/run_sft.sh
```

Additional configurations for Llama and Qwen models are provided under `config/` and `scripts/`.

## Paper

**Mnemonic: Selecting Tuning Gradients to Mitigate Catastrophic Forgetting in LLMs**

Yunchong Qian, Xiaochun Yang, Weiwei Cheng, and Hao Zeng.

The paper link and citation information will be updated after publication.

## Acknowledgements

This repository is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## TODO
Due to code privacy restrictions, the code for some related methods could not be uploaded immediately; it will be added successively once the paper is accepted.
