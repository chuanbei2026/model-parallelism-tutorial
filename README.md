In English | [中文版](README_zh.md)

# Model Parallelism Tutorial

Interactive, notebook-based teaching package for model parallelism and distributed training techniques. Learn DP, TP, PP, SP, CP, EP, GPU communication primitives, and more — with formulas, diagrams, runnable code, and Megatron-LM references.

## Topics

| # | Topic | English | 中文 | 文言 |
|---|-------|---------|------|------|
| 00 | GPU Communication Primitives | [EN](notebooks/00-gpu-communication/01-gpu-communication.en.ipynb) | [中文](notebooks/00-gpu-communication/01-gpu-communication.zh.ipynb) | [文言](notebooks/00-gpu-communication/01-gpu-communication.lzh.ipynb) |
| 01 | Data Parallelism (DP) | [EN](notebooks/01-data-parallelism/01-data-parallelism.en.ipynb) | [中文](notebooks/01-data-parallelism/01-data-parallelism.zh.ipynb) | [文言](notebooks/01-data-parallelism/01-data-parallelism.lzh.ipynb) |
| 02 | Tensor Parallelism (TP) | [EN](notebooks/02-tensor-parallelism/01-tensor-parallelism.en.ipynb) | [中文](notebooks/02-tensor-parallelism/01-tensor-parallelism.zh.ipynb) | [文言](notebooks/02-tensor-parallelism/01-tensor-parallelism.lzh.ipynb) |
| 03 | Pipeline Parallelism (PP) | [EN](notebooks/03-pipeline-parallelism/03-pipeline-parallelism-en.ipynb) | [中文](notebooks/03-pipeline-parallelism/03-pipeline-parallelism-zh.ipynb) | [文言](notebooks/03-pipeline-parallelism/03-pipeline-parallelism-classical.ipynb) |
| 04 | Sequence Parallelism (SP) | [EN](notebooks/04-sequence-parallelism/01-sequence-parallelism.en.ipynb) | [中文](notebooks/04-sequence-parallelism/01-sequence-parallelism.zh.ipynb) | [文言](notebooks/04-sequence-parallelism/01-sequence-parallelism.lzh.ipynb) |
| 05 | Context Parallelism (CP) | [EN](notebooks/05-context-parallelism/01-context-parallelism.en.ipynb) | [中文](notebooks/05-context-parallelism/01-context-parallelism.zh.ipynb) | [文言](notebooks/05-context-parallelism/01-context-parallelism.lzh.ipynb) |
| 06 | Expert Parallelism (EP / MoE) | [EN](notebooks/06-expert-parallelism/ep.en.ipynb) | [中文](notebooks/06-expert-parallelism/ep.zh.ipynb) | [文言](notebooks/06-expert-parallelism/ep.lzh.ipynb) |
| 07 | Parallelism Mix Strategy | [EN](notebooks/07-parallelism-mix-strategy/07-parallelism-mix-strategy.en.ipynb) | [中文](notebooks/07-parallelism-mix-strategy/07-parallelism-mix-strategy.zh.ipynb) | [文言](notebooks/07-parallelism-mix-strategy/07-parallelism-mix-strategy.lzh.ipynb) |

## Setup

### Local (Mac / CPU)

```bash
# Clone the repo
git clone https://github.com/chuanbei2026/model-parallelism-tutorial.git && cd model-parallelism-tutorial

# Install the package in editable mode
pip install -e .

# Launch Jupyter
jupyter notebook
```

Most concept notebooks (theory, diagrams, toy simulations) run locally without a GPU.

### Multi-GPU Environment

Cells tagged with `# [GPU-REQUIRED]` need real CUDA hardware (4+ GPUs recommended). If you have a remote GPU server:

```bash
# On the GPU server
cd /path/to/tinker-model-parallelism
pip install -e .
jupyter notebook --no-browser --port=8888
```

Then forward the port locally:

```bash
# From your local machine
ssh -NL 8888:localhost:8888 <your-gpu-server>
```

Open `http://localhost:8888` in your browser to use the remote Jupyter kernel.

## Notebook Structure

Every notebook follows a standard structure:

1. **Overview** — what the notebook covers, prerequisites
2. **Concepts & Principles** — theory, formulas, intuition
3. **Visual Illustrations** — diagrams, visual explanations
4. **Application in LLMs** — how the technique is used in large language models
5. **Hands-on Code** — minimal, runnable examples
6. **Megatron Reference** — Megatron-LM implementation pointers
7. **Summary & Further Reading** — takeaways and links

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Format code
black mp_tutorial/
isort mp_tutorial/
```
