In English | [中文版](README_zh.md)

# Model Parallelism Tutorial

Interactive, notebook-based teaching package for model parallelism and distributed training techniques. Learn DP, TP, PP, SP, CP, EP, GPU communication primitives, and more — with formulas, diagrams, runnable code, and Megatron-LM references.

## Topics

| # | Topic | English | 中文 | 文言 |
|---|-------|---------|------|------|
| 00 | GPU Communication Primitives | [EN](notebooks/en/00-gpu-communication.ipynb) | [中文](notebooks/zh/00-gpu-communication.ipynb) | [文言](notebooks/lzh/00-gpu-communication.ipynb) |
| 01 | Data Parallelism (DP) | [EN](notebooks/en/01-data-parallelism.ipynb) | [中文](notebooks/zh/01-data-parallelism.ipynb) | [文言](notebooks/lzh/01-data-parallelism.ipynb) |
| 02 | Tensor Parallelism (TP) | [EN](notebooks/en/02-tensor-parallelism.ipynb) | [中文](notebooks/zh/02-tensor-parallelism.ipynb) | [文言](notebooks/lzh/02-tensor-parallelism.ipynb) |
| 03 | Pipeline Parallelism (PP) | [EN](notebooks/en/03-pipeline-parallelism.ipynb) | [中文](notebooks/zh/03-pipeline-parallelism.ipynb) | [文言](notebooks/lzh/03-pipeline-parallelism.ipynb) |
| 04 | Sequence Parallelism (SP) | [EN](notebooks/en/04-sequence-parallelism.ipynb) | [中文](notebooks/zh/04-sequence-parallelism.ipynb) | [文言](notebooks/lzh/04-sequence-parallelism.ipynb) |
| 05 | Context Parallelism (CP) | [EN](notebooks/en/05-context-parallelism.ipynb) | [中文](notebooks/zh/05-context-parallelism.ipynb) | [文言](notebooks/lzh/05-context-parallelism.ipynb) |
| 06 | Expert Parallelism (EP / MoE) | [EN](notebooks/en/06-expert-parallelism.ipynb) | [中文](notebooks/zh/06-expert-parallelism.ipynb) | [文言](notebooks/lzh/06-expert-parallelism.ipynb) |
| 07 | Parallelism Mix Strategy | [EN](notebooks/en/07-parallelism-mix-strategy.ipynb) | [中文](notebooks/zh/07-parallelism-mix-strategy.ipynb) | [文言](notebooks/lzh/07-parallelism-mix-strategy.ipynb) |

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
