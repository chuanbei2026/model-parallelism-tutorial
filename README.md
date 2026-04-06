# Model Parallelism Tutorial

Interactive, notebook-based teaching package for model parallelism and distributed training techniques. Learn DP, TP, PP, SP, CP, EP, GPU communication primitives, and more — with formulas, diagrams, runnable code, and Megatron-LM references.

## Topics

| # | Folder | Topic |
|---|--------|-------|
| 00 | `notebooks/00-gpu-communication/` | GPU Communication Primitives |
| 01 | `notebooks/01-data-parallelism/` | Data Parallelism (DP) |
| 02 | `notebooks/02-tensor-parallelism/` | Tensor Parallelism (TP) |
| 03 | `notebooks/03-pipeline-parallelism/` | Pipeline Parallelism (PP) |
| 04 | `notebooks/04-sequence-parallelism/` | Sequence Parallelism (SP) |
| 05 | `notebooks/05-context-parallelism/` | Context Parallelism (CP) |
| 06 | `notebooks/06-expert-parallelism/` | Expert Parallelism (EP / MoE) |
| 07 | `notebooks/07-advanced-topics/` | Advanced Topics & Combined Strategies |

## Setup

### Local (Mac / CPU)

```bash
# Clone the repo
git clone <repo-url> && cd tinker-model-parallelism

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
