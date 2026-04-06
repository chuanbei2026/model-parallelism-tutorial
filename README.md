In English | [中文版](README_zh.md)

# Deep Learning Systems Tutorial

Interactive, notebook-based tutorials on distributed training, inference optimization, and generative models. Every topic comes with formulas, diagrams, runnable code, and real-world references — in English, Chinese, and Classical Chinese.

## Inspiration

Karpathy recently shared his [obsidian-llm](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) knowledge base setup — a great demonstration of turning personal notes into a queryable system. This sparked a further idea: what if a knowledge base didn't just answer questions, but had a **task head** that could distill papers, PDFs, and blog posts into step-by-step interactive tutorials?

That's how this repo started. The next step is to build an automated workflow: whenever a new paper or blog is added to the knowledge base, Claude Code generates a tutorial notebook and pushes it here. Knowledge in, tutorials out.

Happy to share with everyone — humans and AI alike. Let's grow together.

## Chapters

Each chapter is a series of notebooks progressing from fundamentals to advanced topics. Pick a language to browse:

| Chapter | Description | EN | 中文 | 文言 |
|--------|-------------|----|------|------|
| **Model Parallelism** | GPU communication, DP, TP, PP, SP, CP, EP, mixed strategies, mixed precision, RL alignment | [10 notebooks](notebooks/en/model-parallelism/) | [10 notebooks](notebooks/zh/model-parallelism/) | [10 notebooks](notebooks/lzh/model-parallelism/) |
| **Inference Optimization** | KV cache, flash attention, continuous batching, paged attention, quantization, speculative decoding, compilation, serving | [10 notebooks](notebooks/en/inference/) | [10 notebooks](notebooks/zh/inference/) | [10 notebooks](notebooks/lzh/inference/) |
| **Diffusion Models** | DDPM, sampling acceleration, CFG, latent diffusion, DiT, video generation, training engineering, fine-tuning | [8 notebooks](notebooks/en/diffusion/) | [8 notebooks](notebooks/zh/diffusion/) | [8 notebooks](notebooks/lzh/diffusion/) |
