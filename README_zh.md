[In English](README.md) | 中文版

# 深度学习系统教程

基于 Notebook 的交互式教程，涵盖分布式训练、推理优化与生成模型。每个主题都包含公式、图表、可运行代码和实际参考——提供英文、中文和文言文三种语言。

## 模块

### 1. 模型并行与分布式训练

如何跨 GPU 扩展训练：通信原语、数据/张量/流水线/序列/上下文/专家并行、混合策略与混合精度训练。

| # | 主题 | EN | 中文 | 文言 |
|---|------|----|------|------|
| 00 | GPU 通信原语 | [EN](notebooks/en/00-gpu-communication.ipynb) | [中文](notebooks/zh/00-gpu-communication.ipynb) | [文言](notebooks/lzh/00-gpu-communication.ipynb) |
| 01 | 数据并行 (DP) | [EN](notebooks/en/01-data-parallelism.ipynb) | [中文](notebooks/zh/01-data-parallelism.ipynb) | [文言](notebooks/lzh/01-data-parallelism.ipynb) |
| 02 | 张量并行 (TP) | [EN](notebooks/en/02-tensor-parallelism.ipynb) | [中文](notebooks/zh/02-tensor-parallelism.ipynb) | [文言](notebooks/lzh/02-tensor-parallelism.ipynb) |
| 03 | 流水线并行 (PP) | [EN](notebooks/en/03-pipeline-parallelism.ipynb) | [中文](notebooks/zh/03-pipeline-parallelism.ipynb) | [文言](notebooks/lzh/03-pipeline-parallelism.ipynb) |
| 04 | 序列并行 (SP) | [EN](notebooks/en/04-sequence-parallelism.ipynb) | [中文](notebooks/zh/04-sequence-parallelism.ipynb) | [文言](notebooks/lzh/04-sequence-parallelism.ipynb) |
| 05 | 上下文并行 (CP) | [EN](notebooks/en/05-context-parallelism.ipynb) | [中文](notebooks/zh/05-context-parallelism.ipynb) | [文言](notebooks/lzh/05-context-parallelism.ipynb) |
| 06 | 专家并行 (EP / MoE) | [EN](notebooks/en/06-expert-parallelism.ipynb) | [中文](notebooks/zh/06-expert-parallelism.ipynb) | [文言](notebooks/lzh/06-expert-parallelism.ipynb) |
| 07 | 并行策略混合 | [EN](notebooks/en/07-parallelism-mix-strategy.ipynb) | [中文](notebooks/zh/07-parallelism-mix-strategy.ipynb) | [文言](notebooks/lzh/07-parallelism-mix-strategy.ipynb) |
| 08 | 混合精度训练 | [EN](notebooks/en/08-mixed-precision-training.ipynb) | [中文](notebooks/zh/08-mixed-precision-training.ipynb) | [文言](notebooks/lzh/08-mixed-precision-training.ipynb) |

### 2. 推理优化

如何快速服务模型：KV 缓存、Flash Attention、连续批处理、分页注意力、量化、投机解码、编译优化与服务架构。

| # | 主题 | EN | 中文 | 文言 |
|---|------|----|------|------|
| 00 | KV 缓存 | [EN](notebooks/en/inference/00-kv-cache.ipynb) | [中文](notebooks/zh/inference/00-kv-cache.ipynb) | [文言](notebooks/lzh/inference/00-kv-cache.ipynb) |
| 01 | Flash Attention | [EN](notebooks/en/inference/01-flash-attention.ipynb) | [中文](notebooks/zh/inference/01-flash-attention.ipynb) | [文言](notebooks/lzh/inference/01-flash-attention.ipynb) |
| 02 | 连续批处理 | [EN](notebooks/en/inference/02-continuous-batching.ipynb) | [中文](notebooks/zh/inference/02-continuous-batching.ipynb) | [文言](notebooks/lzh/inference/02-continuous-batching.ipynb) |
| 03 | 分页注意力 | [EN](notebooks/en/inference/03-paged-attention.ipynb) | [中文](notebooks/zh/inference/03-paged-attention.ipynb) | [文言](notebooks/lzh/inference/03-paged-attention.ipynb) |
| 04 | 前缀缓存 | [EN](notebooks/en/inference/04-prefix-caching.ipynb) | [中文](notebooks/zh/inference/04-prefix-caching.ipynb) | [文言](notebooks/lzh/inference/04-prefix-caching.ipynb) |
| 05 | 量化与剪枝 | [EN](notebooks/en/inference/05-quantization-pruning.ipynb) | [中文](notebooks/zh/inference/05-quantization-pruning.ipynb) | [文言](notebooks/lzh/inference/05-quantization-pruning.ipynb) |
| 06 | 投机解码 | [EN](notebooks/en/inference/06-speculative-decoding.ipynb) | [中文](notebooks/zh/inference/06-speculative-decoding.ipynb) | [文言](notebooks/lzh/inference/06-speculative-decoding.ipynb) |
| 07 | 模型编译 | [EN](notebooks/en/inference/07-model-compilation.ipynb) | [中文](notebooks/zh/inference/07-model-compilation.ipynb) | [文言](notebooks/lzh/inference/07-model-compilation.ipynb) |
| 08 | 服务架构 | [EN](notebooks/en/inference/08-serving-architecture.ipynb) | [中文](notebooks/zh/inference/08-serving-architecture.ipynb) | [文言](notebooks/lzh/inference/08-serving-architecture.ipynb) |
| 09 | 注意力机制 | [EN](notebooks/en/inference/09-attention-mechanisms.ipynb) | [中文](notebooks/zh/inference/09-attention-mechanisms.ipynb) | [文言](notebooks/lzh/inference/09-attention-mechanisms.ipynb) |

### 3. 扩散模型

从噪声到图像：DDPM、采样加速、无分类器引导、潜空间扩散、DiT、视频生成、训练工程与微调（ControlNet、LoRA、DreamBooth）。

| # | 主题 | EN | 中文 | 文言 |
|---|------|----|------|------|
| 00 | DDPM 基础 | [EN](notebooks/en/diffusion/00-ddpm-foundations.ipynb) | [中文](notebooks/zh/diffusion/00-ddpm-foundations.ipynb) | [文言](notebooks/lzh/diffusion/00-ddpm-foundations.ipynb) |
| 01 | 采样加速 | [EN](notebooks/en/diffusion/01-sampling-acceleration.ipynb) | [中文](notebooks/zh/diffusion/01-sampling-acceleration.ipynb) | [文言](notebooks/lzh/diffusion/01-sampling-acceleration.ipynb) |
| 02 | 无分类器引导 | [EN](notebooks/en/diffusion/02-classifier-free-guidance.ipynb) | [中文](notebooks/zh/diffusion/02-classifier-free-guidance.ipynb) | [文言](notebooks/lzh/diffusion/02-classifier-free-guidance.ipynb) |
| 03 | 潜空间扩散 | [EN](notebooks/en/diffusion/03-latent-diffusion.ipynb) | [中文](notebooks/zh/diffusion/03-latent-diffusion.ipynb) | [文言](notebooks/lzh/diffusion/03-latent-diffusion.ipynb) |
| 04 | DiT 架构 | [EN](notebooks/en/diffusion/04-dit-architecture.ipynb) | [中文](notebooks/zh/diffusion/04-dit-architecture.ipynb) | [文言](notebooks/lzh/diffusion/04-dit-architecture.ipynb) |
| 05 | 视频生成 | [EN](notebooks/en/diffusion/05-video-generation.ipynb) | [中文](notebooks/zh/diffusion/05-video-generation.ipynb) | [文言](notebooks/lzh/diffusion/05-video-generation.ipynb) |
| 06 | 训练工程 | [EN](notebooks/en/diffusion/06-training-engineering.ipynb) | [中文](notebooks/zh/diffusion/06-training-engineering.ipynb) | [文言](notebooks/lzh/diffusion/06-training-engineering.ipynb) |
| 07 | 微调与适配 | [EN](notebooks/en/diffusion/07-finetuning-adaptation.ipynb) | [中文](notebooks/zh/diffusion/07-finetuning-adaptation.ipynb) | [文言](notebooks/lzh/diffusion/07-finetuning-adaptation.ipynb) |

## 灵感

上周 Karpathy 分享了他的 [obsidian-llm](https://github.com/karpathy/obsidian-llm) 知识库方案。我很喜欢这个想法——但除了一个可查询的知识库，我更想要一个**任务头**：能够将论文、PDF 和博客文章提炼成一步步的交互式教程。这个 repo 就是这样诞生的。

**下一步计划**：构建一个自动化工作流——每当我往知识库添加论文或博客，就自动触发 Claude Code 生成新的教程 notebook 并推送到这里。知识进，教程出。

很高兴与大家分享——无论是人类还是 AI。一起学习吧。
