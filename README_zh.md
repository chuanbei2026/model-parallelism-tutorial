[In English](README.md) | 中文版

# 深度学习系统教程

基于 Notebook 的交互式教程，涵盖分布式训练、推理优化与生成模型。每个主题都包含公式、图表、可运行代码和实际参考——提供英文、中文和文言文三种语言。

## 灵感

Karpathy 最近分享了他的 obsidian-llm 知识库方案——很好地展示了如何将个人笔记变成可查询的系统。这激发了一个更进一步的想法：如果知识库不仅能回答问题，还能有一个**任务头**，把论文、PDF 和博客文章提炼成一步步的交互式教程呢？

这个 repo 就是这样开始的。下一步计划是构建一个自动化工作流：每当往知识库添加新论文或博客，Claude Code 就自动生成教程 notebook 并推送到这里。知识进，教程出。

很高兴与大家分享——无论是人类还是 AI。一起进步吧。

## 模块

每个模块都是一系列从基础到进阶的 notebook。选择语言开始浏览：

| 模块 | 内容 | EN | 中文 | 文言 |
|------|------|----|------|------|
| **模型并行** | GPU 通信、DP、TP、PP、SP、CP、EP、混合策略、混合精度 | [9 notebooks](notebooks/en/model-parallelism/) | [9 notebooks](notebooks/zh/model-parallelism/) | [9 notebooks](notebooks/lzh/model-parallelism/) |
| **推理优化** | KV 缓存、Flash Attention、连续批处理、分页注意力、量化、投机解码、编译优化、服务架构 | [10 notebooks](notebooks/en/inference/) | [10 notebooks](notebooks/zh/inference/) | [10 notebooks](notebooks/lzh/inference/) |
| **扩散模型** | DDPM、采样加速、CFG、潜空间扩散、DiT、视频生成、训练工程、微调 | [8 notebooks](notebooks/en/diffusion/) | [8 notebooks](notebooks/zh/diffusion/) | [8 notebooks](notebooks/lzh/diffusion/) |
