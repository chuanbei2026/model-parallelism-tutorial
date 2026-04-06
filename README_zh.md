[English](README.md)

# 模型并行教程

基于 Notebook 的交互式模型并行与分布式训练教学包。学习 DP、TP、PP、SP、CP、EP、GPU 通信原语等——配有公式、图示、可运行代码和 Megatron-LM 参考。

## 主题

| # | 主题 | English | 中文 | 文言 |
|---|------|---------|------|------|
| 00 | GPU 通信原语 | [EN](notebooks/00-gpu-communication/01-gpu-communication.en.ipynb) | [中文](notebooks/00-gpu-communication/01-gpu-communication.zh.ipynb) | [文言](notebooks/00-gpu-communication/01-gpu-communication.lzh.ipynb) |
| 01 | 数据并行 (DP) | [EN](notebooks/01-data-parallelism/01-data-parallelism.en.ipynb) | [中文](notebooks/01-data-parallelism/01-data-parallelism.zh.ipynb) | [文言](notebooks/01-data-parallelism/01-data-parallelism.lzh.ipynb) |
| 02 | 张量并行 (TP) | [EN](notebooks/02-tensor-parallelism/01-tensor-parallelism.en.ipynb) | [中文](notebooks/02-tensor-parallelism/01-tensor-parallelism.zh.ipynb) | [文言](notebooks/02-tensor-parallelism/01-tensor-parallelism.lzh.ipynb) |
| 03 | 流水线并行 (PP) | [EN](notebooks/03-pipeline-parallelism/03-pipeline-parallelism-en.ipynb) | [中文](notebooks/03-pipeline-parallelism/03-pipeline-parallelism-zh.ipynb) | [文言](notebooks/03-pipeline-parallelism/03-pipeline-parallelism-classical.ipynb) |
| 04 | 序列并行 (SP) | [EN](notebooks/04-sequence-parallelism/01-sequence-parallelism.en.ipynb) | [中文](notebooks/04-sequence-parallelism/01-sequence-parallelism.zh.ipynb) | [文言](notebooks/04-sequence-parallelism/01-sequence-parallelism.lzh.ipynb) |
| 05 | 上下文并行 (CP) | [EN](notebooks/05-context-parallelism/01-context-parallelism.en.ipynb) | [中文](notebooks/05-context-parallelism/01-context-parallelism.zh.ipynb) | [文言](notebooks/05-context-parallelism/01-context-parallelism.lzh.ipynb) |
| 06 | 专家并行 (EP / MoE) | [EN](notebooks/06-expert-parallelism/ep.en.ipynb) | [中文](notebooks/06-expert-parallelism/ep.zh.ipynb) | [文言](notebooks/06-expert-parallelism/ep.lzh.ipynb) |
| 07 | 并行策略组合 | [EN](notebooks/07-parallelism-mix-strategy/07-parallelism-mix-strategy.en.ipynb) | [中文](notebooks/07-parallelism-mix-strategy/07-parallelism-mix-strategy.zh.ipynb) | [文言](notebooks/07-parallelism-mix-strategy/07-parallelism-mix-strategy.lzh.ipynb) |

## 环境搭建

### 本地（Mac / CPU）

```bash
# 克隆仓库
git clone https://github.com/chuanbei2026/model-parallelism-tutorial.git && cd model-parallelism-tutorial

# 以可编辑模式安装
pip install -e .

# 启动 Jupyter
jupyter notebook
```

大多数概念 Notebook（理论、图示、简单模拟）无需 GPU 即可在本地运行。

### 多 GPU 环境

标记有 `# [GPU-REQUIRED]` 的单元格需要真实 CUDA 硬件（建议 4 张以上 GPU）。如果你有远程 GPU 服务器：

```bash
# 在 GPU 服务器上
cd /path/to/tinker-model-parallelism
pip install -e .
jupyter notebook --no-browser --port=8888
```

然后在本地转发端口：

```bash
# 在本地机器上
ssh -NL 8888:localhost:8888 <your-gpu-server>
```

在浏览器中打开 `http://localhost:8888` 使用远程 Jupyter 内核。

## Notebook 结构

每个 Notebook 遵循统一结构：

1. **概述** — 覆盖内容、前置知识
2. **概念与原理** — 理论、公式、直觉
3. **可视化图解** — 图示、视觉解释
4. **在 LLM 中的应用** — 该技术在大语言模型中的使用方式
5. **动手代码** — 最小化、可运行的示例
6. **Megatron 参考** — Megatron-LM 实现指引
7. **总结与延伸阅读** — 要点回顾与链接

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 格式化代码
black mp_tutorial/
isort mp_tutorial/
```
