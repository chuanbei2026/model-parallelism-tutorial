[In English](README.md) | 中文版

# 模型并行教程

基于 Notebook 的交互式教学包，涵盖模型并行与分布式训练技术。学习 DP、TP、PP、SP、CP、EP、GPU 通信原语等——包含公式、图表、可运行代码及 Megatron-LM 参考。

## 主题

| # | 主题 | English | 中文 | 文言 |
|---|------|---------|------|------|
| 00 | GPU 通信原语 | [EN](notebooks/en/00-gpu-communication.ipynb) | [中文](notebooks/zh/00-gpu-communication.ipynb) | [文言](notebooks/lzh/00-gpu-communication.ipynb) |
| 01 | 数据并行 (DP) | [EN](notebooks/en/01-data-parallelism.ipynb) | [中文](notebooks/zh/01-data-parallelism.ipynb) | [文言](notebooks/lzh/01-data-parallelism.ipynb) |
| 02 | 张量并行 (TP) | [EN](notebooks/en/02-tensor-parallelism.ipynb) | [中文](notebooks/zh/02-tensor-parallelism.ipynb) | [文言](notebooks/lzh/02-tensor-parallelism.ipynb) |
| 03 | 流水线并行 (PP) | [EN](notebooks/en/03-pipeline-parallelism.ipynb) | [中文](notebooks/zh/03-pipeline-parallelism.ipynb) | [文言](notebooks/lzh/03-pipeline-parallelism.ipynb) |
| 04 | 序列并行 (SP) | [EN](notebooks/en/04-sequence-parallelism.ipynb) | [中文](notebooks/zh/04-sequence-parallelism.ipynb) | [文言](notebooks/lzh/04-sequence-parallelism.ipynb) |
| 05 | 上下文并行 (CP) | [EN](notebooks/en/05-context-parallelism.ipynb) | [中文](notebooks/zh/05-context-parallelism.ipynb) | [文言](notebooks/lzh/05-context-parallelism.ipynb) |
| 06 | 专家并行 (EP / MoE) | [EN](notebooks/en/06-expert-parallelism.ipynb) | [中文](notebooks/zh/06-expert-parallelism.ipynb) | [文言](notebooks/lzh/06-expert-parallelism.ipynb) |
| 07 | 并行策略混合 | [EN](notebooks/en/07-parallelism-mix-strategy.ipynb) | [中文](notebooks/zh/07-parallelism-mix-strategy.ipynb) | [文言](notebooks/lzh/07-parallelism-mix-strategy.ipynb) |

## 安装

### 本地环境 (Mac / CPU)

```bash
# 克隆仓库
git clone https://github.com/chuanbei2026/model-parallelism-tutorial.git && cd model-parallelism-tutorial

# 以可编辑模式安装
pip install -e .

# 启动 Jupyter
jupyter notebook
```

大部分概念 notebook（理论、图表、小规模模拟）可在本地无 GPU 环境下运行。

### 多 GPU 环境

标记为 `# [GPU-REQUIRED]` 的单元格需要真实的 CUDA 硬件（推荐 4 张以上 GPU）。如果你有远程 GPU 服务器：

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

在浏览器中打开 `http://localhost:8888` 即可使用远程 Jupyter 内核。

## Notebook 结构

每个 notebook 遵循统一结构：

1. **概览** — 内容介绍、前置知识
2. **概念与原理** — 理论、公式、直觉
3. **可视化图解** — 图表、视觉化说明
4. **在 LLM 中的应用** — 该技术在大语言模型中的使用方式
5. **动手代码** — 最小化、可运行的示例
6. **Megatron 参考** — Megatron-LM 实现要点
7. **总结与延伸阅读** — 要点回顾与链接

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 格式化代码
black mp_tutorial/
isort mp_tutorial/
```
