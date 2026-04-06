# Tensor Parallelism (TP)

Tensor parallelism: splitting individual layers (weight matrices) across GPUs. Covers Megatron-style column/row parallel linear layers, attention head partitioning, communication patterns (AllReduce, AllGather), and math behind the splits.
