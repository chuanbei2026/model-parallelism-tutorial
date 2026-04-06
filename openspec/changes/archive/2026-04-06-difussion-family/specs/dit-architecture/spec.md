## ADDED Requirements

### Requirement: Motivation for replacing UNet with Transformer
The notebook SHALL explain why Transformers are replacing UNets in diffusion: better scaling properties, simpler architecture, compatibility with LLM training infrastructure.

#### Scenario: Scaling comparison
- **WHEN** UNet and DiT scaling properties are compared
- **THEN** plots/tables show how DiT FLOPs scale more predictably with model size, and why this matters for training large models

### Requirement: DiT architecture walkthrough
The notebook SHALL detail the DiT architecture: patchification (image→patch tokens), position embeddings, AdaLN-Zero conditioning, and the Transformer block structure.

#### Scenario: Patchification demonstration
- **WHEN** a small image is patchified
- **THEN** code shows the patch embedding, position encoding, and how timestep/class conditioning is injected via AdaLN-Zero, with tensor shapes at each step

### Requirement: AdaLN-Zero conditioning mechanism
The notebook SHALL explain AdaLN-Zero: how it modulates layer norm parameters based on timestep and class, and why it outperforms cross-attention or in-context conditioning for diffusion.

#### Scenario: AdaLN-Zero vs alternatives
- **WHEN** conditioning mechanisms are compared
- **THEN** a comparison table shows AdaLN-Zero, cross-attention, and in-context conditioning with their tradeoffs

### Requirement: Engineering advantages of DiT
The notebook SHALL cover why DiT is preferred for video and large-scale generation: uniform compute per layer, compatibility with tensor parallelism, and easier mixed-precision training.

#### Scenario: Production considerations
- **WHEN** the engineering section is presented
- **THEN** discussion covers: why DiT is easier to parallelize than UNet (no skip connections across stages), how patch size affects quality/speed, and real-world DiT configurations (DiT-XL/2, etc.)
