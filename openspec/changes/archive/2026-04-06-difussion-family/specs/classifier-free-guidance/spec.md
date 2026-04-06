## ADDED Requirements

### Requirement: Conditional vs unconditional generation
The notebook SHALL explain how diffusion models are conditioned on labels/text, contrasting unconditional and conditional denoising.

#### Scenario: Conditional denoising demonstration
- **WHEN** a simple class-conditional model is shown
- **THEN** the notebook demonstrates how the same noise input produces different outputs for different class labels

### Requirement: Classifier guidance derivation
The notebook SHALL derive classifier guidance (Dhariwal & Nichol) showing how a separate classifier steers generation.

#### Scenario: Guided vs unguided comparison
- **WHEN** classifier guidance is applied at different scales
- **THEN** visualization shows how increasing guidance scale increases class fidelity but reduces diversity

### Requirement: Classifier-free guidance mechanism
The notebook SHALL derive CFG from the classifier guidance formula, showing the training trick (random label dropout) and inference formula: eps_guided = eps_uncond + w * (eps_cond - eps_uncond).

#### Scenario: CFG scale sweep
- **WHEN** guidance scale w varies from 1.0 to 15.0
- **THEN** plots show the quality-diversity tradeoff with concrete examples

### Requirement: Engineering aspects of CFG
The notebook SHALL cover the engineering implications: 2x compute cost (two forward passes), batched implementation, negative prompts as alternative unconditional embeddings.

#### Scenario: Batched CFG implementation
- **WHEN** the engineering section is presented
- **THEN** code shows how production systems batch conditional and unconditional passes together, with memory and compute analysis
