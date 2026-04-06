## ADDED Requirements

### Requirement: Pixel-space vs latent-space motivation
The notebook SHALL explain why diffusion in pixel space is impractical at high resolution and how VAE compression solves this.

#### Scenario: Compute comparison
- **WHEN** pixel-space and latent-space FLOPs are compared for 512x512 generation
- **THEN** a table shows the 8x-64x reduction in spatial dimensions and corresponding compute/memory savings

### Requirement: VAE encoder/decoder architecture
The notebook SHALL walk through the VAE component: encoder (image→latent), decoder (latent→image), KL regularization, and common training pitfalls (KL collapse, blurriness).

#### Scenario: VAE encoding demonstration
- **WHEN** a small image is encoded and decoded through a toy VAE
- **THEN** the latent representation is visualized, reconstruction quality is shown, and the KL loss term is explained

### Requirement: UNet architecture for latent diffusion
The notebook SHALL detail the UNet architecture used in Stable Diffusion: ResNet blocks, self-attention, cross-attention for text conditioning, timestep embedding, skip connections.

#### Scenario: Architecture walkthrough
- **WHEN** the UNet section is presented
- **THEN** a PlantUML or matplotlib diagram shows the full UNet structure with labeled components, and code demonstrates a simplified version with real tensors

### Requirement: Text conditioning via cross-attention
The notebook SHALL explain how text embeddings (from CLIP/T5) are injected into the UNet via cross-attention layers.

#### Scenario: Cross-attention demonstration
- **WHEN** text and image features are combined
- **THEN** code shows the cross-attention mechanism: Q from image features, K/V from text features, with attention map visualization

### Requirement: Stable Diffusion full pipeline
The notebook SHALL walk through the complete Stable Diffusion inference pipeline: text encoding → latent noise → iterative denoising → VAE decode.

#### Scenario: End-to-end architecture diagram
- **WHEN** the pipeline section is presented
- **THEN** a PlantUML diagram shows all components and their connections, with data shapes annotated at each stage
