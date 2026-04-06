## ADDED Requirements

### Requirement: Video as temporal extension of image diffusion
The notebook SHALL explain how video generation extends image diffusion with temporal modeling: 3D noise, temporal attention, and frame consistency.

#### Scenario: Memory scaling analysis
- **WHEN** video dimensions (frames × height × width × channels) are analyzed
- **THEN** calculations show the memory explosion: a 16-frame 512x512 video requires 16x the activations, with strategies to manage this

### Requirement: Temporal attention architectures
The notebook SHALL compare approaches: 3D UNet (joint spatial-temporal), factored attention (separate spatial and temporal), and temporal transformer layers.

#### Scenario: Architecture comparison
- **WHEN** the three approaches are presented
- **THEN** diagrams and tables compare compute/memory/quality tradeoffs, with explanation of why factored attention dominates in practice

### Requirement: Production video model analysis
The notebook SHALL analyze the architecture of major video generation models (Sora, CogVideo, HunyuanVideo) at an engineering level.

#### Scenario: Architecture breakdown
- **WHEN** Sora/CogVideo/HunyuanVideo are discussed
- **THEN** PlantUML diagrams show their key architectural choices: spacetime patchification, temporal compression, text conditioning approach, and training strategy

### Requirement: Progressive training strategy
The notebook SHALL explain the image→video progressive training approach: pretraining on images, then fine-tuning with temporal layers on video data.

#### Scenario: Training pipeline
- **WHEN** the progressive training section is presented
- **THEN** a diagram shows the training stages with discussion of: why train on images first, how to initialize temporal layers, and data quality requirements

### Requirement: Temporal consistency engineering
The notebook SHALL cover engineering techniques for maintaining temporal consistency: temporal attention, motion modeling, and common artifacts (flickering, object drift).

#### Scenario: Artifact analysis
- **WHEN** temporal consistency is discussed
- **THEN** descriptions and diagrams illustrate common failure modes and the engineering solutions used in production systems
