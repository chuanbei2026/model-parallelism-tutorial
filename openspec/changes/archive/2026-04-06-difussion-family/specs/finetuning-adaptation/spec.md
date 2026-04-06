## ADDED Requirements

### Requirement: ControlNet architecture and training
The notebook SHALL explain ControlNet: the zero-convolution architecture, how it adds spatial conditioning (edge, depth, pose) to a frozen base model, and practical training recipe.

#### Scenario: ControlNet walkthrough
- **WHEN** the ControlNet section is presented
- **THEN** a PlantUML diagram shows the locked/trainable branches, code demonstrates the zero-convolution mechanism, and a training recipe table covers: learning rate, data requirements, convergence time

### Requirement: LoRA for diffusion models
The notebook SHALL cover LoRA adaptation for diffusion: where to apply (attention layers, cross-attention, all linear), rank selection, and training considerations.

#### Scenario: LoRA configuration analysis
- **WHEN** LoRA is applied to a toy diffusion model
- **THEN** code shows rank decomposition, parameter count savings, and comparison of different rank values on model quality

### Requirement: DreamBooth and subject-driven generation
The notebook SHALL explain DreamBooth: the personalization objective, rare token embedding, prior preservation loss, and common overfitting traps.

#### Scenario: Overfitting demonstration
- **WHEN** DreamBooth training dynamics are simulated
- **THEN** visualizations show the learning curve, overfitting onset (loss too low), and strategies to prevent it (prior preservation, early stopping, regularization)

### Requirement: Training data pipeline engineering
The notebook SHALL cover the data engineering side: caption quality, resolution bucketing, aspect ratio handling, data augmentation choices, and their impact on final model quality.

#### Scenario: Data pipeline design
- **WHEN** the data engineering section is presented
- **THEN** a diagram shows the full data pipeline (raw images → filtering → captioning → bucketing → batching) with practical tips for each stage

### Requirement: Fine-tuning failure modes
The notebook SHALL catalog common fine-tuning failures: catastrophic forgetting, style leakage, concept bleed, and insufficient/excessive training.

#### Scenario: Failure mode guide
- **WHEN** failure modes are presented
- **THEN** a diagnostic table covers: symptom, cause, and fix for each common failure, with emphasis on practical debugging workflow
