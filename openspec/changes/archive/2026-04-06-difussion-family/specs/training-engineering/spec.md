## ADDED Requirements

### Requirement: Noise schedule engineering
The notebook SHALL cover practical noise schedule design: linear, cosine, shifted schedules, and how to tune schedules for different data types (images, audio, video).

#### Scenario: Schedule impact on training
- **WHEN** different noise schedules are applied to the same dataset
- **THEN** visualizations show the SNR curve differences and their impact on generated quality, with guidance on when to use which schedule

### Requirement: Loss weighting strategies
The notebook SHALL implement and compare loss weighting approaches: uniform, SNR-based, min-SNR-gamma, and P2 weighting, with their effect on generation quality.

#### Scenario: Weighting comparison
- **WHEN** training curves with different loss weightings are compared
- **THEN** plots show convergence speed and final quality differences, with explanation of why uniform weighting over-emphasizes high-noise timesteps

### Requirement: Prediction target comparison
The notebook SHALL compare epsilon-prediction, x0-prediction, and v-prediction parameterizations with their engineering tradeoffs.

#### Scenario: Prediction target analysis
- **WHEN** the three parameterizations are presented
- **THEN** code shows each formulation, a table compares their properties (stability, dynamic range, SNR sensitivity), and guidance explains when to use which

### Requirement: EMA and its subtleties
The notebook SHALL explain EMA (Exponential Moving Average) for diffusion training: decay schedule, when to start, warmup, and the power function schedule.

#### Scenario: EMA ablation
- **WHEN** EMA configurations are compared (decay rates, start steps)
- **THEN** simulated training curves show how bad EMA settings degrade quality, with practical recommendations

### Requirement: Mixed precision training pitfalls
The notebook SHALL cover FP16/BF16 training for diffusion models: where precision matters (noise prediction near t=0), gradient scaling, and common numerical issues.

#### Scenario: Precision failure modes
- **WHEN** mixed precision issues are discussed
- **THEN** examples show: FP16 underflow in small noise predictions, loss scaling strategies, and why BF16 is preferred over FP16 for diffusion

### Requirement: Common training failures and debugging
The notebook SHALL catalog common training failures with diagnostics: loss plateaus, mode collapse, blurriness, color shift, and training divergence.

#### Scenario: Failure diagnosis guide
- **WHEN** each failure mode is presented
- **THEN** description includes: what it looks like, root causes, and fixes, formatted as a diagnostic reference table
