## ADDED Requirements

### Requirement: DDIM deterministic sampling
The notebook SHALL derive and implement DDIM sampling, showing how it enables deterministic generation and fewer-step sampling from a DDPM-trained model.

#### Scenario: DDPM vs DDIM comparison
- **WHEN** the same trained model generates samples using DDPM (1000 steps) and DDIM (50 steps)
- **THEN** side-by-side comparison shows DDIM produces comparable quality in 20x fewer steps, with explanation of the non-Markovian formulation

### Requirement: DPM-Solver and higher-order methods
The notebook SHALL explain DPM-Solver as an ODE solver approach, comparing first-order (DDIM-equivalent) and higher-order solvers.

#### Scenario: Solver order comparison
- **WHEN** samples are generated with 1st, 2nd, and 3rd order DPM-Solver at the same step count
- **THEN** visualization shows quality improvement with higher-order solvers, with the ODE formulation explained

### Requirement: Distillation methods
The notebook SHALL cover progressive distillation and consistency models as approaches to reduce sampling steps to 1-4.

#### Scenario: Distillation concept demonstration
- **WHEN** the distillation section is presented
- **THEN** diagrams show the progressive halving strategy (teacher→student) and consistency model training objective, with comparison table of method/steps/quality

### Requirement: Production sampler selection guide
The notebook SHALL provide engineering guidance on choosing samplers for different use cases.

#### Scenario: Decision framework
- **WHEN** the summary section is reached
- **THEN** a comparison table shows sampler/steps/quality/speed/determinism tradeoffs, with practical recommendations for real-time vs batch generation
