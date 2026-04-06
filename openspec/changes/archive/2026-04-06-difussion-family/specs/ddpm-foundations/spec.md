## ADDED Requirements

### Requirement: Forward diffusion process demonstration
The notebook SHALL demonstrate the forward diffusion process (q(x_t|x_0)) by progressively adding Gaussian noise to a sample, showing intermediate steps visually.

#### Scenario: Step-by-step noising
- **WHEN** a 2D data sample or small image is provided
- **THEN** the notebook shows x_0, x_t at multiple timesteps, and x_T (pure noise), with the noise schedule formula and variance values at each step

### Requirement: Noise schedule comparison
The notebook SHALL implement and compare linear and cosine noise schedules, showing beta_t, alpha_bar_t curves and their effect on the noising process.

#### Scenario: Linear vs cosine schedule visualization
- **WHEN** both schedules are computed for T=1000 timesteps
- **THEN** plots show beta_t, alpha_bar_t, and SNR curves side by side, with explanation of why cosine schedule preserves more signal at early timesteps

### Requirement: Reverse process and training objective
The notebook SHALL derive and implement the simplified training objective (predict noise epsilon) from the variational lower bound.

#### Scenario: Training loop demonstration
- **WHEN** a simple denoising model is trained on 2D point data (e.g., Swiss roll)
- **THEN** the notebook shows the loss curve converging and the model learning to denoise, with step-by-step code for: sample x_0, sample t, add noise, predict noise, compute MSE loss

### Requirement: Sampling (reverse denoising)
The notebook SHALL implement DDPM ancestral sampling and visualize the denoising trajectory from x_T to x_0.

#### Scenario: Generating samples from trained model
- **WHEN** the trained denoising model runs the reverse process for T steps
- **THEN** generated samples approximate the original data distribution, with visualization of intermediate denoising steps

### Requirement: Engineering perspective on DDPM
The notebook SHALL include practical discussion of common DDPM training issues and production considerations.

#### Scenario: Debugging tips
- **WHEN** the training section completes
- **THEN** an info_box covers: typical loss curves (what good/bad looks like), the T vs quality tradeoff, and why raw DDPM is rarely used in production (slow sampling)
