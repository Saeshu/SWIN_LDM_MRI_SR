##🧠 3D MRI Super-Resolution using Swin-Enhanced Latent Diffusion

📌 Overview

This project explores 3D MRI super-resolution using latent diffusion models (LDMs), with a focus on preserving anatomical structure during reconstruction.

While diffusion models achieve strong generative performance, they often produce structurally inconsistent or blurry outputs in medical imaging. This work investigates whether improving feature representations—via adaptive convolutions and Swin attention—can reduce these failure modes.

❗ Problem Statement

MRI super-resolution is challenging due to:

- Loss of fine anatomical detail during downsampling
- Sensitivity to noise and reconstruction artifacts
- Difficulty in maintaining global structural consistency across slices

Standard convolutional models are limited in capturing long-range dependencies, leading to:

- Blurry reconstructions
- Structural distortions
- Poor anatomical alignment

##🧠 Core Idea

This project explores whether enhancing latent representations can improve reconstruction quality.

Specifically:

- Use a 3D autoencoder to learn a compact latent space
- Train a latent diffusion model for reconstruction
- Incorporate Swin-based attention to capture long-range dependencies
-  Use adaptive convolutions to improve feature flexibility

##Central Question:
Can better latent representations reduce structural degradation in generative MRI reconstruction?

⚙️ Method
Architecture Components

- 3D Autoencoder (CNN-based)
- Encodes MRI volumes into a latent representation
- Latent Diffusion Model (DDPM / DDIM)
- Learns to denoise latent representations across timesteps
- Swin Attention Module
- Captures long-range spatial dependencies across 3D volumes
- Adaptive Convolutions
- Allows dynamic feature extraction based on input structure

🤔 Why Swin Attention?

Convolutional layers are inherently local and struggle with global structure.

Swin attention:

- Enables hierarchical self-attention
- Captures long-range anatomical relationships
- Improves cross-slice consistency in 3D data

This is critical in MRI, where structures extend throughout the entire volume.

🧪 Experiments
- Training LDMs on 3D MRI volumes
- Evaluating reconstruction quality across diffusion timesteps
- Comparing structural fidelity with/without attention
- Observing latent space behavior during denoising
📊 Results


<!--
Include:

Low-resolution input
Ground truth high-resolution
Model output
Failure cases
Key Observations
Structural details degrade at higher diffusion timesteps
Outputs remain blurry despite low reconstruction loss
Suggests limitations in latent representation quality, not just decoding
🔬 Observations & Insights
Failure modes are often structural, not just pixel-level
Latent spaces may not explicitly encode anatomical consistency
Global context (via attention) helps but does not fully resolve degradation
Indicates a need for better representation learning, not just architecture scaling
🔗 Connection to Research Direction
-->
This project serves as a testbed for studying:

- How failure modes relate to learned representations
- Whether structural inconsistencies can be used as signals
- How latent spaces can be shaped for better generative behavior

