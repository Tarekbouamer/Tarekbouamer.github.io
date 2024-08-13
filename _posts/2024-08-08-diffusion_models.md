---
title: 'Diffusion Models: A Comprehensive Guide'
date: 2024-08-08
permalink: /posts/DM/
tags:
  - Diffusion Models
  - Generative Models
---

## Table of Contents

1. [Introduction to Diffusion Models](#introduction-to-diffusion-models)
   - [What are Diffusion Models?](#what-are-diffusion-models)
   - [Why are They Important?](#why-are-they-important)
2. [Mathematical Foundations](#mathematical-foundations)
   - [Forward Diffusion](#forward-diffusion)
   - [Inverse Diffusion](#inverse-diffusion)
3. [References](#references)

## Introduction to Diffusion Models

### What are Diffusion Models?

Diffusion Models are a type of generative model that leverage diffusion processes to create images or other types of data from an initial noisy state.

Unlike traditional generative models like GANs (Generative Adversarial Networks) or VAEs (Variational Auto-Encoders), which generate data directly, diffusion models work by progressively refining a noisy signal until it closely matches the desired outcome.

These models operate in a latent space where noise is systematically introduced and then removed, guided by various forms of data, including textual descriptions.

We introduce the key concepts of Diffusion Models:

- **Diffusion Process**: The diffusion process refers to the gradual transformation of a noisy input into a clear, coherent output. The process can be likened to reversing the natural diffusion of particles in a medium, where the model learns to denoise the input step by step.

- **Latent Space**: Diffusion Models operate within a latent space, which is a lower-dimensional representation of the data. This allows the model to work more efficiently, reducing computational complexity while maintaining the quality of the generated output.

- **Text-to-Image Generation**: Diffusion Models can generate images based on text prompts. By using models like CLIP, which link textual and visual information, the diffusion process is guided to produce images that align with the given text descriptions.

### Why are They Important?

Diffusion Models have gained attention due to their robustness and ability to consistently generate high-quality data. They have improved upon earlier generative models by providing better control over the generation process, making them more reliable and versatile for a wide range of applications.

Compared to GANs and VAEs, Diffusion Models offer several advantages: (1) they tend to produce images with finer details and fewer artifacts, (2) the diffusion process provides more control over the generation, allowing for finer adjustments, and (3) they are faster and more computationally efficient.

## Mathematical Foundations

The mathematical foundation of Diffusion Models is based on the concept of generating data by refining a noisy input. This process is deeply rooted in probability theory, particularly in the modeling of stochastic processes.

### Forward Diffusion

The forward diffusion process is an essential component of Diffusion Models, where a clean data sample is progressively transformed into a noisier version over multiple steps.

Given a data sample \( x_0 \) with a distribution \( x_0 \sim q(x_0) \), the forward diffusion process generates a sequence of increasingly noisy versions of this data by adding Gaussian noise at each of \( T \) steps. The noise introduced at each step \( t \) is characterized by a variance parameter \( \beta_t \in [0, 1] \). The conditional distribution of \( x_t \) given \( x_{t-1} \) is defined as:

\[
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \mathbf{I}\right)
\]

Here, \( q(x_t \mid x_{t-1}) \) represents a Gaussian distribution where \( x_t \) has a mean of \( \sqrt{1-\beta_t} \cdot x_{t-1} \) and a variance of \( \beta_t \mathbf{I} \). As \( t \) increases, \( x_0 \) gradually loses its original features. In the limit as \( t \) approaches infinity \( (t \to \infty) \), \( x_T \) becomes an isotropic Gaussian distribution:

\[
x_T \sim \mathcal{N}(0, \mathbf{I})
\]

The distribution of the final noisy sample \( x_T \) given the initial data \( x_0 \) can be expressed as:

\[
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})
\]

This equation describes a Markov chain, where the distribution of \( x_t \) depends solely on the previous state \( x_{t-1} \). To simplify the notation, we define \( \alpha_t = 1 - \beta_t \) and \( \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i \). Using these definitions, we can express \( x_t \) as:

\[
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}
\]

\[
= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t \alpha_{t-1}} \bar{\epsilon}_{t-2}
\]

\[
\vdots
\]

\[
= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
\]

Thus, the distribution of \( x_t \) given \( x_0 \) can be rewritten as:

\[
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)
\]




### Inverse Diffusion

The inverse diffusion process is the core generative mechanism in Diffusion Models. If we can reverse the process \( q(x_{t-1} | x_{t})\), we will be able to re-create the true input data \( x_0 \), knowing that \( q(x_{t-1} | x_{t}) \) is also a Gaussian.

Unfortunately, the estimation of \( q(x_{t-1} | x_{t}) \) is intracable, so we need to approximate the process by using a learned model \( p_{\theta} \) to estimate the reverse process, where :

\[
   p_{\theta}(x_{t-1} | x_t) = \mathcal{N}\left(x_{t-1}; \mu_{\theta}(x_t, t), \sigma_{\theta}(x_t, t)\right)
\]

The parameters \( \mu_{\theta}(x_t, t) \) and \( \sigma_{\theta}(x_t, t) \) are learned by the model, and they represent the mean and standard deviation of the Gaussian distribution at each step \( t \).


### Training Diffusion Models

Training Diffusion Models involves optimizing the parameters of the model. Specifically, we aim to train \( \mu_{\theta} \) to predict the mean \( \mu_{t} \) of the reverse process, given by:

\[
\mu_{t} = \frac{1}{\sqrt{\alpha_{t}}}\left(x_{t} - \frac{1-\alpha_{t}}{\sqrt{1-\bar{\alpha}_{t}}} \epsilon_{t}\right)
\]

After a sequence of simplifications, which will be detailed in the next section, the loss function can be expressed as:

\[
L_{t} = \mathbb{E}_{x_{0},t,\epsilon} \left[ \frac{1}{2 ||\sigma_{\theta}(x_{t},t)||_{2}^{2}} ||u_{t} - \mu_{\theta}(x_{t},t)||_{2}^{2}\right]
\]

\[
= \mathbb{E}_{x_{0},t,\epsilon} \left[ \frac{\beta_{t}^{2}}{2 \alpha_{t} (1-\bar{\alpha}_{t}) ||\sigma_{\theta}||_{2}^{2}} ||\epsilon_{t}-\epsilon_{\theta}(\sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1-\bar{\alpha}_{t}}\epsilon_{t})||^{2} \right]
\]

Thus, instead of directly predicting the mean of the Gaussian distribution, we predict the noise \( \epsilon_{t} \) that is added to the input \( x_{0} \) to generate the output \( x_{t} \) at each step \( t \).


<!-- <div align="center">
  <img src="/images/DM/training.png" alt="Training Diffusion Models">
</div> -->



## References
