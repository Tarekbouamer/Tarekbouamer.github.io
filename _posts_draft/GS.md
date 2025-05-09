---
title: 'A Comprehensive Study for Gaussian Splatting'
date: 2025-05-09
permalink: /posts/gaussian-splatting/
tags:
  - Gaussian Splatting
  - Neural Rendering
  - Differentiable Rendering
  - Computer Graphics
  - Machine Learning
---

<div style="display: flex; align-items: center; gap: 20px;">
    <div style="flex: 1;">
        <p>
            <em>
                Gaussian Splatting is a cutting edge technique for real time neural rendering that models scenes using explicit 3D Gaussians. It offers an alternative to neural implicit representations (NIR) and NeRFs. This study provides a rigorous, structured exploration of its mathematical foundations, differentiable rasterization pipeline, and key advancements that are redefining Gaussian-based scene representation.
            </em>
        </p>
    </div>
    <div style="flex: 1; text-align: center;">
        <img src="/images/gaussian-splatting/overview.png" alt="Gaussian Splatting Overview" style="max-width: 100%; height: auto;">
    </div>
</div>

## Table of Contents

1. [Introduction to Gaussian Splatting](#1-introduction-to-gaussian-splatting)  
2. [3D Gaussian Representation](#2-3d-gaussian-representation)  
3. [Differentiable Gaussian Rasterization](#3-differentiable-gaussian-rasterization)  
4. [Optimization and Density Control](#4-optimization-and-density-control)  
5. [Speed and Optimization Improvements](#5-speed-and-optimization-improvements)  
6. [Compact Representations and Compression](#6-compact-representations-and-compression)  
7. [Rendering and Shading Enhancements](#7-rendering-and-shading-enhancements)  
8. [Discussion: Comparisons, Tradeoffs, and Limitations](#8-discussion-comparisons-tradeoffs-and-limitations)  
9. [Conclusion and Future Work](#9-conclusion-and-future-work)  

## 1. Introduction to Gaussian Splatting

**Gaussian Splatting (GS)** is a recent breakthrough in real-time neural rendering that represents 3D scenes as unstructured sets of anisotropic Gaussian ellipsoids. In contrast to NeRFs, which rely on implicit volumetric representations encoded by multi-layer perceptrons (MLPs), GS employs an explicit, differentiable formulation that supports efficient rasterization and fast convergence.

While NeRF and its variants have achieved impressive photorealism in novel view synthesis, they exhibit notable limitations: slow training and inference, high memory usage, and inefficiencies in modeling empty space. Gaussian Splatting addresses these challenges through:

- **Explicit 3D Gaussians** for spatial representation  
- **Alpha compositing** and **spherical harmonics** for view-dependent appearance  
- **Tile-based differentiable rasterization** for real-time performance on modern GPUs  

Introduced by [Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), Gaussian Splatting replaces neural field representations with a compact, interpretable formulation that enables fast optimization, real-time rendering, and high-fidelity view synthesis.

### Point-Based Rendering

To understand the foundations of Gaussian Splatting, it is important to first examine point-based rendering a general framework for novel view synthesis that predates neural radiance fields.

In this paradigm, a 3D scene is modeled as a set of discrete spatial primitives such as particles, surfels, or ellipsoids. These points are projected onto the image plane and blended to form continuous surfaces and radiance fields in screen space.

Classical point-based methods rely on simple splatting of fixed radius points, which often leads to visual artifacts such as aliasing, discontinuities, and poor reconstruction of view dependent effects.

To address these issues, modern techniques introduce several enhancements:

- **Anisotropic splats** that align with surface geometry and adapt their shape per viewpoint  
- **Opacity-aware alpha blending** to simulate volumetric transmittance and occlusion  
- **Spherical Harmonics (SH)** to encode view-dependent appearance

Let $\mathcal{G} = \{g_i\}_{i=1}^N$ denote a set of $N$ point primitives, where each $g_i$ is defined by parameters $(\mu_i, \Sigma_i, \alpha_i, c_i, SH_i)$. The rendered pixel color $C$ along a camera ray $\mathbf{r}$ is computed via front-to-back alpha compositing:

$$
C(\mathbf{r}) = \sum_{i=1}^N T_i \alpha_i c_i
\quad \text{where} \quad
T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

This formulation is mathematically equivalent to the volumetric rendering equation used in NeRF, but instead of stochastic sampling in a continuous field, it operates on projected 2D Gaussian splats.

**Gaussian Splatting** builds directly on this foundation by:

- Using **learned, anisotropic 3D Gaussians** as spatial primitives  
- Introducing **differentiable rasterization** with visibility-aware compositing  
- Applying **interleaved optimization and adaptive densification** to improve scene coverage

In the next section, we formalize the structure and mathematical projection of these Gaussians, which serve as the core rendering primitives in GS.
