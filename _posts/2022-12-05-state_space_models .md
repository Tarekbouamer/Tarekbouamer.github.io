---
title: 'State Space Models (SSM)'
date: 2022-12-05
permalink: /posts/SSM/
tags:
  - Machine Learning
  - State Space Models
  - Mamba
---

# Table of Contents

  1. [Introduction](#introduction)
  2. [References](#references)

## Introduction

### State Space Models (SSMs)

Sequence modeling is central to many machine learning tasks. A persistent challenge is modeling long-range dependencies (LRDs). Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), and Transformers have specialized variants designed to capture these dependencies, yet they often struggle to scale efficiently with very long sequences.

State space models (SSMs) have emerged as a promising alternative for sequence modeling, providing a principled way to represent system states and their transitions over time. The Structured State Space (S4) model, introduced by Gu et al. (2021), overcomes prior limitations of SSMs by efficiently capturing long-range dependencies while minimizing computational and memory bottlenecks.

### SSMs vs. Transformers

Transformers are popular in sequence modeling because their self-attention mechanisms can capture long-range dependencies. However, the quadratic complexity of self-attention limits their scalability for very long sequences. In contrast, SSMs provide a more efficient way to model long-range dependencies by combining continuous-time, recurrent, and convolutional properties.

### The Future of SSMs

The S4 model is a significant advancement in sequence modeling, offering a robust alternative to existing techniques for tasks requiring efficient handling of long-range dependencies. It provides a strong foundation for future developments in sequence modeling.

*In this post, we'll explore the key concepts behind the Structured State Space (S4) model and its applications in sequence modeling.*

## Structured State Space (S4)

<!-- <div align="center">
  <img src="/images/SSM/modeling.png" alt="SSM Modeling">
</div> -->


### State Space Models (SSMs)

<div align="center">
  <img src="/images/SSM/ssr.png" alt="SSM SSR">
  Figure 1: Block diagram representation of the linear state-space equations.
</div>


State space models (SSMs) are a class of probabilistic models that describe a system's evolution over time. They consist of two primary components:

1. **State Equation:** Models the system's internal dynamics $ \dot{x}(t) = f(x(t), u(t)) $
2. **Observation Equation:** Relates the system's internal state to observed data $ y(t) = g(x(t), u(t)) $

*we note that the SSMs are widely used in control theory and signal processing to model complex systems with hidden states such as Kalman filters and Hidden Markov Models (HMMs).*

The SSM models are defined by a learned parameters $ \{A, B, C \} $, where:

- $ A $: State transition matrix
- $ B $: Input matrix
- $ C $: Observation matrix
- $ D $: Feedforward matrix, and for the SSM model we set, $ D = 0 $

$$
\begin{aligned}
\dot{x}(t) &= Ax(t) + Bu(t)  \\
y(t) &= Cx(t)
\end{aligned}
$$


We can say that SSM model maps a 1-D input signal $ u(t) $ to an $ p\times D $ latent state $ x(t) $, which is then mapped to an $ q \times D $ output signal $ y(t) $.

### HiPPO: High-order Polynomial Projection Operators

State space representation has an exponential nature which makes the SSMs suffer from the exponential scaling of the gradient through long sequences and impractical for long-range dependencies.

To address this issue the authors leveraged HiPPO theory to allow the state $ x(t) $ to memorize the the input $ u(t) $ through the state transition matrix $ A $, named **HiPPO Matrix**. At any time $t$, the current state $x(t)$ can be used to approximately reconstruct the entire input $ u $ up to time $ t $.

$$
\text{A HiPPO matrix} = - \left\{
\begin{array}{ll}
(2n + 1)^{1/2} (2k + 1)^{1/2} && \text{if } n > k \\
n + 1 && \text{if } n = k \\
0 && \text{if } n < k
\end{array}
\right.
$$

For the MNIST benchmark the matrix improved the performance from 60 % to 98 % accuracy 🎉.

### Discrete-Time SSMs

To obtain a **sequence-to-sequence model** that maps a discrete input sequence $ u(t) $ to an output sequence $ y(t) $, we need to discretize the continuous-time SSM. This is done by introducing a step size $ \Delta $, resulting in the following discrete-time SSM:

$$
\begin{aligned}
x_k &= \bar{A} x_{k-1} + \bar{B} u_k \\
y_k &= \bar{C} x_k
\end{aligned}
$$

Using the bilinear transformation, we get:

$$
\begin{aligned}
\frac{x(t + \Delta) - x(t)}{\Delta} \approx \frac{1}{2} \left[ Ax(t + \Delta) + Bu(t + \Delta) + Ax(t) + Bu(t) \right]
\end{aligned}
$$

Rearranging and simplifying:

$$
\begin{aligned}
x(t + \Delta) - x(t) = \frac{\Delta}{2} \left[ Ax(t + \Delta) + Bu(t + \Delta) + Ax(t) + Bu(t) \right]
\end{aligned}
$$

Solving for \(x(t + \Delta)\):

$$
\begin{aligned}
x_{k+1} = \left(I - \frac{\Delta}{2} A\right)^{-1} \left( \left(I + \frac{\Delta}{2} A\right)x_k + \frac{\Delta}{2} (Bu_{k+1} + Bu_k) \right)
\end{aligned}
$$

where:

- \(x_k\) and \(x_{k+1}\) denote the state vectors at the current and next time steps, respectively.
- \(u_k\) and \(u_{k+1}\) are the control inputs.

We obtain the discrete state space representation by setting:

$$
\begin{aligned}
\bar{A} &= \left(I - \frac{\Delta}{2} A\right)^{-1} \left(I + \frac{\Delta}{2} A\right) \\
\bar{B} &= \left(I - \frac{\Delta}{2} A\right)^{-1} \Delta B \\
\bar{C} &= C
\end{aligned}
$$

and the observation matrix remains the same.

### Training SSMs: the convolutional representation

For non-recurrent SSMs, we find a connection between LTI systems and convolutional neural networks. Assuming the initial state \( x_0 = 0 \), the output of the SSM can be expressed as:

$$
\begin{array}{ccc}
k = 0 \left\{
\begin{aligned}
x_0 &= \bar{B} u_0 \\
y_0 &= \bar{C} x_0 = \bar{C} \bar{B} u_0
\end{aligned}
\right.
&
k = 1 \left\{
\begin{aligned}
x_1 &= \bar{A} x_0 + \bar{B} u_1 = \bar{A} \bar{B} u_0 + \bar{B} u_1 \\
y_1 &= \bar{C} x_1 = \bar{C} \bar{A} \bar{B} u_0 + \bar{C} \bar{B} u_1
\end{aligned}
\right.
&
k = 2 \left\{
\begin{aligned}
x_2 &= \bar{A} x_1 + \bar{B} u_2 = \bar{A}^2 \bar{B} u_0 + \bar{A} \bar{B} u_1 + \bar{B} u_2 \\
y_2 &= \bar{C} x_2 = \bar{C} \bar{A}^2 \bar{B} u_0 + \bar{C} \bar{A} \bar{B} u_1 + \bar{C} \bar{B} u_2
\end{aligned}
\right.
\end{array}
$$

the sequence can be vectorized into a convolutional form:

$$
\begin{array}{ccc}
\begin{aligned}
y_k &= \sum_{i=0}^{k} \bar{C} \bar{A}^{k-i} \bar{B} u_i \\
y_k &= \bar{C} \bar{A}^k \bar{B} u_0 + \bar{C} \bar{A}^{k-1} \bar{B} u_1 + \ldots + \bar{C} \bar{A} \bar{B} u_{k-1} + \bar{C} \bar{B} u_k \\
y_k &= \bar{K} * u
\end{aligned}
&
\begin{aligned}
\bar{K} = \begin{bmatrix}
\bar{C} \bar{A}^k \bar{B} \\
\bar{C} \bar{A}^{k-1} \bar{B} \\
\vdots \\
\bar{C} \bar{A} \bar{B} \\
\bar{C} \bar{B}
\end{bmatrix}
\end{aligned}
\end{array}
$$

where \( \bar{K} \) represents **the SSM convolutional kernel**.

## Methods

### Diagolnalization of the HiPPO Matrix

We have shown that the discrete-SSM, has a repeated multiplication of the matrix \( \bar{A} \) which requires a complexity of $ O(N^{2} L) $ and a memory of $ O(N L) $, where \( N \) is the number of states and \( L \) is the length of the sequence.

Assuming that $ x = V \hat{x} $, the state space model can be diagonalized as:

$$
\begin{aligned}
\dot{\hat{x}} &= V^{-1}AV  \hat{x}  + V^{-1} B u \\
y &= C V \hat{x}
\end{aligned}
$$

Thus, for a diagonal matrix \( A \), the state space model can be solved in $ O(N L) \log^{2}{N+L} $, as $\bar{K}$ is a Vendermonde matrix, leading to a faster computation. Unfortunately, the diagonalization of the HiPPO does not work due to numerical issues.


### Normal Plus Low-Rank (NPLR) Decomposition

Although the HiPPO matrix is not normal, it can be decomposed into a normal and a low-rank matrix. However, unlike the diagonal matrices, the sum is slow.

The authors applied three new techniques to overcome the latter issue:

$$
\begin{aligned}
\text{HiPPO} = \text{Normal} + \text{Low-Rank}
\end{aligned}
$$

where the Normal part is a diagonal matrix and the Low-Rank part is a low-rank matrix. The NPLR decomposition allows for a more efficient computation of the HiPPO matrix, reducing the complexity to $ O(N L) \log^{2}{N+L} $.


## References

[1] [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396), Albert Gu and Karan Goel and Christopher Re .