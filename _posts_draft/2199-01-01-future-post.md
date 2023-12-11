<!-- ---
title: 'Neural Implicit Representation'
date: 2023-11-01
permalink: /posts/NIR/
tags:
  - Neural Implicit Representation
  - 3D reconstruction
  - Deep Learning
---

## Neural Implicit Representation

### Table of Contents

  1. [Introduction](#introduction)
  2. [Definition](#definition)
  3. [Representations](#representations)
    - [2.1 Voxel](#21-voxel)
    - [2.2 Point](#22-point)
    - [2.3 Mesh](#23-mesh)
    - [2.4 Occupancy Networks](#24-occupancy-networks)
      - [2.4.1 Appearance, Geometry, and Surfaces properties](#241-appearance-geometry-and-surfaces-properties)
      - [2.4.2 Convolutional Occupancy Networks](#242-convolutional-occupancy-networks)
  4. [Mesh Extraction](#mesh-extraction)
  5. [Neural Rendering](#neural-rendering)
  6. [References](#references)
  
### Introduction

In this tutorial, we will approach the Neural Implicit Representation principals and its applications in computer vision, graphics, and  robotics.

### Conceptional Overview

Volume Rendering and View Synthesis are two techniques used in computer graphics:

### Definition

Implicit Neural Representation (*INR*) is a novel concept within machine learning and computer graphics that represents an object or scene as a continuous function, rather than an explicit surface or structure. Implicit Neural Representation aims to learn a mathematical function \( f(x, y) = 0 \) or implicit representation that can generate the desired data points.

<div align="center">
  <img src="/images/NIR/nir.png" alt="NIR">
</div>
</p>

Learning-based approaches for 3D reconstruction have gained popularity for its rich representation to 3D models, compared to the traditional Multi View Stereo (*MVS*) algorithms. Through literature, Deep learning approaches are categorized into three representations:

#### Representations

*What is a good representation ?*

<div align="center">
  <img src="/images/NIR/nir_representations.png" alt="NIR">
</div>
</p>

##### 2.1 Voxel

Voxel are easy to process by neural network and commonly used in generative 3D tasks, by discretizing  the space into a set of 3D voxel grids. However, Due to its cubic memory \(O(n^3)\), the voxels representations are limited to small resolutions of the underlying 3D grid. [2]

##### 2.2 Point

As an alternative to the voxel representation, the output can be represented as a set of 3D point clouds.However, the point representation doesn't preserve the model connectivity and topology, hence require a post-processing steps to extract 3D mesh. The point representation is also limited by the number of points, which affects the resolution of the final model. [3]

##### 2.3 Mesh

Representing the output as a set of triangles (vertices and faces) is a very complicated structure that requires reference template from the same object class. Yet, the approach is still limited by the memory requirements and the resolution of the mesh. [4]

##### 2.4 Occupancy Networks

the *Occupancy Networks* implicitly represents the 3D surface as a decision boundary of a nonlinear classifier, and for every point \(\mathbf{p} \in \mathbb{R}^3\) in the 3D space, the network predicts the probability of the point being inside the object. The occupancy function is defined as:

$$
\mathbf{o} :\mathbb{R}^3 \rightarrow [0, 1]
$$


The occupancy function is approximated by a deep neural network \(f_\theta\) with parameters \(\theta\) that takes an observation \(\mathbf{x} \in X\) as input condition (ex. image, point clouds,...), and it has a function from \(\mathbf{p} \in \mathbb{R}^3\) to \(\mathbb{R}\) as an occupancy probability.

For each input pair \((p,x)\), we can write the *occupancy network function* as:

$$
f_\theta : \mathbb{R}^3  \times X \rightarrow [0, 1]
$$

<div align="center">
  <img src="/images/NIR/nir_arch.png" alt="NIR">
</div>
</p>

The advantage of the occupancy network is a continuous representation with an infinite resolution, the representation is not restricted to a specific class as in the mesh representation and it has a low memory footprint.

To learn the parameters \(\theta\), we randomly sample 3D points (ex. \(K\)=2048) in the volume and minimize the binary cross-entropy \(BCE\) loss function:

$$
L(\theta, \psi) = \sum_{j=0}^{N} BCE(f_\theta (p_{ij}, z{i}) , o_{ij} )
$$

- In practice, we sample the 3D points uniformly inside the bounding box of the object.

#### 2.4.1 Appearance, Geometry, and Surfaces properties

The implicit representation can be extended to have more objects properties and reasoning, such as the surface lightening and the view point. The occupancy network can conditioned by the viewing direction \(v\) and light location \(l\) for any 3D point \(p\), for each input tuple \((p,v,l)\), we can write the *occupancy network function* as:

$$
f_\theta : \mathbb{R}^3  \times \mathbb{R}^3 \times \mathbb{R}^M \rightarrow [0, 1]
$$

<div align="center">
  <img src="/images/NIR/nir_light_view.png" alt="NIR">
</div>
</p>

The network encodes both an input 2D image and the corresponding 3D shape into a latent representations \(z\) and \(s\), as a conditioning to the occupancy network. The model predicts the occupancy probability for each 3D point \(p\) and the color \(c\), *Surface Light Fields*.

- The light \(l\),  denotes the light source parameters, such as the light direction, color, and intensity.

The network is trained to minimize the photometric loss function between the predicted image \(I\) and the input image \(\hat{I}\):

$$
L(I, \hat{I}) = \left \| I - \hat{I}   \right \|_1
$$

#### 2.4.2 Convolutional Occupancy Networks

> Large-scale representation learning for 3D scenes ?

Implicit Neural Representations have demonstrates a good results for small objects and small scenes, however, most of approaches fails to scale to large scenes, due to:

- The previous architectures does incorporate the local information in the observation.
- It doesn't exploit and capture the translational equivariance of the 3D scene.

The Convolutional Occupancy Networks introduces the  convolutional networks into the implicit modeling for an accurate and rich large scale 3D scenes. The convolutional network incorporates the local as well the global information, and the inductive biases to obtain a better generalization, more specifically the translation equivariance.

We process the inputs thought an encoder to extract feature embeddings, we use PointNet for input point clouds and a 3D-CNN for a input voxel.

- **Planar Encoding**:

<div align="center">
  <img src="/images/NIR/nir_plane.png" alt="NIR">
</div>
</p>

For each input point, we perform an orthographic projection onto a canonical plane, aligned with the axes of the coordinate frame, which we discretize at a resolution of H × W pixel cells.

We aggregate features projecting onto the same pixel using average
pooling, resulting in planar features with dimensionality \(H × W × d\).

- **Volume Encoding**:

<div align="center">
  <img src="/images/NIR/nir_volume.png" alt="NIR">
</div>
</p>

The volumetric encodings represents the 3D information better than a 2D planar, However, the resolution is restricted by the memory footprint.
The average pooling is performed  all over the voxel cell, resulting in a feature volume with dimensionality \(H × W × D × d\).

- **Convolutional Decoder**:

The convolutional decoder processes the resulting feature planes and feature volumes using 2D and 3D U-Net network to aggregate the local and global information, and the equivariance to translation properties in the output features, enabling structured reasoning.

- **Occupancy Prediction**:

Given the aggregated features, we predict the occupancy probability for each 3D point \(p\) by projecting each point onto the corresponding plane and query the feature vector using bilinear interpolation. For multiple planes, we sum the features of all planes. For the volume, we query the feature vector using trilinear interpolation.

For a resulting feature vector \(x\) at point \(p\), denoted as  \(\psi(p,x)\), we predict the occupancy probability using fully connected layers occupancy network, as:

$$
f_\theta : (p, \psi(p,x)) \rightarrow [0, 1]
$$

<div style="display: flex; align-items: center;">
  <div style="flex: 1;">

- In comparison to the occupancy network, the convolutional occupancy network has a better accuracy and a faster convergence.

- The convolutional occupancy network shows a good generalization and scalability for large scenes, we can use a hierarchical approach to process the scene using sliding window.  
  
  </div>
  <div style="flex: 1;">
    <img src="/images/NIR/nir_large_scene.png" alt="NIR"\>
  </div>

</div>

- In comparison to the occupancy network, the convolutional occupancy network has a better accuracy and a faster convergence.

- The convolutional occupancy network shows a good generalization and scalability for large scenes, we can use a hierarchical approach to process the scene using sliding window.

### Mesh Extraction

The mesh extraction is a post-processing step that extracts the 3D mesh from the occupancy network.

#### 3.1 Marching Cubes

The marching cubes algorithm is a method for extracting a polygonal mesh of an isosurface from a 3D scalar field. The iso-surface is formed by connecting the vertices of the cubes that are intersected by the iso-surface.

Algorithm:

- We divide the 3D space into a grid of cubes (8 vertices).
- For each cube, we evaluate the cube vertices and compare them to the threshold value \(\tau\).
- We construct a triangulation for each cube, based on the vertices that are intersected by the iso-surface.
- Based on the triangulations, We generate polygons for each cube  and we merge the polygons to form the final mesh.
- We optimize the mesh by removing the duplicated vertices and edges.

#### 3.2 Multiresolution Iso-Surface Extraction (MISE)

MISE is a method that incrementally building an octree to extract high resolution meshes from the occupancy function.

<div align="center">
  <img src="/images/NIR/nir_MISE.png" alt="NIR">
</div>
</p>

- We divide the 3D space into an initial resolution (ex. \(32^2\)), and we compute the occupancy function \(f_\theta(p,x)\) for each cell.

- We set a threshold value \(\tau\) and we mark a grid points "occupied" if the occupancy function \(f_\theta(p,x)\) is greater than the threshold.

- We subdivide the query space into 8 sub-cells and we evaluate the occupancy function \(f_\theta(p,x)\) for each cell.

- We repeat the process until we reach the desired resolution.

- At the end, we apply the marching cubes algorithm to extract an approximate iso-surface, defined by the threshold value \(\tau\): { \( \ p \in \mathbb{R}^3  \ \ | \ \ f_\theta(p, x) = \tau \) }



### Differentiable Volumetric Rendering

> Learning from images only !

Learning based 3D reconstruction methods have shown impressive results, however these methods require 3D supervision from the real world or synthetic data.

<dev>
  <img src="/images/NIR/dvr_arch.png" alt="NIR">
</dev>
</p>

*Differentiable Rendering* aims to learn 3D reconstruction from RGB images only, by using the concept of implicit representation in deriving the depth gradients.

The input image is processed with an encoder to extract latent representation \(z \in \mathbb{Z} \) as a conditioning to the occupancy network \(f_\theta\), as introduced in the Occupancy Networks. The 3D surface shape is determined by a threshold value \(\tau\), such as \(f_\theta(p, z) = \tau\).

The texture of a 3D shape can be described using a texture field \(t_\theta: \mathbb{R}^3 \times \mathbb{Z} \rightarrow \mathbb{R}^3\); which regresses the RGB color value for every point \(p \in \mathbb{R}^3 \), conditioned on the same latent representation \(z \in \mathbb{Z}\). The texture of an object is determined by the value of \(t_\theta\) at the surface \(f_\theta = \tau\).

<dev>
  <img src="/images/NIR/dvr_backpropagation.png" alt="NIR">
</dev>
</p>

We define the photometric loss function between the input image \(I\), and the rendered image \(\hat{I}\) as:

$$
L(\hat{I}, I) = \sum_{u} \left \| \hat{I_u} - I_u  \right \|_1
$$
\(u\) denotes the pixel location in the image.

1. **Volume Rendering** : is the process of creating a 2D projection of a 3D discretely sampled dataset. A volume rendering algorithm obtains the color and for every voxel in the space through which rays from the camera are casted.  The RGBα color is converted to an RGB color and recorded in the corresponding pixel of the 2D image. The process is repeated for every pixel until the entire 2D image is rendered.

2. **View Synthesis**: is the opposite of volume rendering, it involves creating a 3D view from a series of 2D images. This can be done using a series of photos that show an object from multiple angles, create a hemispheric plan of the object, and place each image in the appropriate place around the object. A view synthesis function attempts to predict the depth given a series of images that describe different perspectives of an object.



### Neural Rendering

Neural Rendering is defined as [1]:
>*Deep image or video generation approaches that enable explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure*.

Neural Rendering refers to the *mapping* that generate images by tracing a ray into the scene and taking an integral over the volume. The mapping is typically represented by a neural network, which is trained on a large set of example images. The input parameters can be a simple 2D coordinate or a 3D scene representation.

The network is trained to predict the color and density of the scene at any 3D location. The color is used to compute the contribution of the sample to the integral while the density is used to compute the next sample location.

Neural Rendering can be used for a wide range of applications, including novel view synthesis, semantic photo editing, relighting, and the creation of  avatars for virtual and augmented reality (VR/AR).

### References

[1] [State of the Art on Neural Rendering](https://arxiv.org/abs/2004.03805)
[2] [Voxnet: A 3D convolutional neural network for real-time object recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)
[3] [A point set generation network for 3D object reconstruction from a single image](https://arxiv.org/abs/1612.00603)
[4] [AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation](https://arxiv.org/abs/1802.05384)
[5] [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/pdf/1812.03828.pdf)
[6] [Marching Cubes: A High Resolution 3D Surface Construction Algorithm](https://dl.acm.org/doi/10.1145/37402.37422) -->
