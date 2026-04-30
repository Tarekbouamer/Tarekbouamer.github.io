---
title: "DeepPCF-MVS: Deep Plane Estimation and Filtering for Complete Multi-View Stereo"
excerpt: "DeepPCF-MVS: Deep Plane Estimation and Filtering for Complete Multi-View Stereo<br /><img src='/images/DeepPCF-MVS.png'>"
collection: portfolio
---

## Abstract

DeepPCF-MVS is a 3D vision project focused on more complete and reliable multi-view stereo reconstruction.

Abstract Multi-View Stereo (MVS)-based 3D reconstruction is a major topic in computer vision for which a vast number of methods have been proposed over the last decades showing impressive visual results. Long-since, benchmarks like Middlebury numerically rank the individual methods considering accuracy and completeness as quality attributes. While the Middlebury benchmark provides low-resolution images only, the recently published ETH3D and Tanks and Temples benchmarks allow for an evaluation of high-resolution and large-scale MVS from natural camera configurations. This benchmarking reveals that still only few methods can be used for the reconstruction of large-scale models. We present an effective pipeline for large-scale 3D reconstruction which extends existing methods in several ways: (i) We introduce an outlier filtering considering the MVS geometry and make use of machine-learned confidences for filtering. (ii) To avoid incomplete models from local matching methods we propose a plane completion method based on growing superpixels allowing a generic generation of high-quality 3D models. We show further improvements by utilizing plane detections from a deep neural network in addition to superpixel segmentation masks to generate improved plane-based segmentation masks. (iii) Finally, we use deep learning for a subsequent filtering of outliers in segmented sky areas. We give experimental evidence on benchmarks that our contributions improve the quality of the 3D model and our method is state-of-the-art in high-quality 3D reconstruction from high-resolution images or large image sets.

The work reports competitive results on ETH3D and Tanks and Temples, showing improved completeness, accuracy, and F1-score, and was submitted to IJCV.

## Links

* [Paper](https://tarekbouamer.github.io/files/DeepPCF-MVS.pdf)
