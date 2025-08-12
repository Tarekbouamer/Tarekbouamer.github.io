---
title: 'Foundation Models for Earth Observation (EO)'
date: 2025-08-12
permalink: /posts/foundation-models-earth-observation/
tags:
- Earth Observation
- Foundation Models
- Remote Sensing
- Multimodal Learning
- Climate Monitoring
---

<div style="display: flex; align-items: center; gap: 20px;">
    <div style="flex: 1;">
        <p>
            <em>
                Foundation Models (FMs) are transforming Earth Observation (EO) by unifying diverse satellite, environmental, and sensor datasets into powerful multimodal representations. This study surveys the state of the art, covering adaptive architectures, large-scale pretraining pipelines, and generative “any-to-any” frameworks. We examine how these advances are accelerating applications and the integration of EO into digital twins.
            </em>
        </p>
    </div>
    <div style="flex: 1; text-align: center;">
        <img src="/images/FM-EO/3d-abc-fm-overview.png" alt="Foundation Models for Earth Observation Overview" style="max-width: 100%; height: auto;">
    </div>
</div>

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Datasets for EO](#2-datasets-for-eo)
   - [Scene Classification](#scene-classification)
   - [Object Detection](#object-detection)
   - [Semantic Segmentation](#semantic-segmentation)
   - [Change Detection](#change-detection)
   - [Regression](#regression)
3. [Geospatial Foundation Models](#3-geospatial-foundation-models)
   - [Self-Supervised Learning](#self-supervised-learning)
   - [Multi-Modality](#multi-modality)
   - [Contrastive Learning](#contrastive-learning)
   - [Self-Distillation](#self-distillation)
   - [Vision-Language Modeling](#vision-language-modeling)
   - [Generative Modeling](#generative-modeling)
4. [Pre-PANGAEA Benchmarks](#4-pre-pangaea-benchmarks)
5. [PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models](#5-pangaea-a-global-and-inclusive-benchmark-for-geospatial-foundation-models)
   - [Dataset](#dataset)
   - [Modeling](#modeling)
   - [Training and Fine-Tuning](#training-and-fine-tuning)
   - [Evaluation Protocol](#evaluation-protocol)
   - [Remarks](#remarks)  

## 1. Introduction

Earth Observation (EO) Foundation Models are redefining remote sensing and geo-spatial analytics by unifying diverse data sources, including optical and radar imagery, climate records, and elevation, into a unified representation, enabling generalization across tasks, resolutions, sensor modalities, geographic regions, and temporal scales.

Such models could deliver scalable and reliable solutions for tracking environmental change, supporting disaster response, improving climate monitoring, and advancing predictive Earth system science.

In this blog, we are exploring the latest advancements in Foundation Models for Earth Observation and their potential impact through the lens of available benchmarks.

---

## 2. Datasets for EO

### Scene Classification

- Often focuses on land cover, early datasets include the UC-Merced and WHU-RS19, and MillionAID which consists of RGB imagery.

- EuroSAT focuses on Sentinel-2 images, while BigEarthNet incorporates both Sentinel-1 and Sentinel-2 for multi-label land cover classification task.

- Beyond land cover, METER-ML focuses on methane source mapping, CV4A Kenya for crop type identification, BrazilDAM for dam detection, and ForestNet for deforestation monitoring.

### Object Detection

- The task of locating and identifying objects within an image is crucial for many EO applications. Notable datasets, such as fMoW (Functional Map of the World), provide 62 object categories on high-resolution satellite imagery, compared to the 20 classes on the multi-resolution dataset DIOR.

- Ships, cars, planes, and oil storage tanks are among the key objects of interest in these datasets.

### Semantic Segmentation

- This task involves classifying each pixel in an image into a predefined category. Datasets like ISPRS 2D Vaihingen and Potsdam provide multi-modal data for semantic segmentation, while LoveDA covers Aerial imagery and Five-Billion Pixels Gaofen-2 satellite imagery (China).

- Datasets such as DeepGlobe are on global scale, FLAIR-1 introduces temporal and spectral information from optical satellite imagery for the French Landscape.

- Other datasets are for cloud cover segmentation, building footprint detection and agriculture monitoring, etc.

### Change Detection

- Crucial for flood monitoring, tracking urban development, assessing natural disasters by taking 2 images as inputs. SZTAKI AirChange focuses on identifying changes from aerial image pairs, across seasons and years. SECOND uses multi-temporal aerial images for land cover variation and semantic class changes.

- Levir-CD is a very high resolution dataset for urban building changes, and OSCD focuses on cities across the globe using Sentinel-2 images.

### Regression

- This task involves predicting continuous values and physics from satellite imagery, notable datasets such as SatBird models the birds species distribution across various habitats, ClimSim for climate forecasting and 3DCD for elevation change prediction.

---

## 3. Geospatial Foundation Models

Geospatial foundation models (GFMs) refer to large scale, self-supervised models designed to process geospatial data and support wide range of downstream applications by training of large amount of data to learn rich representations to generalize across different tasks and domains.

In comparison to Vision-Language Foundation Models (VLFM), GFMs are specifically tailored for satellite imagery as primary modality, dominated by Multi-Spectral and SAR imagery.

As we have presented earlier notable datasets, we will now discuss briefly the pre-PANGAEA models and benchmarks.

- **SatlasNet**: it employs Swin-Transformer as backbone trained on large annotated dataset with 7 label types (costly annotation). This is why many GFMs leverage Self-Supervised Learning (SSL) techniques to reduce the reliance on labeled data.

### Self-Supervised Learning

- **SSL4EO**: introduces globally distributed datasets based on Landsat-8 and Sentinel 1/2 imagery, these datasets were also used to train Masked Autoencoders (MAE), DINO and MoCo, etc.

- **RingMO**: uses Masked Image Modeling (MIM) techniques to learn to reconstruct masked regions in satellite images. **RingMO-Sense** extends this approach by incorporating spatiotemporal evolution, and **RingMO-SAM** makes use of Segment Anything Model (SAM) for image segmentation task.

- **CtxMIM**: combines MIM and Siamese networks for improved context feature learning.

- **SatMAE**: employs Masked Autoencoders (MAE) for self-supervised training with multi-spectral and spatiotemporal location embedding, its variant **SatMAE++** enhanced the features using multi-level feature extraction.

- **Cross-scale MAE**: enforces cross scale consistency through contrastive and generative loss.

- **Scale-MAE**: introduces resolution-aware positional encoding to learn features at different scales.

### Multi-Modality

- **OmniSat**: leverages a multi-modal fusion approach by including spatially aligned VHR images with SAR time series.

- **DOFA**: leverages wavelength as a unifying parameter across various modalities.

### Contrastive Learning

- **GASSL**: leverages spatially aligned images over time to construct temporal positive pairs.

- **MATTERS**: learns the invariance to illumination and viewing angle to achieve consistency in terms of appearance and texture.

- **CROMA**: combines contrastive learning with masked autoencoders (MAE) to learn both uni-modal and multi-modal representations from aligned radar and optical imagery.

### Self-Distillation

- **DINO-MM**: uses Teacher-Student framework for self-distillation on concatenated SAR-optical images as raw inputs feed to the two encoders.

- **DeCUR**: decouples the representation learning by separating the inter and intra-modal features based on Barlow Twins.

### Vision-Language Modeling

- **RemoteCLIP**: first FM to learn a representation by aligning remote sensing images with textual descriptions.

### Generative Modeling

- **DiffusionSAT** and **MetaEearth** generate realistic EO images all over the Earth based on the diffusion models.

---

## 4. Pre-PANGAEA Benchmarks

- **PhilEO Bench**: evaluate only through the lens of Sentinel-2.
- **GEO-Bench** and **SustainBench**: includes data from different sensors, however they cover land observation tasks primarily, and focuses on mono-temporal applications.
- **FoMo-Bench** and **Crop Type Bench**: restrict their scope application to forest and agriculture monitoring.

---

## 5. PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models

PANGAEA proposes a global and inclusive benchmark for geospatial foundation models (GFMs) to evaluate different models, including UNet and ViT, across multiple domains, assess their effectiveness and limitations under various scenarios, and provide insights for future research. It is designed to be extensible for new datasets, models, tasks, and evaluation protocols.

It addresses major gaps in existing benchmarks, including limited diversity, narrow image resolution ranges, restricted sensor modality coverage, and geographic bias. The benchmark provides a standardized, globally representative evaluation framework to ensure fair comparison.

### Dataset

- It covers 11 datasets across urban, agriculture, marine, forest, and disaster domains.

- It supports tasks including marine debris segmentation, building damage mapping through change detection, semantic segmentation, and pixel-level regression, enabling applications in urban monitoring, agricultural analysis, marine assessment, and forest environmental monitoring.

- The datasets span spatial resolutions from 1.5 m to 30 m per pixel and include uni-temporal, bi-temporal, and multi-temporal satellite image series.

- Multiple sensor types are supported, including optical RGB, multi-spectral imagery, and synthetic aperture radar (SAR), ensuring varied spatial, spectral, and temporal characteristics.

### Modeling

- PANGAEA focuses on approach, impact, and reproducibility, evaluating diverse strategies such as multi-modal contrastive learning, masked image modeling (MIM), supervised training, and generative approaches.

- It is model-agnostic, assessing architectures from transformers to CNNs under self-supervised, supervised, contrastive, and generative paradigms.  

### Training and Fine-Tuning

- Training strategies include full fine-tuning, low-label fine-tuning, and cross-resolution evaluation to assess label efficiency and scalability.

- Baseline models include UNet and ViT-B/16 trained from scratch, with GFMs such as AnySat, TerraMind, and Copernicus-FM evaluated for direct comparison.

### Evaluation Protocol

- The evaluation protocol ensures fairness across different pre-training conditions by incorporating spectral band matching, standardized multi-temporal data handling, and a consistent UPerNet decoder for all models.  

### Remarks

- No single GFM performs best across all tasks: spectral-rich pretraining benefits agriculture and forest applications, high-resolution pretraining improves urban change detection, and cross-geographic generalization remains challenging.  

- The benchmark and its code are openly released to support reproducible and extensible model evaluation for the Earth Observation community.  
