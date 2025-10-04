# Enhanced Semantic Segmentation with Attention-Based Decoder Frameworks

Welcome to my bachelor's thesis repository! This project explores advanced decoder architectures for semantic segmentation, focusing on improving output resolution and accuracy compared to traditional approaches.

## 🎯 Project Overview

The field of computer vision is deeply intertwined with neural networks, as they are the most prominent approach for solving tasks such as **semantic segmentation**. Semantic segmentation is a pixel-wise classification of images, wherein each pixel is classified into the class it represents on the surface of the image.

### The Challenge

Due to the inherent complexity of this task, many networks output segmentation maps of lower resolution than the input image to reduce computational burden. These lower-size masks then typically get resized to full image resolution to compare them to the ground truth map. This practice can lead to a **loss in performance**, especially in applications where small details around the edges of objects are of importance.

## 🚀 Our Solution

This work focuses on designing **three novel decoder frameworks** that can output segmentation maps with enhanced resolution and accuracy:

### 🏗️ Architecture Frameworks

-   **U-Att-Large**: Uses four-level features from the standard SegFormer MiT-B0 backbone, outputting segmentation maps at full input resolution
-   **U-Att-Full**: Combined with a modified MiT-B0-Full backbone, receiving six-level features for enhanced detail capture
-   **U-Att-Small**: Uses MiT-B0 encoder but produces maps at 1/4 resolution (like standard SegFormer) while maintaining improved accuracy

Our decoder frameworks leverage **attention-based transformer blocks** from the SegFormer encoder and combine them with components inspired by **U-Net** architecture.

## 📊 Key Results

We evaluated our models on the **Cityscapes dataset** with impressive improvements over the baseline SegFormer All-MLP decoder:

### MiT-B0 Backbone Results

-   **U-Att-Large**: **66.43% mIoU** (vs. 62.96% baseline) - _+3.47% improvement_
-   **U-Att-Small**: **66.20% mIoU** with **reduced computational complexity**

### MiT-B0-Full Backbone Results

-   **U-Att-Full**: **65.20% mIoU** (vs. 62.09% baseline) - _+3.11% improvement_

## 📁 Repository Structure

```
├── code/                    # Main implementation files
│   ├── backbone/           # Modified backbone architectures
│   ├── better/            # Enhanced decoder implementations
│   ├── configs/           # Training and model configurations
│   ├── att_configs/       # Attention mechanism configurations
│   └── training/          # Training scripts and utilities
├── thesis.pdf            # Complete thesis document
└── README.md             # This file
```

---

_This research contributes to advancing semantic segmentation by demonstrating that attention-based decoder frameworks can significantly improve both accuracy and computational efficiency compared to traditional MLP-based approaches._
