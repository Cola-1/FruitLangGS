# CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting

This repository contains the official implementation of the paper:

**CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting**  
📄 [arXiv:2506.01109](https://arxiv.org/abs/2506.01109)

## 🔍 Highlights

- Real-time 3D fruit counting with >300 FPS rendering speed.  
- Open-vocabulary semantic control using language-conditioned 3D Gaussians.  
- Fully 3D prompt-based filtering without per-frame supervision.  
- Accurate counting under occlusion and cluttered orchard environments.  
- Outperforms FruitNeRF in speed and achieves comparable or better accuracy.

## 🧾 Abstract

> Accurate fruit counting in real-world agricultural environments is a long-standing challenge due to visual occlusions, semantic ambiguity, and the high computational demands of 3D reconstruction. Existing methods based on neural radiance fields suffer from low inference speed, limited generalization, and lack support for open-set semantic control. This paper presents FruitLangGS, a real-time 3D fruit counting framework that addresses these limitations through spatial reconstruction, semantic embedding, and language-guided instance estimation. FruitLangGS first reconstructs orchard-scale scenes using an adaptive Gaussian splatting pipeline with radius-aware pruning and tile-based rasterization for efficient rendering. To enable semantic control, each Gaussian encodes a compressed CLIP-aligned language embedding, forming a compact and queryable 3D representation. At inference time, prompt-based semantic filtering is applied directly in 3D space, without relying on image-space segmentation or view-level fusion. The selected Gaussians are then converted into dense point clouds via distribution-aware sampling and clustered to estimate fruit counts. Experimental results on real orchard data demonstrate that FruitLangGS achieves higher rendering speed, semantic flexibility, and counting accuracy compared to prior approaches, offering a new perspective for language-driven, real-time neural rendering in agricultural robotics.
