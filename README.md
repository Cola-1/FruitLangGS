# 🍎 CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting

<p align="center">
  <img src="assets/mainless1.png" alt="CountingFruit Teaser" width="95%">
</p>

This repository contains the **official implementation** of the paper:

**CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting**  
📄 [arXiv](https://arxiv.org/abs/2506.01109) ｜ 🌐 [Project Page](https://fruitlanggs.github.io/)

---

## 🔍 Highlights

- ✅ **Real-time 3D fruit counting** with **>300 FPS** rendering speed  
- ✅ **Open-vocabulary semantic control** using **language-conditioned 3D Gaussians**  
- ✅ **Fully 3D prompt-based filtering** without per-frame supervision  
- ✅ **Robust to occlusion** and cluttered orchard environments  
- ✅ **Fast** while achieving **better accuracy**

---

## 🧾 Abstract

<div style="max-width: 800px; margin: auto; font-size: 0.9em;">

accurate fruit counting in real-world agricultural environments is a longstanding challenge due to visual occlusions, semantic ambiguity, and the high computational demands of 3d reconstruction. existing methods based on neural radiance fields suffer from low inference speed, poor generalization, and limited support for open-set semantic control. this paper presents fruitlanggs, a real-time 3d fruit counting system designed to address these limitations through a unified and explicit representation of both spatial and semantic information. fruitlanggs models a fruit scene as a collection of adaptive 3d gaussians, each augmented with compressed language features aligned to clip embeddings. these gaussians are jointly optimized to encode geometry, appearance, and prompt-resolvable semantics during training. at inference time, the system supports direct language-guided filtering of fruit regions in 3d space without requiring per-frame segmentation or multi-view fusion. to ensure high rendering efficiency, fruitlanggs incorporates an adaptive radius-based early culling strategy and a tile-level load balancing mechanism, enabling stable performance exceeding 300 fps on real orchard data. the semantically filtered gaussians are further converted into dense point clouds for robust clustering and fruit counting. experimental results demonstrate that fruitlanggs surpasses prior approaches in rendering speed and counting accuracy, establishing a novel direction for language-driven real-time neural rendering in agricultural applications.

</div>


## 🔗 Links

- 🌐 **Project Page:** [fruitlanggs.github.io](https://fruitlanggs.github.io/)  
- 📄 **arXiv Paper:** [arXiv:2506.01109](https://arxiv.org/abs/2506.01109)

---

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2025countingfruit,
  title={CountingFruit: Real-Time 3D Fruit Counting with Language-Guided Semantic Gaussian Splatting},
  author={Li, Fengze and Liu, Yangle and Ma, Jieming and Liang, Hai-Ning and Shen, Yaochun and Li, Huangxiang and Wu, Zhijing},
  journal={arXiv preprint arXiv:2506.01109},
  year={2025}
}
