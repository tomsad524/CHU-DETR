# CHU-DETR

Official code release of **"Infrared-Visible Dual-Modal Object Detection Transformer with Cross-Hierarchical Attention and U-like Gated Interaction"**.

> **CHU-DETR** (Cross-Hierarchical Attention and U-like Gated Interaction DETR) is an end-to-end dual-modal object detection framework that leverages complementary infrared and visible information for robust detection in complex unstructured environments (e.g., autonomous driving, security surveillance, and UAV aerial imaging). Built upon the DINO detection transformer, CHU-DETR introduces three key innovations: Cross-Hierarchical Attention Feature Fusion (CHAF), U-like Gated Feature Interaction (UGFI), and Position-Supervised Loss (PSL).

<p align="center">
  <img src="figs/framework.png" alt="CHU-DETR Framework" width="90%">
</p>

---

## Highlights

- **CHAF** — Cross-Hierarchical Attention Feature Fusion: establishes a top-down context transmission path with multi-branch (1×1, 5×5, 7×7) cross-attention for precise extraction and robust fusion of dual-modal complementary features across multiple scales, effectively overcoming cross-modal physical parallax
- **UGFI** — U-like Gated Feature Interaction: employs pixel-level gating weights to dynamically filter non-target noise and synergize deep global semantics with shallow high-frequency details, significantly enhancing the spatial localization accuracy of tiny and occluded objects during the decoding stage
- **PSL** — Position-Supervised Loss: uses solely the position metric (IoU) to supervise the classification scores of positive samples, resolving the optimization ambiguity in bipartite matching between classification and localization

---

## News

- **2026.07**: The core model architecture, evaluation scripts, and dataset processing tools are publicly available. Full training pipeline and pre-trained model weights will be released upon paper acceptance. Stay tuned.

---

## Results

All results obtained with ResNet-50 backbone, trained for 12 epochs on a single NVIDIA RTX 4090.

| Dataset | Backbone | mAP50 | mAP75 | mAP |
|:---|:---:|:---:|:---:|:---:|
| **FLIR** | ResNet50 | 87.0 | 48.2 | **50.1** |
| **LLVIP** | ResNet50 | **98.4** | 81.7 | 69.0 |
| **KAIST** | ResNet50 | **78.4** | 26.2 | **36.3** |
| **GIR** | ResNet50 | **92.5** | 69.3 | **60.6** |

> Our method outperforms state-of-the-art dual-modal detectors on all four benchmarks. See the paper (Section 4.2) for detailed per-dataset comparisons with DMFFNet, DAMSDet, SQR-DETR, MRT-DETR, LCAFNet, InfoCalib, and other recent methods.

---

## Installation

### Requirements

| Component | Version |
|:---|:---|
| Linux | Ubuntu 20.04+ |
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.10 |
| CUDA | ≥ 11.3 |

### Setup

```bash
# Clone the repository
git clone https://github.com/tomsad524/CHU-DETR.git
cd CHU-DETR

# Install dependencies
pip install -r requirements.txt

# Build the Deformable Attention CUDA operator
cd models/dino/ops
bash make.sh
cd ../../..
```

If the CUDA operator build fails, verify that your `CUDA_HOME` is correctly set and that your `nvcc` version matches your PyTorch CUDA version.

---

## Data Preparation

All datasets need to be in COCO format. The expected directory structures are shown below.

### FLIR (Aligned)

Download the aligned version from [FLIR-aligned](https://github.com/nicolalandro/FLIR-aligned). The dataset contains 5,142 IR-RGB image pairs (4,129 training / 1,013 testing) covering person, car, and bicycle.

```
FLIR_aligned_coco/
├── annotations/
│   ├── train.json
│   └── val.json
├── train_RGB/
├── train_thermal/
├── val_RGB/
└── val_thermal/
```

### LLVIP

Download from [LLVIP](https://github.com/bupt-ai-cz/LLVIP). The dataset contains 15,488 strictly aligned IR-RGB pairs (12,025 training / 3,463 testing) designed for pedestrian detection in low-light surveillance scenarios.

```
LLVIP/
├── coco_annotations/
│   ├── LLVIP_train.json
│   └── LLVIP_test.json
├── visible/train/
├── visible/test/
├── infrared/train/
└── infrared/test/
```

### KAIST

Download the raw KAIST multispectral pedestrian dataset, then convert to COCO format using the provided script. The dataset contains 95,328 image pairs (7,601 training / 2,252 testing) captured in driving environments. We use the improved annotations from [Zhang et al. (ICCV 2019)](https://github.com/luzhang16/KAIST-Pedestrian-Detection).

```bash
python tools/convert_kaist_to_coco.py \
    --kaist_root /path/to/raw_kaist/ \
    --output_root /path/to/KAIST_COCO/
```

After conversion:

```
KAIST_COCO/
├── annotations/
│   ├── train.json
│   └── val.json
├── train_RGB/
├── train_thermal/
├── val_RGB/
└── val_thermal/
```

### GIR

The GIR dataset is constructed from the RGBT210 video sequences by Li et al. Frames are extracted at specific intervals and manually filtered to remove severe motion blur and extreme thermal noise. The dataset comprises 5,105 image pairs (4,084 training / 1,021 testing) covering five classes: person, dog, car, bicycle, and motorcycle.

```bash
python tools/convert_gir_to_coco.py \
    --gir_root /path/to/raw_gir/ \
    --output_root /path/to/GIR_COCO/
```

After conversion:

```
GIR_COCO/
├── annotations/
│   ├── train.json
│   └── val.json
├── train_RGB/
├── train_thermal/
├── val_RGB/
└── val_thermal/
```

---

## Model Architecture

The dual-modal framework is built on DINO with two independent ResNet-50 backbones for IR and RGB feature extraction. The core modules are defined under `models/dino/`.

| File | Component | Description |
|:---|:---|:---|
| `models/dino/dino.py` | Main Framework | End-to-end dual-modal DETR integrating CHAF, UGFI, and PSL |
| `models/dino/deformable_transformer.py` | Transformer | 6-layer encoder + 6-layer decoder with UGFI gated feature interaction |
| `models/dino/backbone.py` | Dual Backbone | Weight-independent ResNet-50 branches for IR and RGB |
| `models/dino/attention.py` | CHAF Attention | Multi-branch (1×1, 5×5, 7×7) cross-attention for multi-scale fusion |
| `models/dino/matcher.py` | Bipartite Matcher | Hungarian matcher with PSL-integrated cost computation |
| `models/dino/dn_components.py` | Denoising | Contrastive denoising training components for accelerated convergence |

Configuration files for different DINO variants (4-scale, 5-scale, Swin backbone, ConvNeXt backbone) are in `config/DINO/`.

| Config | Backbone | Scales |
|:---|:---|:---|
| `DINO_4scale.py` | ResNet-50 | 4 | ← used in our experiments |
| `DINO_5scale.py` | ResNet-50 | 5 |
| `DINO_4scale_swin.py` | Swin Transformer | 4 |
| `DINO_4scale_convnext.py` | ConvNeXt | 4 |
| `DINO_5scale_swin.py` | Swin Transformer | 5 |
| `DINO_5scale_bis.py` | ResNet-50 | 5 (bis) |

---

## Evaluation

Pre-trained model weights will be released upon paper acceptance. Once the checkpoint is available:

```bash
python main_eval.py -c config/DINO/DINO_4scale.py \
    --dataset_file flir_fusion \
    --coco_path /path/to/FLIR_aligned_coco/ \
    --resume ./checkpoints/flir_checkpoint.pth \
    --eval
```

Replace `flir_fusion` with `llvip_fusion`, `kaist_fusion`, or `gir_fusion` for the other datasets. Evaluation reports mAP50, mAP75, and mAP (COCO-style, averaged over IoU thresholds 0.50–0.95 with step 0.05).

---

## Note on Training Code

The complete training pipeline (`main.py`, `engine.py`) and pre-trained model weights for all four datasets (FLIR, LLVIP, KAIST, GIR) are ready and will be released in full upon acceptance of the manuscript. This is a common practice to protect the originality of the contribution during peer review. We appreciate your understanding.

---

## Project Structure

```
CHU-DETR/
├── config/DINO/              # Model configuration files
├── datasets/                 # Dataset loading, transforms, and evaluation
│   └── torchvision_datasets/ # COCO-format dataset wrappers (including dual-modal)
├── models/dino/              # Core model architecture
│   └── ops/                  # Deformable attention CUDA operator
├── tools/                    # Dataset conversion utilities
├── figs/                     # Framework illustration
├── main.py                   # Training entry point (to be released upon acceptance)
├── engine.py                 # Training engine (to be released upon acceptance)
├── main_eval.py              # Evaluation entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # Apache 2.0
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{hou2026chudetr,
  title   = {Infrared-Visible Dual-Modal Object Detection Transformer with
             Cross-Hierarchical Attention and U-like Gated Interaction},
  author  = {Hou, Zhiqiang and Wu, Xingping and Zhao, Jialu and
             Ma, Sugang and Liu, Yang and Lu, Ruitao and Xi, Jianxiang},
  journal = {Infrared Physics \& Technology},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This codebase is built upon [DINO](https://github.com/IDEA-Research/DINO) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). We thank the authors for their excellent work.
