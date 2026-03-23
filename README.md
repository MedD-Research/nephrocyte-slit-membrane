# Slit Diaphragm Width Quantification Pipeline

A semi-automated image analysis pipeline for quantifying slit diaphragm width in *Drosophila* nephrocytes from confocal fluorescence microscopy images. The pipeline combines manual region-of-interest (ROI) delineation in Fiji/ImageJ with a custom-trained U-Net convolutional neural network and the Local Thickness algorithm.

---

## Pipeline Overview

```
Confocal image (SNS staining)
        │
        ▼
[Step 1] 01_ROI_extraction.ijm       — Fiji: manually delineate nephrocytes,
                                        export cropped ROIs and binary cell masks
        │
        ▼
[Step 2] 02_unet_inference.ipynb     — Python: segment slit membrane and protein
                                        clusters using pre-trained U-Net model
        │
        ▼
[Step 3] 03_channel_splitting.ipynb  — Python: separate membrane and cluster
                                        channels from U-Net output
        │
        ▼
[Step 4] 04_local_thickness.ijm      — Fiji: measure slit membrane width using
                                        Local Thickness plugin
        │
        ▼
[Step 5] 05_extract_thickness_data.ipynb  — Python: bin thickness values and
                                             export to CSV
        │
        ▼
[Step 6] 06_visualization_statistics.ipynb — Python: calculate group statistics
                                              and generate plots
```

---

## Requirements

### Fiji/ImageJ
- [Fiji](https://fiji.sc/) with the **Local Thickness** plugin installed
  - Install via: *Help → Update → Manage update sites → BoneJ* (includes Local Thickness)

### Python — two environments

The pipeline uses two separate Python environments. We recommend using [conda](https://docs.conda.io/) to manage them.

**Step 2 — U-Net inference** (`02_unet_inference.ipynb`): Python 3.10, see `requirements_inference.txt`

```bash
conda create -n unet_inference python=3.10
conda activate unet_inference
# Install PyTorch (see https://pytorch.org/get-started/locally/ for your platform)
# Apple Silicon Mac:
conda install pytorch==2.5.1 torchvision==0.20.1 -c pytorch
# CPU-only (Windows/Linux):
conda install pytorch==2.5.1 torchvision==0.20.1 cpuonly -c pytorch
# Then install remaining packages:
pip install -r requirements_inference.txt
```

> **Apple Silicon note:** The inference notebook runs on Apple Silicon Macs using the MPS (Metal Performance Shaders) GPU backend. No NVIDIA GPU is required. It will also run on CPU on any platform, though more slowly.

**Steps 3, 5, 6 — post-processing and visualization** (`03_channel_splitting.ipynb`, `05_extract_thickness_data.ipynb`, `06_visualization_statistics.ipynb`): Python 3.11, see `requirements_analysis.txt`

```bash
conda create -n analysis python=3.11
conda activate analysis
pip install -r requirements_analysis.txt
```

---

## Model Weights

The pre-trained U-Net model weights (`056_v8.pth`) are available on Zenodo:

> [Zenodo DOI — link to be added upon publication]

Download `056_v8.pth` and place it in the same folder as `02_unet_inference.ipynb` before running inference.

### Model details
- Architecture: U-Net with encoder stages [64, 128, 256, 512], bottleneck 1024 channels, batch normalization, dropout 0.05
- Training: 205 manually annotated confocal images (189 training / 16 test)
- Output: 2-channel binary mask — channel 0: slit membrane, channel 1: protein clusters
- Best checkpoint: epoch 71, test loss 0.274

---

## Usage

### Step 1 — ROI extraction (Fiji)
1. Open your confocal image in Fiji
2. Manually delineate each nephrocyte using the ROI Manager
3. Run `01_ROI_extraction.ijm`
4. When prompted, select output folders for ROI images and binary cell masks

### Step 2 — U-Net inference (Python)
1. Open `02_unet_inference.ipynb` in Jupyter
2. Set the paths to your ROI images, cell masks, and model weights file
3. Run all cells — outputs are 2-channel binary segmentation masks

### Step 3 — Channel splitting (Python)
1. Open `03_channel_splitting.ipynb`
2. Set the path to the U-Net output masks
3. Run all cells — saves separate membrane and cluster mask images

### Step 4 — Local Thickness (Fiji)
1. Open `04_local_thickness.ijm` in Fiji
2. Set `inputDir` and `outputDir` at the top of the script to your local paths
3. Run the macro — processes all TIF files in the input folder

### Step 5 — Extract thickness data (Python)
1. Open `05_extract_thickness_data.ipynb`
2. Set the path to the Local Thickness output folder
3. Run all cells — exports binned thickness distributions as CSV files

### Step 6 — Visualization and statistics (Python)
1. Open `06_visualization_statistics.ipynb`
2. Set the path to the CSV folder and define your experimental groups
3. Run all cells — generates histograms and exports summary statistics for GraphPad Prism

---

## Training Data

The annotated image dataset used for model training is available on Zenodo:

> [Zenodo DOI — link to be added upon publication]

See `Dataset_v8_documentation.md` for a detailed description of the dataset and augmentation strategy.

---

## Citation

If you use this pipeline in your work, please cite:

> [Citation to be added upon publication]

---

## License

MIT License. See `LICENSE` for details.
