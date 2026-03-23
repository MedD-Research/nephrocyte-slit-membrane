# Training — U-Net Model for Slit Diaphragm Segmentation

This folder contains the training notebook and augmentation scripts used to train the U-Net model for slit membrane segmentation in *Drosophila* nephrocytes.

---

## Contents

| File | Description |
|------|-------------|
| `unet_training_056_v2_mac.ipynb` | Training notebook — defines the U-Net architecture, loads training/test data, and runs the training loop |
| `add_subtract.py` | Additive brightness shift (used for `_plus30` augmentation) |
| `blur.py` | Gaussian blur augmentation |
| `elastic_deformation_paired_tifs.py` | Elastic deformation of image+mask pairs (α=50, σ=5) |
| `linear_gradient.py` | Linear intensity gradient overlay |
| `multiply.py` | Intensity multiplication augmentation |
| `poisson.py` | Poisson noise augmentation |
| `power.py` | Power/gamma transform augmentation |
| `radial_vignetting.py` | Radial vignetting augmentation |
| `random_field.py` | Spatially smooth random intensity field augmentation |

---

## Augmentation workflow

The augmentation scripts were applied to the original 21 annotated images to generate a larger training set. The dependency between scripts is:

```
original images
    │
    ├── blur.py             → images_blur/
    ├── multiply.py         → images_multiply/
    ├── power.py            → images_power/
    ├── poisson.py          → images_poisson/
    ├── add_subtract.py     → images_plus30/
    │       │
    │       ├── linear_gradient.py   → images_linear_gradient/
    │       └── random_field.py      → images_random_field/
    │
    └── radial_vignetting.py → images_vn/

elastic_deformation_paired_tifs.py  (applied to selected images + their masks)
```

Note: `linear_gradient.py` and `random_field.py` take the brightness-shifted images (`images_plus30/`) as input, not the originals.

All augmentation scripts process images only. Masks were copied unchanged for all augmentations except elastic deformation, which deforms image and mask simultaneously to preserve pixel-level correspondence.

The augmented dataset and train/test split used for training are available on Zenodo (see main README).

---

## Training notebook

The training notebook (`unet_training_056_v2_mac.ipynb`) was developed and run on Apple Silicon Mac (MPS backend). It will also run on CPU on any platform.

**Environment:** use the `unet_inference` conda environment (see `requirements_inference.txt` in the root folder).

**To retrain the model:**
1. Download the training dataset from Zenodo
2. Set the paths to `train_set/` and `test_set/` in the notebook
3. Run all cells

---

## Setting paths

Each augmentation script has two path variables near the top marked with `*** SET THIS PATH ***`. Edit these before running to point to your local image and output folders.

`add_subtract.py` uses command-line arguments instead of hardcoded paths:

```bash
python add_subtract.py --source_dir /path/to/images --output_dir /path/to/images_plus30 --value 30 --suffix _plus30
```
