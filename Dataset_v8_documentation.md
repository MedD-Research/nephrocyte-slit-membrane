# U-Net Training Dataset Documentation — Model v8

Generated 2026-03-11 by inspecting the dataset folder and augmentation scripts.

---

## Location

All dataset files are in:

```
/Volumes/4TbSSD/00_Code_Experiments/Code056_sanja_nephrocytes_dataset_to_train_unet/datasets_2/
```

Directory structure:

```
datasets_2/
├── images/               ← 21 source images (base set, before photometric augmentation)
├── masks/                ← manual annotations for the 21 source images
├── images_blur/          ← Gaussian blur augmentation
├── images_poisson/       ← Poisson noise augmentation
├── images_vn/            ← radial vignetting augmentation
├── images_linear_gradient/ ← linear gradient overlay augmentation
├── images_multiply/      ← intensity multiplication augmentation
├── images_plus30/        ← additive brightness shift (+30) augmentation
├── images_power/         ← power/gamma transform augmentation
├── images_random_field/  ← smooth spatially-varying intensity field augmentation
├── masks_blur/ … masks_random_field/  ← corresponding masks (identical to base mask)
├── masks_properties.csv
├── test_masks_properties.csv
├── train_set/
│   ├── train_images/     ← 189 training images (assembled from the folders above)
│   └── train_masks/
└── test_set/
    ├── test_images/      ← 16 test images
    └── test_masks/
```

---

## Source images (21 tiles in `datasets_2/images/`)

These are the base confocal crops that were manually annotated.

| Image name          | Notes |
|---------------------|-------|
| A14-tile02.tif      | |
| A14-tile51.tif      | |
| A15-tile10.tif      | |
| A15-tile20.tif      | |
| A15-tile21.tif      | |
| E10-tile50.tif      | |
| E10-tile57.tif      | |
| E10-tile60.tif      | |
| E10-tile61.tif      | |
| E10-tile62.tif      | |
| E10-tile74.tif      | |
| E13-tile00.tif      | |
| E13-tile02.tif      | |
| E13-tile10.tif      | |
| PN1-tile20-2.tif    | elastic deformation variant — see note below |
| PN1-tile20-3.tif    | elastic deformation variant — see note below |
| PN1-tile20-4.tif    | elastic deformation variant — see note below |
| Е16-tile00.tif      | **Cyrillic Е** in filename — see note below |
| Е16-tile01.tif      | Cyrillic Е |
| Е16-tile02.tif      | Cyrillic Е |
| Е16-tile10.tif      | Cyrillic Е |

### Note on PN1-tile20 (special case)

The original un-deformed `PN1-tile20.tif` is **not present** in the training dataset.
Instead, three elastically deformed variants (`-2`, `-3`, `-4`) were created using
`056_codes/elastic_deformation_paired_tifs.py` (α = 50, σ = 5) and treated as
independent source images. Their corresponding masks were created by applying the same
deformation field to the manually annotated mask.

This means one original annotated tile was expanded into 3 training source images via
elastic deformation, and each of those 3 then received all 8 photometric augmentations
(total 27 training files from this one tile).

The undeformed PN1-tile20 was presumably set aside (possibly because its quality was
less suitable, or to avoid overlap between training and test sets).

### Note on Е16 filenames (Cyrillic encoding)

The `Е16-*` files use a **Cyrillic capital "Е"** (Unicode U+0415) in place of the
Latin "E" (U+0045). This is invisible when displayed in most fonts but can cause issues
in scripts that filter by prefix. All current processing scripts handle these files
correctly because they operate on all `.tif` files, but be careful if you write new
code that selects files by name prefix.

---

## Augmentation pipeline

### Photometric augmentations (8 types, applied to all training source images)

Each source image was augmented 8 ways, producing 8 additional files per tile.
All augmentations were applied to images only; the mask is identical to the source mask.

| Suffix | Script | Description | Key parameters |
|--------|--------|-------------|----------------|
| `-blur` | `blur.py` | Gaussian blur | — |
| `-pn` | `poisson.py` | Poisson noise | — |
| `-vn` | `radial_vignetting.py` | Radial vignetting | — |
| `-lg` | `linear_gradient.py` | Linear intensity gradient overlay | — |
| `-mult` | `multiply.py` | Intensity multiplication | random factor 0.5–1.8× per image |
| `-plus30` | `add_subtract.py` | Additive brightness shift | +30 intensity units |
| `-pow` | `power.py` | Power/gamma transform | random γ ∈ [0.5, 1.8] per image |
| `-rf` | `random_field.py` | Smooth spatially-varying intensity multiplier | field σ = 50, range 0.6–1.5× |

### Geometric augmentation

Elastic deformation (α = 50, σ = 5) was applied to **one tile only** (PN1-tile20),
generating 3 independent deformed versions before photometric augmentation.

Augmentation scripts for flipping (`flip_tifs.py`, `flip_no_rename.py`) and rotation
(`rotate_tiff.py`, `rotate_no_rename.py`) exist in `056_codes/` but were **not applied**
to this dataset — no flipped or rotated variants appear in the assembled training set.

---

## Training set composition (189 images)

Each of the 21 source images appears in 9 forms (1 original + 8 augmented):

```
21 source images × 9 versions = 189 training images
```

Breakdown by tile (3 PN1 elastic variants × 9 each = 27; remaining 18 tiles × 9 = 162; total = 189).

---

## Test set composition (16 images)

The test set contains 4 unique tiles with **fewer** augmentation variants than training:

| Tile | Augmentations present | Files |
|------|----------------------|-------|
| A15-tile01 | base + blur, mult, pow, vn | 5 |
| A15-tile11 | base only | 1 |
| E13-tile01 | base + blur, mult, pow, vn | 5 |
| Е16-tile11 | base + blur, mult, pow, vn (Cyrillic Е) | 5 |
| **Total** | | **16** |

Note: the test set uses only 4 of the 8 augmentation types (blur, mult, pow, vn) and does
not include pn, lg, plus30, or rf variants. A15-tile11 is unaugmented in the test set.

---

## Model trained on this dataset

- Model file: `056_v8.pth`
- Training report: `056_v8_training_output/training_report.txt`
- Best checkpoint: epoch 71, test loss 0.274
- Architecture: U-Net, encoder [64, 128, 256, 512], bottleneck 1024, batch norm, dropout 0.05
- Loss: cross-entropy (ignore_index = 255)
- Optimizer: Adam, initial LR 1×10⁻⁴, weight decay 1×10⁻⁵
- Scheduler: ReduceLROnPlateau, patience 10, factor 0.5
- Early stopping: patience 15 (stopped at epoch 86)

---

## Augmentation scripts location

```
/Volumes/4TbSSD/00_Code_Experiments/Code056_sanja_nephrocytes_dataset_to_train_unet/056_codes/
```

Relevant scripts:
- `elastic_deformation_paired_tifs.py` — elastic deformation (paired image+mask)
- `blur.py`, `poisson.py`, `radial_vignetting.py`, `linear_gradient.py`
- `multiply.py`, `add_subtract.py`, `power.py`, `random_field.py`
- `dark_noise.py` — exists but **not used** for this dataset
- `flip_tifs.py`, `flip_no_rename.py`, `rotate_tiff.py`, `rotate_no_rename.py` — exist but **not used**
