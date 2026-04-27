"""
Nephrocyte Fluorescence Image Preprocessor — FAST version
==========================================================
Uses OpenCV for morphology/CLAHE/Gaussian (~10-30× faster)
and multiprocessing for parallel image processing.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import tifffile
import cv2

# ─────────────────────────────────────────────
#  USER SETTINGS
# ─────────────────────────────────────────────

INPUT_FOLDER    = "/Volumes/4TbSSD/nephrocytes_lukas/roi"
OUTPUT_FOLDER   = "/Volumes/4TbSSD/nephrocytes_lukas/roi_preprocessed"
REFERENCE_IMAGE = None

# ─────────────────────────────────────────────
#  PREPROCESSING PARAMETERS
# ─────────────────────────────────────────────

BG_RADIUS       = 50
CLAHE_KERNEL_SIZE = 64
CLAHE_CLIP_LIMIT  = 0.03
GAUSS_SIGMA     = 0.7
EXTENSIONS      = {".tif", ".tiff", ".png"}

# Parallelism: set to 1 for sequential, or None for auto (cpu_count)
MAX_WORKERS     = None  # None = use all CPU cores

# ─────────────────────────────────────────────
#  FAST PIPELINE (OpenCV)
# ─────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load image as float32 in [0, 1]."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img.ndim == 3:
        # Extract green channel (index 1 in RGB; but OpenCV loads BGR → index 1 is still green)
        img = img[:, :, 1]

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= np.iinfo(np.uint16).max if img.max() > 255 else 255.0
    return img


def subtract_background_fast(img: np.ndarray, radius: int) -> np.ndarray:
    """
    Background subtraction via morphological opening.

    OpenCV's morphologyEx is implemented in optimized C++ with SIMD —
    typically 10-50× faster than skimage.morphology.opening for large kernels.
    """
    # OpenCV requires kernel diameter to be odd
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # OpenCV morphological opening: erode then dilate
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    corrected = img - background
    corrected = np.clip(corrected, 0, None)
    cmax = corrected.max()
    if cmax > 0:
        corrected /= cmax
    return corrected


def apply_clahe_fast(img: np.ndarray, kernel_size: int, clip_limit: float) -> np.ndarray:
    """
    CLAHE using OpenCV — significantly faster than skimage.exposure.equalize_adapthist.

    OpenCV's CLAHE expects uint8 or uint16, so we convert, apply, convert back.
    """
    # Convert to uint16 for better precision than uint8
    img_u16 = np.clip(img * 65535, 0, 65535).astype(np.uint16)

    # clip_limit in OpenCV is absolute (applied to histogram bins in the tile).
    # skimage's clip_limit ∈ [0,1] maps roughly to OpenCV's clipLimit * (tile_area / n_bins).
    # A reasonable conversion: OpenCV clipLimit ≈ clip_limit * tile_pixels / 256
    # For simplicity, a clipLimit of 2.0–4.0 in OpenCV ≈ 0.01–0.03 in skimage.
    tile_area = kernel_size * kernel_size
    # Convert skimage-style clip_limit to OpenCV-style
    cv_clip = clip_limit * tile_area / 256
    cv_clip = max(1.0, cv_clip)  # OpenCV minimum

    clahe = cv2.createCLAHE(
        clipLimit=cv_clip,
        tileGridSize=(
            max(1, img.shape[1] // kernel_size),
            max(1, img.shape[0] // kernel_size),
        )
    )
    result = clahe.apply(img_u16)
    return result.astype(np.float32) / 65535.0


def apply_gaussian_fast(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur using OpenCV (already fast, but OpenCV is still ~2-3× faster)."""
    if sigma <= 0:
        return img
    # ksize=0 means OpenCV auto-computes kernel size from sigma
    return cv2.GaussianBlur(img, (0, 0), sigma)


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Simple histogram matching (CDF-based), no skimage dependency needed."""
    src_u16 = np.clip(source * 65535, 0, 65535).astype(np.uint16)
    ref_u16 = np.clip(reference * 65535, 0, 65535).astype(np.uint16)

    src_hist, _ = np.histogram(src_u16.ravel(), bins=65536, range=(0, 65535))
    ref_hist, _ = np.histogram(ref_u16.ravel(), bins=65536, range=(0, 65535))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Build lookup table
    lut = np.interp(src_cdf, ref_cdf, np.arange(65536))
    result = lut[src_u16]
    return (result / 65535.0).astype(np.float32)


def save_image(img: np.ndarray, path: Path):
    """Save as 8-bit TIFF."""
    img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    tifffile.imwrite(str(path), img_u8)


def preprocess_single(img: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Full pipeline — fast version."""
    img = subtract_background_fast(img, BG_RADIUS)
    img = apply_clahe_fast(img, CLAHE_KERNEL_SIZE, CLAHE_CLIP_LIMIT)
    img = apply_gaussian_fast(img, GAUSS_SIGMA)

    if reference is not None:
        img = histogram_match(img, reference)
        cmax = img.max()
        if cmax > 0:
            img /= cmax

    return img


# ─────────────────────────────────────────────
#  WORKER FUNCTION (for multiprocessing)
# ─────────────────────────────────────────────

def _process_one(args: tuple) -> str:
    """Process a single image — called in worker processes."""
    fpath_str, out_dir_str, ref_data = args
    fpath = Path(fpath_str)
    out_dir = Path(out_dir_str)

    img = load_image(fpath)
    result = preprocess_single(img, ref_data)

    out_name = fpath.stem + "_preprocessed.tif"
    save_image(result, out_dir / out_name)
    return fpath.name


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    input_dir  = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ])

    if not image_files:
        print(f"[ERROR] No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")

    # Load and preprocess reference once
    reference = None
    if REFERENCE_IMAGE is not None:
        ref_path = Path(REFERENCE_IMAGE)
        if ref_path.exists():
            print(f"Loading reference: {ref_path.name}")
            reference = load_image(ref_path)
            reference = subtract_background_fast(reference, BG_RADIUS)
            reference = apply_clahe_fast(reference, CLAHE_KERNEL_SIZE, CLAHE_CLIP_LIMIT)
        else:
            print(f"[WARNING] Reference not found: {ref_path}")

    workers = MAX_WORKERS or max(1, multiprocessing.cpu_count() - 1)
    print(f"Processing with {workers} workers...\n")

    # Build work items — reference array is serialized once via pickle
    work = [
        (str(f), str(output_dir), reference)
        for f in image_files
    ]

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, w): w[0] for w in work}
        for future in as_completed(futures):
            done += 1
            try:
                name = future.result()
                print(f"  [{done}/{len(image_files)}] ✓ {name}")
            except Exception as e:
                print(f"  [{done}/{len(image_files)}] ✗ {Path(futures[future]).name}: {e}")

    print(f"\nDone! {len(image_files)} images → {output_dir}")


if __name__ == "__main__":
    main()