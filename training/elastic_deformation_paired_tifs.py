import os
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates, gaussian_filter
import tifffile
from tqdm import tqdm
import traceback

# Define source and destination directories
BASE_DIR = Path("/path/to/your/dataset/")  # *** SET THIS PATH ***
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
OUTPUT_IMAGES_DIR = BASE_DIR / "elastic_images"
OUTPUT_MASKS_DIR = BASE_DIR / "elastic_masks"
OUTPUT_PAIRED_DIR = BASE_DIR / "elastic_paired"

# Elastic deformation parameters
ALPHA = 50  # Deformation strength
SIGMA = 5   # Smoothness of deformation

# Bit depth conversion
BIT_CONVERSION_FACTOR = 257  # 8-bit to 16-bit: 255 * 257 = 65535

# Processing options
SAVE_PAIRED_TIFF = True  # Save 2-channel TIFF for inspection
SKIP_EXISTING = False    # Set to True to skip already processed pairs




def find_valid_pairs(images_dir, masks_dir):
    """
    Find and validate image-mask pairs.

    Matching logic:
    - image_name.tif → image_name_seg.tif or image_name_seg.tiff
    - image_name.tiff → image_name_seg.tif or image_name_seg.tiff

    Validation checks:
    - Both files exist
    - Dimensions match
    - Image is 8-bit (uint8)
    - Mask is numeric array

    Args:
        images_dir (Path): Directory containing images
        masks_dir (Path): Directory containing masks

    Returns:
        tuple: (valid_pairs, skipped_info)
            - valid_pairs: list of tuples (image_path, mask_path, dimensions)
            - skipped_info: dict with skip reasons
    """
    valid_pairs = []
    skipped_info = {
        'no_mask': [],
        'dimension_mismatch': [],
        'wrong_dtype': [],
        'read_error': []
    }

    # Find all image files
    image_files = []
    for pattern in ['*.tif', '*.tiff']:
        for img_path in images_dir.glob(pattern):
            if not img_path.name.startswith('.') and not img_path.name.startswith('._'):
                image_files.append(img_path)
    image_files = sorted(image_files)

    print(f"Found {len(image_files)} image files")

    for img_path in image_files:
        # Construct expected mask paths (try both .tif and .tiff)
        base_name = img_path.stem  # Remove extension
        mask_path_tif = masks_dir / f"{base_name}_seg.tif"
        mask_path_tiff = masks_dir / f"{base_name}_seg.tiff"

        # Check which mask exists
        if mask_path_tif.exists():
            mask_path = mask_path_tif
        elif mask_path_tiff.exists():
            mask_path = mask_path_tiff
        else:
            skipped_info['no_mask'].append(img_path.name)
            continue

        # Load and validate dimensions and dtypes
        try:
            # Load image
            img = tifffile.imread(img_path)

            # Load mask
            mask = tifffile.imread(mask_path)

            # Check dimensions
            if img.shape != mask.shape:
                skipped_info['dimension_mismatch'].append(
                    (img_path.name, f"Image: {img.shape}, Mask: {mask.shape}")
                )
                continue

            # Check image is 8-bit or 16-bit
            if img.dtype not in [np.uint8, np.uint16]:
                skipped_info['wrong_dtype'].append(
                    (img_path.name, f"Expected uint8 or uint16, got {img.dtype}")
                )
                continue

            # Check mask is numeric
            if not np.issubdtype(mask.dtype, np.number):
                skipped_info['wrong_dtype'].append(
                    (mask_path.name, f"Non-numeric dtype: {mask.dtype}")
                )
                continue

            # Valid pair found
            valid_pairs.append((img_path, mask_path, img.shape))

        except Exception as e:
            skipped_info['read_error'].append((img_path.name, str(e)))

    return valid_pairs, skipped_info


def elastic_deformation_multichannel(image, alpha, sigma, random_state=None):
    """
    Apply elastic deformation to multi-channel image with same displacement field.

    This is the KEY function that ensures paired deformation.

    Process:
    1. Generate displacement fields (dx, dy) based on first channel dimensions
    2. Apply same displacement to all channels
    3. Use map_coordinates with order=1 (bilinear) for images
    4. Use map_coordinates with order=0 (nearest) for masks to preserve labels

    Args:
        image (np.ndarray): Image array
            - Shape: (height, width) for single channel
            - Shape: (height, width, channels) for multi-channel
        alpha (float): Deformation strength
        sigma (float): Smoothness of deformation
        random_state (np.random.RandomState): Random state for reproducibility

    Returns:
        np.ndarray: Deformed image with same shape as input
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    # Handle both 2D and 3D arrays
    is_multichannel = image.ndim == 3
    if not is_multichannel:
        image = image[..., np.newaxis]  # Add channel dimension

    height, width, n_channels = image.shape

    # Generate random displacement fields (ONCE for all channels)
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha

    # Create meshgrid of coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply displacement
    indices = np.array([
        np.reshape(y + dy, (-1,)),
        np.reshape(x + dx, (-1,))
    ])

    # Deform each channel with same displacement
    deformed_channels = []
    for c in range(n_channels):
        channel = image[:, :, c]

        # Use different interpolation for different channel types
        # Channel 0 (image): order=1 (bilinear)
        # Channel 1 (mask): order=0 (nearest neighbor) to preserve integer labels
        interpolation_order = 1 if c == 0 else 0

        deformed = map_coordinates(
            channel,
            indices,
            order=interpolation_order,
            mode='reflect'
        )
        deformed_channels.append(deformed.reshape(height, width))

    # Stack channels
    result = np.stack(deformed_channels, axis=-1)

    # Remove channel dimension if input was 2D
    if not is_multichannel:
        result = result[..., 0]

    return result


def process_pair(img_path, mask_path, output_img_dir, output_mask_dir,
                 output_paired_dir=None, alpha=50, sigma=5, random_state=None,
                 save_paired=False, skip_existing=False):
    """
    Process a single image-mask pair through the complete pipeline.

    Pipeline steps:
    1. Load 8-bit image
    2. Convert image to 16-bit (multiply by 257)
    3. Load mask and ensure uint16 format
    4. Stack into 2-channel array (height, width, 2)
    5. Apply elastic deformation with same displacement field
    6. Split channels
    7. Save outputs

    Args:
        img_path (Path): Path to image file
        mask_path (Path): Path to mask file
        output_img_dir (Path): Output directory for deformed images
        output_mask_dir (Path): Output directory for deformed masks
        output_paired_dir (Path, optional): Output directory for 2-channel TIFFs
        alpha (float): Deformation strength
        sigma (float): Smoothness
        random_state: Random state
        save_paired (bool): Whether to save 2-channel TIFF
        skip_existing (bool): Skip if output already exists

    Returns:
        dict: Processing result with status and info
    """
    base_name = img_path.stem

    # Define output paths
    output_img_path = output_img_dir / f"{base_name}-ed.tif"
    output_mask_path = output_mask_dir / f"{base_name}-ed_seg.tif"

    # Check if outputs exist
    if skip_existing:
        if output_img_path.exists() and output_mask_path.exists():
            return {
                'status': 'skipped',
                'reason': 'outputs_exist',
                'files': (output_img_path.name, output_mask_path.name)
            }

    try:
        # Step 1: Load image (8-bit or 16-bit)
        img = tifffile.imread(img_path)

        # Step 2: Convert to 16-bit if needed
        if img.dtype == np.uint8:
            img_16bit = (img.astype(np.uint32) * BIT_CONVERSION_FACTOR).astype(np.uint16)
        else:
            img_16bit = img.astype(np.uint16)

        # Step 3: Load mask and convert to uint16
        mask = tifffile.imread(mask_path)
        mask_16bit = mask.astype(np.uint16)

        # Step 4: Stack into 2-channel array (height, width, 2)
        paired = np.stack([img_16bit, mask_16bit], axis=-1)

        # Step 5: Apply elastic deformation
        deformed_paired = elastic_deformation_multichannel(
            paired,
            alpha=alpha,
            sigma=sigma,
            random_state=random_state
        )

        # Step 6: Split channels
        deformed_img = deformed_paired[:, :, 0].astype(np.uint16)
        deformed_mask = deformed_paired[:, :, 1].astype(np.uint16)

        # Step 7: Save outputs
        tifffile.imwrite(output_img_path, deformed_img)
        tifffile.imwrite(output_mask_path, deformed_mask)

        # Optional: Save 2-channel TIFF for inspection
        if save_paired and output_paired_dir:
            paired_path = output_paired_dir / f"{base_name}_paired_ed.tif"
            # Transpose to (channels, height, width) for proper multi-channel TIFF
            # Fiji/ImageJ expects channels first: (C, H, W)
            paired_chw = np.transpose(deformed_paired, (2, 0, 1))
            tifffile.imwrite(paired_path, paired_chw, imagej=True, metadata={'axes': 'CYX'})

        return {
            'status': 'success',
            'files': (output_img_path.name, output_mask_path.name),
            'shape': deformed_img.shape
        }

    except Exception as e:
        return {
            'status': 'error',
            'reason': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    """
    Main execution function with progress tracking and reporting.
    """
    print("="*70)
    print("PAIRED ELASTIC DEFORMATION PIPELINE")
    print("="*70)
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Masks directory: {MASKS_DIR}")
    print(f"Output images: {OUTPUT_IMAGES_DIR}")
    print(f"Output masks: {OUTPUT_MASKS_DIR}")
    print(f"Parameters: alpha={ALPHA}, sigma={SIGMA}")
    print("="*70)
    print()

    # Create output directories
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PAIRED_TIFF:
        OUTPUT_PAIRED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Find valid pairs
    print("Step 1: Finding and validating image-mask pairs...")
    valid_pairs, skipped_info = find_valid_pairs(IMAGES_DIR, MASKS_DIR)

    print(f"\nFound {len(valid_pairs)} valid pairs")
    print("\nSkipped files summary:")
    print(f"  - No matching mask: {len(skipped_info['no_mask'])}")
    print(f"  - Dimension mismatch: {len(skipped_info['dimension_mismatch'])}")
    print(f"  - Wrong data type: {len(skipped_info['wrong_dtype'])}")
    print(f"  - Read errors: {len(skipped_info['read_error'])}")

    if not valid_pairs:
        print("\nNo valid pairs to process. Exiting.")
        return

    print("\n" + "="*70)
    print("Step 2: Processing pairs...")
    print("="*70)

    # Initialize random state if needed
    random_state = None  # Each pair gets different random deformation

    # Process each pair
    results = {
        'success': [],
        'skipped': [],
        'error': []
    }

    for img_path, mask_path, dimensions in tqdm(valid_pairs, desc="Processing pairs", unit="pair"):
        result = process_pair(
            img_path=img_path,
            mask_path=mask_path,
            output_img_dir=OUTPUT_IMAGES_DIR,
            output_mask_dir=OUTPUT_MASKS_DIR,
            output_paired_dir=OUTPUT_PAIRED_DIR if SAVE_PAIRED_TIFF else None,
            alpha=ALPHA,
            sigma=SIGMA,
            random_state=random_state,
            save_paired=SAVE_PAIRED_TIFF,
            skip_existing=SKIP_EXISTING
        )

        results[result['status']].append((img_path.name, result))

    # Final report
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {len(results['success'])} pairs")
    print(f"Skipped (already exist): {len(results['skipped'])} pairs")
    print(f"Failed with errors: {len(results['error'])} pairs")

    # Print error details
    if results['error']:
        print("\nErrors encountered:")
        for name, result in results['error'][:5]:
            print(f"  - {name}: {result['reason']}")
        if len(results['error']) > 5:
            print(f"  ... and {len(results['error']) - 5} more errors")

    print("\nOutput locations:")
    print(f"  - Deformed images: {OUTPUT_IMAGES_DIR}")
    print(f"  - Deformed masks: {OUTPUT_MASKS_DIR}")
    if SAVE_PAIRED_TIFF:
        print(f"  - 2-channel TIFFs: {OUTPUT_PAIRED_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
