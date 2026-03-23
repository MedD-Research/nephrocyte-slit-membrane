import os
import numpy as np
from pathlib import Path
import tifffile
from scipy.ndimage import gaussian_filter

# Define source and destination directories
source_dir = Path("/path/to/your/images_plus30")       # *** SET THIS PATH *** (input: brightness-shifted images)
output_dir = Path("/path/to/your/images_random_field") # *** SET THIS PATH ***

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Random field parameters
MIN_MULTIPLIER = 0.6  # Minimum intensity multiplier
MAX_MULTIPLIER = 1.5  # Maximum intensity multiplier
FIELD_SIGMA = 50      # Smoothness of random field (larger = smoother)

# Counter for reporting
processed_count = 0

print(f"Processing images from: {source_dir}")
print(f"Output directory: {output_dir}")
print(f"Random field intensity range: {MIN_MULTIPLIER}x - {MAX_MULTIPLIER}x")
print(f"Field smoothness (sigma): {FIELD_SIGMA}\n")


def create_random_field(height, width, min_val=0.7, max_val=1.3, sigma=50):
    """
    Generate a smooth random intensity field using Gaussian filtering.

    Args:
        height (int): Image height
        width (int): Image width
        min_val (float): Minimum multiplier value
        max_val (float): Maximum multiplier value
        sigma (float): Gaussian smoothing sigma (larger = smoother field)

    Returns:
        np.ndarray: Smooth random field with shape (height, width)
    """
    # Generate random noise
    random_noise = np.random.rand(height, width)

    # Apply Gaussian smoothing to create smooth variations
    smooth_field = gaussian_filter(random_noise, sigma=sigma)

    # Normalize to [0, 1] range
    smooth_field = (smooth_field - smooth_field.min()) / (smooth_field.max() - smooth_field.min())

    # Scale to desired range [min_val, max_val]
    intensity_field = min_val + (max_val - min_val) * smooth_field

    return intensity_field


# Get all .tif and .tiff files, excluding hidden files and ._ files
tif_files = [
    f for f in source_dir.rglob("*")
    if f.suffix.lower() in ['.tif', '.tiff']
    and not f.name.startswith('.')
    and not f.name.startswith('._')
]

print(f"Found {len(tif_files)} valid .tif/.tiff files")
print(f"{'='*60}\n")

# Process each image
for tif_file in tif_files:
    try:
        # Read the image
        img = tifffile.imread(tif_file)

        # Create random smooth field with same dimensions as image
        random_field = create_random_field(
            img.shape[0],
            img.shape[1],
            min_val=MIN_MULTIPLIER,
            max_val=MAX_MULTIPLIER,
            sigma=FIELD_SIGMA
        )

        # Apply random field by multiplying image with the field
        modified = img.astype(np.float32) * random_field

        # Clip values to valid range and convert back to original dtype
        if img.dtype == np.uint8:
            modified = np.clip(modified, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            modified = np.clip(modified, 0, 65535).astype(np.uint16)
        else:
            modified = modified.astype(img.dtype)

        # Create output filename with _rf suffix
        output_filename = tif_file.stem + "-rf.tif"
        output_path = output_dir / output_filename

        # Save the processed image
        tifffile.imwrite(output_path, modified)

        processed_count += 1
        print(f"✓ Processed: {tif_file.name} -> {output_filename}")

    except Exception as e:
        print(f"✗ Error processing {tif_file.name}: {str(e)}")

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
print(f"{'='*60}")
