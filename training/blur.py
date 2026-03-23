import os
import numpy as np
from pathlib import Path
import tifffile
from scipy.ndimage import gaussian_filter

# Define source and destination directories
source_dir = Path("/path/to/your/images")       # *** SET THIS PATH ***
output_dir = Path("/path/to/your/images_blur")  # *** SET THIS PATH ***

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Gaussian blur parameters
SIGMA_MIN = 0.5
SIGMA_MAX = 2.0

# Counter for reporting
processed_count = 0

print(f"Processing images from: {source_dir}")
print(f"Output directory: {output_dir}")
print(f"Gaussian blur sigma range: {SIGMA_MIN} - {SIGMA_MAX}\n")

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

        # Generate random sigma value between SIGMA_MIN and SIGMA_MAX
        sigma = np.random.uniform(SIGMA_MIN, SIGMA_MAX)

        # Apply Gaussian blur
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)

        # Convert back to original dtype
        if img.dtype == np.uint8:
            blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            blurred = np.clip(blurred, 0, 65535).astype(np.uint16)
        else:
            # For other dtypes, just cast back
            blurred = blurred.astype(img.dtype)

        # Create output filename with _blur suffix
        output_filename = tif_file.stem + "-blur.tif"
        output_path = output_dir / output_filename

        # Save the processed image
        tifffile.imwrite(output_path, blurred)

        processed_count += 1
        print(f"✓ Processed: {tif_file.name} -> {output_filename} (sigma={sigma:.2f})")

    except Exception as e:
        print(f"✗ Error processing {tif_file.name}: {str(e)}")

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
print(f"{'='*60}")
