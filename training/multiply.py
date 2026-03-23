import os
import numpy as np
from pathlib import Path
import tifffile

# Define source and destination directories
source_dir = Path("/path/to/your/images")           # *** SET THIS PATH ***
output_dir = Path("/path/to/your/images_multiply")  # *** SET THIS PATH ***

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Multiplication parameters
MULTIPLY_FACTOR_MIN = 0.5  # Minimum multiplication factor (darken)
MULTIPLY_FACTOR_MAX = 1.8  # Maximum multiplication factor (brighten)
USE_RANDOM_FACTOR = True   # If True, use random factor per image; if False, use fixed factor

# Counter for reporting
processed_count = 0

print(f"Processing images from: {source_dir}")
print(f"Output directory: {output_dir}")
if USE_RANDOM_FACTOR:
    print(f"Multiplication factor range: {MULTIPLY_FACTOR_MIN} - {MULTIPLY_FACTOR_MAX} (random per image)\n")
else:
    print(f"Multiplication factor: {MULTIPLY_FACTOR_MIN} (fixed)\n")

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

        # Determine multiplication factor
        if USE_RANDOM_FACTOR:
            factor = np.random.uniform(MULTIPLY_FACTOR_MIN, MULTIPLY_FACTOR_MAX)
        else:
            factor = MULTIPLY_FACTOR_MIN

        # Multiply pixel values by factor
        multiplied = img.astype(np.float32) * factor

        # Clip values to valid range and convert back to original dtype
        if img.dtype == np.uint8:
            multiplied = np.clip(multiplied, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            multiplied = np.clip(multiplied, 0, 65535).astype(np.uint16)
        else:
            multiplied = multiplied.astype(img.dtype)

        # Create output filename with _mult suffix
        output_filename = tif_file.stem + "-mult.tif"
        output_path = output_dir / output_filename

        # Save the processed image
        tifffile.imwrite(output_path, multiplied)

        processed_count += 1
        print(f"✓ Processed: {tif_file.name} -> {output_filename} (factor={factor:.3f})")

    except Exception as e:
        print(f"✗ Error processing {tif_file.name}: {str(e)}")

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
print(f"{'='*60}")
