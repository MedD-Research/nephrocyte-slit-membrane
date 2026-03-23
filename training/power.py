import os
import numpy as np
from pathlib import Path
import tifffile

# Define source and destination directories
source_dir = Path("/path/to/your/images")        # *** SET THIS PATH ***
output_dir = Path("/path/to/your/images_power")  # *** SET THIS PATH ***

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Power transformation parameters
POWER_FACTOR_MIN = 0.5   # Minimum power factor (gamma < 1: brighten)
POWER_FACTOR_MAX = 1.8   # Maximum power factor (gamma > 1: darken)
USE_RANDOM_FACTOR = True # If True, use random factor per image; if False, use fixed factor

# Counter for reporting
processed_count = 0

print(f"Processing images from: {source_dir}")
print(f"Output directory: {output_dir}")
if USE_RANDOM_FACTOR:
    print(f"Power factor (gamma) range: {POWER_FACTOR_MIN} - {POWER_FACTOR_MAX} (random per image)")
else:
    print(f"Power factor (gamma): {POWER_FACTOR_MIN} (fixed)")
print("Note: gamma < 1 brightens, gamma > 1 darkens\n")

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

        # Determine power factor (gamma)
        if USE_RANDOM_FACTOR:
            gamma = np.random.uniform(POWER_FACTOR_MIN, POWER_FACTOR_MAX)
        else:
            gamma = POWER_FACTOR_MIN

        # Normalize to [0, 1] range based on dtype
        if img.dtype == np.uint8:
            max_val = 255.0
        elif img.dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = float(img.max())

        # Normalize, apply power transformation, then scale back
        normalized = img.astype(np.float32) / max_val
        powered = np.power(normalized, gamma)
        scaled = powered * max_val

        # Clip values to valid range and convert back to original dtype
        if img.dtype == np.uint8:
            result = np.clip(scaled, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            result = np.clip(scaled, 0, 65535).astype(np.uint16)
        else:
            result = scaled.astype(img.dtype)

        # Create output filename with _pow suffix
        output_filename = tif_file.stem + "-pow.tif"
        output_path = output_dir / output_filename

        # Save the processed image
        tifffile.imwrite(output_path, result)

        processed_count += 1
        effect = "brighten" if gamma < 1.0 else "darken" if gamma > 1.0 else "unchanged"
        print(f"✓ Processed: {tif_file.name} -> {output_filename} (gamma={gamma:.3f}, {effect})")

    except Exception as e:
        print(f"✗ Error processing {tif_file.name}: {str(e)}")

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
print(f"{'='*60}")
