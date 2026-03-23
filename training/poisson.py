import os
import numpy as np
from pathlib import Path
import tifffile

# Define source and destination directories
source_dir = Path("/path/to/your/images")          # *** SET THIS PATH ***
output_dir = Path("/path/to/your/images_poisson")  # *** SET THIS PATH ***

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Counter for reporting
processed_count = 0

print(f"Processing images from: {source_dir}")
print(f"Output directory: {output_dir}\n")

# Get all .tif files, excluding hidden files and ._ files
tif_files = [
    f for f in list(source_dir.glob("*.tif")) + list(source_dir.glob("*.tiff"))
    if not f.name.startswith('.') and not f.name.startswith('._')
]

print(f"Found {len(tif_files)} valid .tif files")
print(f"{'='*60}\n")

# Process each image
for tif_file in tif_files:
    try:
        # Read the image using tifffile
        img_array = tifffile.imread(tif_file)

        # Detect bit depth from original array dtype
        is_16bit = img_array.dtype == np.uint16

        # Apply Poisson noise
        # Poisson noise is proportional to the square root of the intensity
        # Scale the image to have reasonable lambda values for Poisson distribution

        if is_16bit:
            # Scale down to reasonable range for Poisson, then scale back
            scale_factor = 65535.0 / 255.0
            scaled_img = img_array.astype(np.float32) / scale_factor
            # Apply Poisson noise (lambda = pixel intensity)
            noisy_scaled = np.random.poisson(scaled_img)
            noisy_image = noisy_scaled * scale_factor
            noisy_image = np.clip(noisy_image, 0, 65535).astype(np.uint16)
        else:  # 8-bit grayscale
            # For 8-bit, use pixel values directly as lambda parameter
            # Poisson noise is naturally generated from the intensity
            noisy_image = np.random.poisson(img_array.astype(np.float32))
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # Create output filename with _pn suffix
        output_filename = tif_file.stem + "-pn.tif"
        output_path = output_dir / output_filename

        # Save the processed image using tifffile
        tifffile.imwrite(output_path, noisy_image)

        processed_count += 1
        print(f"✓ Processed: {tif_file.name} -> {output_filename}")

    except Exception as e:
        print(f"✗ Error processing {tif_file.name}: {str(e)}")

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
print(f"{'='*60}")
