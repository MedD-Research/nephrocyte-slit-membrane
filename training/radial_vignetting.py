import os
import numpy as np
from pathlib import Path
import tifffile

# Define source and destination directories
source_dir = Path("/path/to/your/images")             # *** SET THIS PATH ***
output_dir = Path("/path/to/your/images_vn")          # *** SET THIS PATH ***

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Vignetting parameters
CENTER_BRIGHTNESS = 1.2  # Center brightness multiplier
EDGE_BRIGHTNESS = 0.8    # Edge brightness multiplier

# Counter for reporting
processed_count = 0

print(f"Processing images from: {source_dir}")
print(f"Output directory: {output_dir}")
print(f"Vignetting: center={CENTER_BRIGHTNESS}x, edges={EDGE_BRIGHTNESS}x\n")


def create_radial_gradient(height, width, center_value=1.2, edge_value=0.8):
    """
    Create a smooth radial gradient from center to edges.

    Args:
        height (int): Image height
        width (int): Image width
        center_value (float): Brightness multiplier at center
        edge_value (float): Brightness multiplier at edges

    Returns:
        np.ndarray: Radial gradient mask with shape (height, width)
    """
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate center coordinates
    center_y, center_x = height / 2, width / 2

    # Calculate distance from center for each pixel
    # Normalize by the maximum possible distance (corner to center)
    max_distance = np.sqrt(center_y**2 + center_x**2)
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    normalized_distance = distance / max_distance

    # Create smooth gradient from center_value to edge_value
    # Use normalized distance (0 at center, 1 at corners)
    gradient = center_value + (edge_value - center_value) * normalized_distance

    return gradient


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

        # Create radial gradient mask with same dimensions as image
        gradient_mask = create_radial_gradient(
            img.shape[0],
            img.shape[1],
            center_value=CENTER_BRIGHTNESS,
            edge_value=EDGE_BRIGHTNESS
        )

        # Apply vignetting by multiplying image with gradient mask
        vignetted = img.astype(np.float32) * gradient_mask

        # Clip values to valid range and convert back to original dtype
        if img.dtype == np.uint8:
            vignetted = np.clip(vignetted, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            vignetted = np.clip(vignetted, 0, 65535).astype(np.uint16)
        else:
            vignetted = vignetted.astype(img.dtype)

        # Create output filename with _vignette suffix
        output_filename = tif_file.stem + "-vignette.tif"
        output_path = output_dir / output_filename

        # Save the processed image
        tifffile.imwrite(output_path, vignetted)

        processed_count += 1
        print(f"✓ Processed: {tif_file.name} -> {output_filename}")

    except Exception as e:
        print(f"✗ Error processing {tif_file.name}: {str(e)}")

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
print(f"{'='*60}")
