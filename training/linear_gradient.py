import os
import numpy as np
from pathlib import Path
import tifffile

# Define source and destination directories
source_dir = Path("/path/to/your/images_plus30")          # *** SET THIS PATH *** (input: brightness-shifted images)
output_dir = Path("/path/to/your/images_linear_gradient") # *** SET THIS PATH ***

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Linear gradient parameters
MIN_MULTIPLIER = 0.7  # Minimum intensity multiplier (dark edge)
MAX_MULTIPLIER = 1.5  # Maximum intensity multiplier (bright edge)

# Counter for reporting
processed_count = 0

print(f"Processing images from: {source_dir}")
print(f"Output directory: {output_dir}")
print(f"Linear gradient range: {MIN_MULTIPLIER}x - {MAX_MULTIPLIER}x\n")


def create_linear_gradient(height, width, min_val=0.8, max_val=1.2, direction='random'):
    """
    Create a linear brightness gradient from one edge to the opposite edge.

    Args:
        height (int): Image height
        width (int): Image width
        min_val (float): Minimum multiplier value (dark edge)
        max_val (float): Maximum multiplier value (bright edge)
        direction (str): Gradient direction - 'horizontal', 'vertical', or 'random'

    Returns:
        np.ndarray: Linear gradient with shape (height, width)
    """
    # Randomly choose direction if not specified
    if direction == 'random':
        direction = np.random.choice(['horizontal', 'vertical', 'diagonal_lr', 'diagonal_rl'])

    if direction == 'horizontal':
        # Left to right gradient
        gradient = np.linspace(min_val, max_val, width)
        gradient = np.tile(gradient, (height, 1))

    elif direction == 'vertical':
        # Top to bottom gradient
        gradient = np.linspace(min_val, max_val, height)
        gradient = np.tile(gradient[:, np.newaxis], (1, width))

    elif direction == 'diagonal_lr':
        # Diagonal gradient (top-left to bottom-right)
        y, x = np.ogrid[:height, :width]
        # Normalize diagonal distance from 0 to 1
        normalized = (x + y) / (width + height - 2)
        gradient = min_val + (max_val - min_val) * normalized

    elif direction == 'diagonal_rl':
        # Diagonal gradient (top-right to bottom-left)
        y, x = np.ogrid[:height, :width]
        # Normalize diagonal distance from 0 to 1
        normalized = ((width - 1 - x) + y) / (width + height - 2)
        gradient = min_val + (max_val - min_val) * normalized

    return gradient, direction


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

        # Create random linear gradient with same dimensions as image
        gradient, direction = create_linear_gradient(
            img.shape[0],
            img.shape[1],
            min_val=MIN_MULTIPLIER,
            max_val=MAX_MULTIPLIER,
            direction='random'
        )

        # Apply gradient by multiplying image with the gradient
        modified = img.astype(np.float32) * gradient

        # Clip values to valid range and convert back to original dtype
        if img.dtype == np.uint8:
            modified = np.clip(modified, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            modified = np.clip(modified, 0, 65535).astype(np.uint16)
        else:
            modified = modified.astype(img.dtype)

        # Create output filename with _lg suffix
        output_filename = tif_file.stem + "_lg.tif"
        output_path = output_dir / output_filename

        # Save the processed image
        tifffile.imwrite(output_path, modified)

        processed_count += 1
        print(f"✓ Processed: {tif_file.name} -> {output_filename} (direction: {direction})")

    except Exception as e:
        print(f"✗ Error processing {tif_file.name}: {str(e)}")

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
print(f"{'='*60}")
