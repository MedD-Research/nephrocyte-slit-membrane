import os
import numpy as np
from pathlib import Path
import tifffile
import argparse

def adjust_image_values(source_dir, output_dir, value, suffix):
    """
    Add or subtract a specific value from all images in a directory.
    Positive values will be added, negative values will be subtracted.

    Args:
        source_dir: Path to source directory containing images
        output_dir: Path to output directory for processed images
        value: Value to add (positive) or subtract (negative)
        suffix: Suffix to add to output filenames
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Counter for reporting
    processed_count = 0

    operation = "add" if value >= 0 else "subtract"
    print(f"Processing images from: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Operation: {operation} {abs(value)}")
    print(f"{'='*60}\n")

    # Get all .tif and .tiff files, excluding hidden files and ._ files
    tif_files = [
        f for f in source_dir.rglob("*")
        if f.suffix.lower() in ['.tif', '.tiff']
        and not f.name.startswith('.')
        and not f.name.startswith('._')
    ]

    print(f"Found {len(tif_files)} valid .tif files\n")

    # Process each image
    for tif_file in tif_files:
        try:
            # Read the image using tifffile
            img_array = tifffile.imread(tif_file)

            # Detect bit depth from original array dtype
            is_16bit = img_array.dtype == np.uint16

            # Convert to float for processing
            img_float = img_array.astype(np.float32)

            # Apply operation (just add the value - negative values will subtract)
            adjusted = img_float + value

            # Clip values to valid range and convert back to appropriate dtype
            if is_16bit:
                adjusted = np.clip(adjusted, 0, 65535).astype(np.uint16)
            else:  # 8-bit
                adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

            # Create output filename with suffix
            output_filename = tif_file.stem + suffix + ".tif"
            output_path = output_dir / output_filename

            # Save the processed image using tifffile
            tifffile.imwrite(output_path, adjusted)

            processed_count += 1
            print(f"✓ Processed: {tif_file.name} -> {output_filename}")

        except Exception as e:
            print(f"✗ Error processing {tif_file.name}: {str(e)}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count}/{len(tif_files)} images")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Add or subtract a value from images (use positive to add, negative to subtract)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Subtract 30 (default)
  python add_subtract.py

  # Subtract 50
  python add_subtract.py -v -50

  # Add 30
  python add_subtract.py -v 30

  # Subtract 100 with custom input
  python add_subtract.py -v -100 -i /path/to/images
        """
    )

    parser.add_argument(
        '-v', '--value',
        type=float,
        default=30,
        help='Value to add/subtract. Use positive to add, negative to subtract (default: 30)'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='/Volumes/SanDisk1Tb/00_Code_Experiments/Code056_sanja_nephrocytes_dataset_to_train_unet/datasets_2/images',
        help='Input directory'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory name (default: auto-generated based on value)'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default=None,
        help='Suffix for output filenames (default: auto-generated based on value)'
    )

    args = parser.parse_args()

    # Automatically determine output folder name based on value
    if args.output is None:
        if args.value >= 0:
            output_folder = f"plus{int(args.value)}"
        else:
            output_folder = f"minus{int(abs(args.value))}"
    else:
        output_folder = args.output

    # Automatically determine suffix based on value
    if args.suffix is None:
        if args.value >= 0:
            suffix = f"_plus{int(args.value)}"
        else:
            suffix = f"_minus{int(abs(args.value))}"
    else:
        suffix = args.suffix

    # Define paths
    source_dir = Path(args.input)
    base_dir = Path("/Volumes/SanDisk1Tb/00_Code_Experiments/Code056_sanja_nephrocytes_dataset_to_train_unet/datasets_2")
    output_dir = base_dir / output_folder

    # Run the adjustment
    adjust_image_values(source_dir, output_dir, args.value, suffix)
