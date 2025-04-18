import os
import argparse
from PIL import Image
from tqdm import tqdm


def convert_jpgs_to_binary(input_dir, output_file, max_images=640):
    # Get sorted list of JPEG files
    jpg_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    )

    i = 0
    with open(output_file, "wb") as bin_file:
        for filename in tqdm(jpg_files, desc="Processing JPEGs"):
            img_path = os.path.join(input_dir, filename)

            # Open image and convert to grayscale
            with Image.open(img_path) as img:
                # Convert to grayscale and resize to 256x256 if needed
                img = img.convert("L").resize((256, 256))

                # Write raw pixel data to binary file
                bin_file.write(img.tobytes())
                i += 1
            if i >= max_images:
                print(f"Processed {i} images, stopping.")
                break
    print(f"Processed {i} images in total.")
    print(f"Created {output_file} with {len(jpg_files)} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JPEG folder to input_data.dat"
    )
    parser.add_argument("input_dir", help="Directory containing input JPEGs")
    parser.add_argument("output_file", help="Output binary file")
    # Add max_images argument
    parser.add_argument(
        "--max-images",
        type=int,
        default=640,
        help="Maximum number of images to process",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} not found")

    convert_jpgs_to_binary(args.input_dir, args.output_file, args.max_images)
    print(
        f"Successfully created {args.output_file} with {args.max_images} images"
    )
