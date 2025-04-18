import os
import argparse
from PIL import Image
from tqdm import tqdm


def convert_binary_to_jpgs(input_file, output_dir, img_size=(256, 256)):
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Calculate expected file size
    bytes_per_image = img_size[0] * img_size[1]
    file_size = os.path.getsize(input_file)
    num_images = file_size // bytes_per_image

    with open(input_file, "rb") as bin_file:
        for i in tqdm(range(num_images), desc="Converting to JPEGs"):
            # Read image bytes
            img_bytes = bin_file.read(bytes_per_image)

            if len(img_bytes) != bytes_per_image:
                raise ValueError("Unexpected end of file")

            # Create image from bytes
            img = Image.frombytes("L", img_size, img_bytes)

            # Save as JPEG
            output_path = os.path.join(output_dir, f"processed_{i:04d}.jpg")
            img.save(output_path, quality=95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert output_data.dat to JPEGs"
    )
    parser.add_argument(
        "input_file", help="Input binary file (output_data.dat)"
    )
    parser.add_argument("output_dir", help="Output directory for JPEGs")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} not found")

    convert_binary_to_jpgs(args.input_file, args.output_dir)
    print(
        f"Successfully converted to {len(os.listdir(args.output_dir))} JPEGs in {args.output_dir}"
    )
