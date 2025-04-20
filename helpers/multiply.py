import os
import shutil

# Set these paths
source_dir = "data"  # Folder with original images
output_dir = "data_10"  # Folder to save 1000x copies
multiplication_factor = 100  # How many times to duplicate each image

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of image files (can be modified to filter by type if needed)
image_files = [
    f
    for f in os.listdir(source_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
]

# Duplicate images
for img_name in image_files:
    img_path = os.path.join(source_dir, img_name)
    name, ext = os.path.splitext(img_name)

    for i in range(multiplication_factor):
        new_name = f"{name}_copy_{i:04d}{ext}"
        new_path = os.path.join(output_dir, new_name)
        shutil.copyfile(img_path, new_path)

print(
    f"Successfully created {len(image_files) * multiplication_factor} images in '{output_dir}'"
)
