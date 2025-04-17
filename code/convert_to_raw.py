import numpy as np
from PIL import Image
import sys

if __name__ == "__main__":
    input_jpg = sys.argv[1]
    output_raw = sys.argv[2]

    img = Image.open(input_jpg).convert("L")  # Convert to grayscale
    np.asarray(img, dtype=np.uint8).tofile(output_raw)

    # Save dimensions to a header file
    with open(f"{output_raw}.header", "w") as f:
        f.write(f"{img.width} {img.height}")
