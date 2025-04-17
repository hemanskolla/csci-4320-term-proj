# view.py
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    input_raw = sys.argv[1]
    width, height = map(int, open(f"{input_raw}.header").read().split())

    data = np.fromfile(input_raw, dtype=np.uint8).reshape(height, width)
    plt.imshow(data, cmap="gray")
    plt.show()
