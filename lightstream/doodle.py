import torch
from sys import getsizeof
import numpy as np

from torchvision.models import densenet121

if __name__ == "__main__":
    #print(getsizeof(np.random.randint(256, size=(8192, 8192,3), dtype=np.uint8).astype(np.float64)) / 1000000000, "GB")

    model = densenet121()
    print(model)