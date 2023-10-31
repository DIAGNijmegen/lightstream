import sys
import numpy as np

print(sys.getsizeof(np.random.random((16384, 16384, 3)).astype(np.float32)) / 1e9)
