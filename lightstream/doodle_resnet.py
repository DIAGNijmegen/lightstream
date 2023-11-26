import torch
from sys import getsizeof
import numpy as np

from torchvision.models import convnext_tiny
from torchvision.models import resnet18
from lightstream.models.resnet.resnet import split_resnet
import torchvision
from torchvision.ops.misc import Permute
from scnn import StreamingCNN
from torchviz import make_dot

def create_dummy_data(img_size):
    image = torch.FloatTensor(3, img_size, img_size).normal_(0, 1)
    image = image.type(dtype)
    image = image.to(device)

    target = torch.tensor(50.0)  # large value so we get larger gradients
    target = target.type(dtype)
    target = target.to(device)

    return image, target

def freeze_streaming_normalization_layers(model):
    """Do not use normalization layers within lightstream, only local ops are allowed"""
    freeze_layers = [l for l in model.modules() if isinstance(l, torch.nn.BatchNorm2d)]

    for mod in freeze_layers:
        mod.eval()

if __name__ == "__main__":
    tile_size = 1920
    image = torch.ones((1, 3, 1920 + 320, 1920 + 320))

    # print(getsizeof(np.random.randint(256, size=(8192, 8192,3), dtype=np.uint8).astype(np.float64)) / 1000000000, "GB")

    model = resnet18()
    model, head = split_resnet(model)
    model = torch.nn.Sequential(
        model[0:4], model[4]
    )

    freeze_streaming_normalization_layers(model)

    # temp = model[0](image)
    # out = model[1](temp)
    check_1 = model(image)
    make_dot(check_1, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")

    sCNN = StreamingCNN(model, tile_shape=(1, 3, tile_size, tile_size), verbose=True)
    check_2 = sCNN(image)

    print(check_1.shape)
    print(check_2.shape)

    if check_1.shape == check_2.shape:
        out = torch.sum(check_1 - check_2)
        print(out)

    """
    def _prev_stats(self, tensor):
        prev = tensor.grad_fn
        prev_stats = None

        while True:
            if prev in self._stats_per_grad_fn:
                prev_stats = self._stats_per_grad_fn[prev]
                break
            if hasattr(prev, "next_functions") and len(prev.next_functions) > 0:
                if len(prev.next_functions) == 1:
                    prev = prev.next_functions[0][0]
            else:
                break
        return prev_stats


 
    def traversal(grad_fn):
        prev_stats = None

        if grad_fn in self._stats_per_grad_fn:
            prev_stats = self._stats_per_grad_fn[grad_fn]
            break
        elif hasattr(grad_fn, "next_functions") and len(grad_fn.next_functions) > 0:
            children = [x[0] for x in grad_fn.next_functions]

            for x in children:
                traversal(x)
        else:
            break

        return prev_stats

    """