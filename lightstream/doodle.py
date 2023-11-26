import torch
from torchvision.models import convnext_tiny
import torchvision
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


def _reset_parameters_to_constant(model):
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Conv2d)):
            # to counter floating precision errors, we assign 1 to the weights and
            # normalize the output after the conv.
            torch.nn.init.constant_(mod.weight, 1)
            if mod.bias is not None:
                torch.nn.init.constant_(mod.bias, 0)
    return model


def freeze_stochastic_depth(model):
    for m in model.modules():
        if isinstance(m, torchvision.ops.StochasticDepth):
            m.training = False
    return model


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            try:
                n = int(n)
                model[n] = new
            except:
                setattr(model, n, new)

def set_layer_scale(model, val):
    for x in model.modules():
        if hasattr(x, "layer_scale"):
            x.layer_scale.data.fill_(val)

if __name__ == "__main__":
    tile_size = 1920
    image = torch.ones((1, 3, 1920 + 320, 1920 + 320))

    # print(getsizeof(np.random.randint(256, size=(8192, 8192,3), dtype=np.uint8).astype(np.float64)) / 1000000000, "GB")

    model = convnext_tiny().features
    model = torch.nn.Sequential(
        model
    )

    replace_layers(model, torch.nn.LayerNorm, torch.nn.Identity())
    replace_layers(model, torch.nn.Linear, torch.nn.Identity())
    replace_layers(model, torch.nn.GELU, torch.nn.Identity())

    model = freeze_stochastic_depth(model)
    model = _reset_parameters_to_constant(model)

    # temp = model[0](image)
    # out = model[1](temp)

    check_1 = model(image)
    make_dot(check_1, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    set_layer_scale(model, 1.0)

    sCNN = StreamingCNN(model, tile_shape=(1, 3, tile_size, tile_size), verbose=True)
    set_layer_scale(sCNN.stream_module, 1e-6)
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