import torch
import lightning as L

from lightstream.scnn import StreamingCNN


class StreamingModule(L.LightningModule):
    def __init__(self, stream_network, tile_size, use_streaming=True, train_streaming_layers=True, *args, **kwargs):
        super().__init__()
        self.tile_size = tile_size
        self.use_streaming = use_streaming
        self.train_streaming_layers = train_streaming_layers
        self._stream_module = stream_network
        self.params = self.get_trainable_params()
        self.stream_network = StreamingCNN(
            stream_network,
            tile_shape=(1, 3, tile_size, tile_size),
            deterministic=kwargs.get("deterministic", True),
            saliency=kwargs.get("saliency", False),
            gather_gradients=kwargs.get("gather_gradients", False),
            copy_to_gpu=kwargs.get("copy_to_gpu", True),
            verbose=kwargs.get("verbose", True),
            statistics_on_cpu=kwargs.get("statistics_on_cpu", False),
            normalize_on_gpu=kwargs.get("normalize_on_gpu", False),
            mean=kwargs.get("mean", [0.485, 0.456, 0.406]),
            std=kwargs.get("std", [0.229, 0.224, 0.225]),
            state_dict=kwargs.get("state_dict", None),
        )

        if not self.use_streaming:
            self.disable_streaming()

        # set dtype to float16 after initialization, becuase half is not implemented for cpu
        print("dtype in streaming:", kwargs.get("dtype"))
        if kwargs.get("dtype") == torch.half:
            self.stream_network.dtype = torch.half

    def freeze_streaming_normalization_layers(self):
        """Do not use normalization layers within lightstream, only local ops are allowed"""
        freeze_layers = [l for l in self.stream_network.stream_module.modules() if isinstance(l, torch.nn.BatchNorm2d)]

        for mod in freeze_layers:
            mod.eval()

    def disable_streaming(self):
        """Disable streaming hooks and replace streamingconv2d  with conv2d modules"""
        self.stream_network.disable()
        self.use_streaming = False

    def enable_streaming(self):
        """Enable streaming hooks and use streamingconv2d modules"""
        self.stream_network.enable()
        self.use_streaming = True

    def forward_streaming(self, x):
        out = self.stream_network(x) if self.use_streaming else self.stream_network.stream_module(x)
        return out

    def backward_streaming(self, image, gradient):
        """backward only if streaming is turned on. If not, let pytorch do backward via loss.backward()"""
        if self.use_streaming:
            self.stream_network.backward(image, gradient)

    def _configure_tile_delta(self):
        delta = self.tile_size - (
            self.stream_network.tile_gradient_lost.left + self.stream_network.tile_gradient_lost.right
        )
        delta = delta // self.stream_network.output_stride[-1]
        delta *= self.stream_network.output_stride[-1]
        # if delta < 3000:
        #     delta = (3000 // delta + 1) * delta
        print("tile delta value:", delta.detach().cpu())
        return delta.detach().cpu().numpy()

    def get_trainable_params(self):
        print("Get trainable params", self.train_streaming_layers)
        if self.train_streaming_layers:
            params = list(self._stream_module.parameters())
            return params
        else:
            for param in self._stream_module.parameters():
                param.requires_grad = False
