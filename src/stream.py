from scnn import StreamingCNN
import torch
import lightning as L


class StreamingModule(L.LightningModule):
    def __init__(self, network, tile_size, *args, **kwargs):
        super().__init__()
        self.network = self.convert_to_streaming_network(network, tile_size, **kwargs)

    def freeze_normalization_layers(self):
        """Do not use normalization layers within streaming"""
        self.freeze_layers = [
            l for l in self.network.modules() if isinstance(l, torch.nn.BatchNorm2d)
        ]
        for mod in self.freeze_layers:
            mod.eval()

    def freeze_streaming_network(self):
        """
        Freeze the parameters of the entire streaming network
        """
        pass

    def convert_to_streaming_network(self, model, tile_size, **kwargs):
        model = StreamingCNN(
            model,
            tile_shape=(1, 3, tile_size, tile_size),
            deterministic=kwargs.get("deterministic", True),
            saliency=kwargs.get("saliency", False),
            gather_gradients=kwargs.get("gather_gradients", False),
            copy_to_gpu=kwargs.get("copy_to_gpu", True),
            verbose=kwargs.get("verbose", True),
        )

        model = self.freeze_normalization_layers(model)

        return model

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    print("hi")
