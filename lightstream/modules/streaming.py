import torch
import lightning as L

from lightstream.scnn import StreamingCNN


class StreamingModule(L.LightningModule):
    def __init__(self, stream_network, tile_size, use_streaming=True, train_streaming_layers=True, *args, **kwargs):
        super().__init__()
        self.use_streaming = use_streaming
        self.train_streaming_layers = train_streaming_layers
        self._stream_module = stream_network
        self.params = self.get_trainable_params()

        # StreamingCNN options
        self.tile_size = tile_size
        self.deterministic = kwargs.get("deterministic", True)
        self.saliency = kwargs.get("saliency", False)
        self.copy_to_gpu = kwargs.get("copy_to_gpu", False)
        self.verbose = kwargs.get("verbose", True)
        self.mean = torch.Tensor(kwargs.get("mean", [0.485, 0.456, 0.406]))
        self.std = torch.Tensor(kwargs.get("std", [0.229, 0.224, 0.225]))
        self.statistics_on_cpu = kwargs.get("statistics_on_cpu", True)
        self.normalize_on_gpu = kwargs.get("normalize_on_gpu", False)

        if not self.statistics_on_cpu:
            # Move to cuda manually if statistics are computed on gpu
            device = torch.device("cuda")
            stream_network.to(device)

        self.stream_network = StreamingCNN(
            stream_network,
            tile_shape=(1, 3, tile_size, tile_size),
            deterministic=self.deterministic,
            saliency=self.saliency,
            copy_to_gpu=self.copy_to_gpu,
            verbose=self.verbose,
            statistics_on_cpu=self.statistics_on_cpu,
            normalize_on_gpu=self.normalize_on_gpu,
            mean=torch.Tensor(self.mean),
            std=torch.Tensor(self.std),
            state_dict=kwargs.get("state_dict", None),
        )

        if not self.use_streaming:
            self.disable_streaming()

    def freeze_streaming_normalization_layers(self):
        """Do not use normalization layers within lightstream, only local ops are allowed"""
        freeze_layers = [
            l
            for l in self.stream_network.stream_module.modules()
            if isinstance(l, (torch.nn.BatchNorm2d, torch.nn.LayerNorm))
        ]

        for mod in freeze_layers:
            mod.eval()

    def on_train_epoch_start(self) -> None:
        """on_train_epoch_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        self.freeze_streaming_normalization_layers()

    def on_validation_start(self):
        """on_validation_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        # Update streaming to put all the inputs/tensors on the right device
        self.stream_network.device = self.device
        self.stream_network.mean = self.mean.to(self.device)
        self.stream_network.std = self.std.to(self.device)
        self.stream_network.dtype = self.dtype

    def on_train_start(self):
        """on_train_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        # Update streaming to put all the inputs/tensors on the right device
        self.stream_network.device = self.device
        self.stream_network.mean = self.mean.to(self.device)
        self.stream_network.std = self.std.to(self.device)
        self.stream_network.dtype = self.dtype
    
    def on_test_start(self):
        """on_train_start hook

        Do not override this method. Instead, call the parent class using super().on_train_start if you want
        to add this hook into your pipelines

        """
        # Update streaming to put all the inputs/tensors on the right device
        self.stream_network.device = self.device
        self.stream_network.mean = self.mean.to(self.device)
        self.stream_network.std = self.std.to(self.device)
        self.stream_network.dtype = self.dtype

    def disable_streaming(self):
        """Disable streaming hooks and replace streamingconv2d  with conv2d modules

        This will still use the StreamingCNN backward and forward functions, but the memory gains from gradient
        checkpointing will be turned off.
        """
        self.stream_network.disable()
        self.use_streaming = False

    def enable_streaming(self):
        """Enable streaming hooks and use streamingconv2d modules"""
        self.stream_network.enable()
        self.use_streaming = True

    def forward_streaming(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor
        The input tensor in [1,C,H,W] format

        Returns
        -------
        out: torch.Tensor
        The output of the streaming model

        """
        out = self.stream_network(x) if self.use_streaming else self.stream_network.stream_module(x)
        return out

    def backward_streaming(self, image, gradient):
        """Perform the backward pass using the streaming network

        Backward only if streaming is turned on.
        This method is primarily a convenience function

        Parameters
        ----------
        image: torch.Tensor
            The input image in [1,C,H,W] format
        gradient: torch.Tensor
            The gradient of the next layer in the model to continue backpropagation with

        Returns
        -------

        """

        if self.use_streaming and self.train_streaming_layers:
            self.stream_network.backward(image, gradient)

    def configure_tile_delta(self):
        """
        Helper function that returns the tile stride during streaming.

        Streaming assumes that the input image is perfectly divisible with the network output stride or the
        tile stride. This function will return the tile stride, which can then be used within data processing pipelines
        to pad/crop images to a multiple of the tile stride.

        Examples:

        Returns
        -------
        tile_delta: numpy.ndarray
            the tile stride.


        """
        delta = self.tile_size - (
            self.stream_network.tile_gradient_lost.left + self.stream_network.tile_gradient_lost.right
        )
        delta = delta // self.stream_network.output_stride[-1]
        delta *= self.stream_network.output_stride[-1]
        return delta.detach().cpu().numpy()

    def get_trainable_params(self):
        """Get trainable parameters for the entire model

        If self.streaming_layers is True, then the parameters of the streaming network will be trained.
        Otherwise, the parameters will be left untrained (no gradients will be collected)

        """
        if self.train_streaming_layers:
            params = list(self._stream_module.parameters())
            return params
        else:
            print("WARNING: Streaming network will not be trained")
            for param in self._stream_module.parameters():
                param.requires_grad = False

    def _remove_streaming_network(self):
        """Converts the streaming network into a non-streaming network

        The former streaming encoder can be addressed as self.stream_network
        This function is currently untested and breaks the class, since there is no way to rebuild the streaming network
        other than calling a new class directly.

        """

        # Convert streamingConv2D into regular Conv2D and turn off streaming hooks
        self.disable_streaming()
        self.use_streaming = False
        temp = self.stream_network.stream_module

        # torch modules cannot be overridden normally, so delete and reassign
        del self.stream_network
        self.stream_network = temp

    def _build_streaming_network(self, **kwargs):
        """
        (re)-build the streaming network
        """

        stream_network = self.stream_network
        del self.stream_network

        self.stream_network = StreamingCNN(
            stream_network,
            tile_shape=(1, 3, self.tile_size, self.tile_size),
            deterministic=kwargs.get("deterministic", True),
            saliency=kwargs.get("saliency", False),
            gather_gradients=kwargs.get("gather_gradients", False),
            copy_to_gpu=kwargs.get("copy_to_gpu", True),
            verbose=kwargs.get("verbose", True),
            statistics_on_cpu=kwargs.get("statistics_on_cpu", False),
            normalize_on_gpu=kwargs.get("normalize_on_gpu", False),
            mean=self.mean,
            std=self.std,
            state_dict=kwargs.get("state_dict", None),
        )
