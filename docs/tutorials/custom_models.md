# Custom models
Custom models can be created in one of three ways:

* Using the `StreamingModule` (recommended)
* Using the `ImageNetClassifier` (a subclass of `StreamingModule`)
* Creating your own class (not recommended)

The `StreamingModule` and `ImageNetClassifier` classes are both regular `LightningModule` objects and should be treated as such.
Both classes have several helper functions and a custom initialization that create the streaming model instance. Secondly, 
the helper functions make sure that several settings, such as freezing normalization layers and setting them to `eval()` mode, both during training and inference.
This is necessary since streaming does not work with layers that are not locally defined, but rather need the entire input image.

!!! warning

    Please consult this documentation thoroughly before creating your own module. Do not carelessly override the following callbacks/hooks:
    on_train_epoch_start, on_train_start, on_validation_start, on_test_start

## Model prerequisites
Before implementing a streaming version of your model, make sure that the following conditions hold:

- Your model is at least partly a CNN
- Within the CNN, fully connected layers or global pooling layers are not used. This also means it should not have Squeeze and Excite (SE) blocks, since these are global operations, rather than local.
    - A small exception to this are normalization layers. If your model contains any, they must be set to `eval()` both during inference **and** training. Since most normalization require the entire input to correctly calculate means and standard deviations, they will theoretically not work with streaming during training. During inference, the means and standard deviations can be applied tile-wise.


## Splitting and creating the model(s)
Usually a CNN contains a part that can be streamed, and a part that cannot be used with streaming. In the case of e.g. a ResNet architecture, the `fc` layers of the model contains global pooling and fully connected layers, and thus are not fit for streaming. 
That's why models are usually split. The correct model can then be made streamable using the `StreamingModule`"

```python
def split_resnet(net, **kwargs):
    """ Split resnet architectures into backbone and fc modules

    The stream_net will contain the CNN backbone that is capable for streaming.
    The fc model is not streamable and will be a separate module
    If num_classes are specified as a kwarg, then a new fc model will be created with the desired classes

    Parameters
    ----------
    net: torch model
        A ResNet model in the format provided by torchvision
    kwargs

    Returns
    -------
    stream_net : torch.nn.Sequential
        The CNN backbone of the ResNet
    head : torch.nn.Sequential
        The head of the model, defaults to the fc model provided by torchvision.

    """


    num_classes = kwargs.get("num_classes", 1000)
    stream_net = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
    )

    # 1000 classes is the default from ImageNet classification
    if num_classes != 1000:
        net.fc = torch.nn.Linear(512, num_classes)
        torch.nn.init.xavier_normal_(net.fc.weight)
        net.fc.bias.data.fill_(0)  # type:ignore

    head = nn.Sequential(net.avgpool, nn.Flatten(), net.fc)

    return stream_net, head


```

After defining the model split, the `StreamingModule` or `BaseModule` can be used to create a model consisting of a streamable CNN backbone and optional head networks.
Both the  `StreamingModule` and `BaseModule` inherit from `lightning.LightningModule`, meaning all the regular pytorch lightning functions and workflows become available here well.
The streamable ResNet is defined below:

```python
class StreamingResNet(StreamingModule):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(
        self,
        model_name: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        train_streaming_layers: bool = True,
        metrics: MetricCollection | None = None,
        **kwargs
    ):
        assert model_name in list(StreamingResNet.model_choices.keys())
        network = StreamingResNet.model_choices[model_name](weights="DEFAULT")
        stream_network, head = split_resnet(network, num_classes=kwargs.pop("num_classes", 1000))

        self._get_streaming_options(**kwargs)
        print("metrics", metrics)
        super().__init__(
            stream_network,
            head,
            tile_size,
            loss_fn,
            train_streaming_layers=train_streaming_layers,
            metrics=metrics,
            **self.streaming_options,
        )
        
    def _get_streaming_options(self, **kwargs):
        """Set streaming defaults, but overwrite them with values of kwargs if present."""

        # We need to add torch.nn.Batchnorm to the keep modules, because of some in-place ops error if we don't
        # https://discuss.pytorch.org/t/register-full-backward-hook-for-residual-connection/146850
        streaming_options = {
            "verbose": True,
            "copy_to_gpu": False,
            "statistics_on_cpu": True,
            "normalize_on_gpu": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "add_keep_modules": [torch.nn.BatchNorm2d],
        }
        self.streaming_options = {**streaming_options, **kwargs}

```
The actual streaming module can be configured with varying settings. These can be passed as a kwarg dictionary within the `super().__init__()` call of the parent class. Under the hood, this dictionary with options is passed to a constructor which takes care of properly building the streaming module.
A more detailed explanation about the constructor can be found below.


## Custom forward/backward logic
Since we inherit directory from the lightning modules, the routine for forward and backpropagation remains mostly similar to that of pytorch lightning. However, there are a few tweaks and tricks that must be taken into account:

- Forward pass: Can be run as a usual, but we recommend making the input image accessible via self: i.e. `self.image = image`

```python
def forward_head(self, x):
    return self.head(x)

def forward(self, x):
    fmap = self.forward_streaming(x)
    out = self.forward_head(fmap)
    return out

```

- Backward pass/training step: The StreamingCNN object that defines the streaming network requires the gradient and input image as input. To actually backpropagate the input, you can use the `backward_streaming` function which requires the input image and the gradient of the head of the model.

```python

def training_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    image, target = batch
    
    # This is needed later in the backward function!
    self.image = image

    self.str_output = self.forward_streaming(image)

    if self.use_streaming:
        self.str_output.requires_grad = self.training

    out = self.forward_head(self.str_output)
    loss = self.loss_fn(out, target)

    self.log_dict({"entropy loss": loss.detach()}, prog_bar=True)
    return loss

def backward(self, loss):
    loss.backward()
    if self.train_streaming_layers and self.use_streaming:
        self.backward_streaming(self.image, self.str_output.grad)
    del self.str_output


```


- Hooks: Several hooks in pytorch lightning are used to set the normalization layers to `eval()` and set the inputs/models to the right device.
    - on_training_start: Allocates the input and models to the correct device at training time.
    - on_validation_start: Allocates the input and models to the correct device at validation time.
    - on_test_start: Allocates the input and models to the correct device at test time.
    - on_train_epoch_start(self): sets all the normalization layers to eval() during training

**Warning: do not override these hooks with your own code. If you need these hooks for any reason, then call the parent method first using e.g.  `super().on_training_start`**


## Streaming using the constructor
Under the hood of the `StreamingModule` class, we have an additional `Constructor` class that actually builds and defines the streaming module. The following arguments can be passed to it:
At the very least, a torch model and a tile size must be provided.
```python
model: torch.nn.modules,
tile_size: int,
verbose: bool = False,
deterministic: bool = False,
saliency: bool = False,
copy_to_gpu: bool = False,
statistics_on_cpu: bool = False,
normalize_on_gpu: bool = False,
mean: list[float, float, float] | None = None,
std: list[float, float, float] | None = None,
tile_cache: dict | None = None,
add_keep_modules: list[torch.nn.modules] | None = None,
before_streaming_init_callbacks: list[Callable[[torch.nn.modules], None], ...] | None = None,
after_streaming_init_callbacks: list[Callable[[torch.nn.modules], None], ...] | None = None,
```

### Constructor default behaviour
By default, the constructor will perform the following steps:   
1. All layers except convolution, local max pooling, and local average pooling layers are set to `nn.Identity`   
2. `before_streaming_init_callbacks` are executed. These are user-specified, and by default, no callbacks are executed.   
3. The streaming module is constructed from the convolution/local pooling layers. Within this step, the model's weights are altered to calculate tile statistics.   
4. The streaming module's `nn.Identity` layers from step 1 are restored back to their old layers, and the correct model weights are reloaded   
5. `after_streaming_init_callbacks` are executed. These are user-specified, and by default, no callbacks are executed.   
6. The streaming module is returned

The before and after streaming initialization callbacks are added for flexibility, since we cannot take all possible variations for model creation into account. 
An example of where these callbacks come in handy is given in the code for the streaming convnext model. Within this model, we need to additionally turn off the stochastic depth operation, as well as the layer scale, which are not normal layers.
For a more detailed example, we invite the reader to look at the code for either the [Resnet](/models/resnet) or [Convnext](/models/convnext) models within the repository.