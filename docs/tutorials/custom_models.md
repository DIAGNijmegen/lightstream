# Custom models
Custom models can be created in one of three ways:

* Using the `StreamingModule` (recommended)
* Using the `BaseModel` (a subclass of `StreamingModule`)
* Creating your own class (not recommended)

The `StreamingModule` and `BaseModel` classes are both regular `LightningModule` objects and should be treated as such.
Both classes have several helper functions and a custom initialization that create the streaming model instance. Secondly, 
the helper functions make sure that several settings, such as freezing normalization layers and setting them to `eval()` mode, both during training and inference.
This is necessary since streaming does not work with layers that are not locally defined, but rather need the entire input image.

!!! warning

    Please consult this documentation thoroughly before creating your own module. Do not carelessly override the following callbacks/hooks:
    on_train_epoch_start, on_train_start, on_validation_start, on_test_start

## Model prerequisites
Before implementing a streaming version of your model, make sure that the following conditions hold:
- Your model is at least partly a CNN
- Within the CNN, fully connected layers or global pooling layers are not used. In practice, this means it should not have Squeeze and Excite (SE) blocks, since these are global operations, rather than global.


## Splitting the model


## Custom forward/backward logic


## callbacks/hooks