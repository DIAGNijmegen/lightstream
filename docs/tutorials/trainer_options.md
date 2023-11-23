# Trainer settings
The `Trainer` settings defined in native Pytorch Lightning will

## Multi GPU support

Training on multiple gpu's in streaming is handled by Pytorch lightning. However, the native `auto` option within the `Trainer` class will not work (we omit the technical details why here).
For the `strategy` argument of the Trainer, we therefore recommend to use the `ddp_find_unused_parameters_true` instead, which does not conflict with streaming and gradient checkpointing.

## Gradient accumulation
Specifying higher batch sizes will not affect normalization layers during training, as they should be on `eval()` mode. However, gradient accumulation is still possible and can
stabilize training under certain circumstances. This can be easily set using the `accumulate_grad_batches` argument.

## Precision
We recommend to train using mixed precision training wherever possible and to let pytorch handle the conversion. This can be set using the `16-mixed` option.

## Loggers and callbacks
Callbacks for a variety of training strategies (checkpointing, early stopping, etc) are natively supported by Pytorch Lightning. Please consult their respective documentation on how to do this.
The same can be said for standard logging solutions (Tensorboard, Wandb).

## Example

```python
trainer = pl.Trainer(
    default_root_dir="path_to_save_dir",
    accelerator="gpu",
    max_epochs=100,
    devices=2,
    strategy="ddp_find_unused_parameters_true",
    accumulate_grad_batches=8,
    precision="16-mixed",
    logger=wandb_logger,
)

```

