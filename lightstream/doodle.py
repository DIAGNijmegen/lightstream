import dataclasses
from dataclasses_json import dataclass_json


import argparse


@dataclass_json
@dataclasses.dataclass
class ExperimentOptions:
    """REQUIRED"""

    image_path: str = ""

    data_path = root / Path("/input/data/images")
    mask_path = root / Path("/input/data/tissue_masks")
    train_csv = root / Path("/input/data/train.csv")
    val_csv = root / Path("/input/data/val.csv")
    test_csv = root / Path("/input/data/test.csv")

    model = StreamingCLAM(
        "resnet18",
        tile_size=9984,
        loss_fn=torch.nn.functional.cross_entropy,
        branch="sb",
        n_classes=2,
        max_pool_kernel=0,
        statistics_on_cpu=True,
        verbose=False,
        train_streaming_layers=False,
    )

    tile_delta = model._configure_tile_delta()
    network_output_stride = max(
        model.stream_network.output_stride[1] * model.max_pool_kernel, model.stream_network.output_stride[1]
    )

    train_dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(train_csv),
        tile_size=9984,
        img_size=32768,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="",
        variable_input_shapes=True,
        tile_delta=tile_delta,
        network_output_stride=network_output_stride,
        filetype=".tif",
        read_level=1,
    )

    val_dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(val_csv),
        tile_size=9984,
        img_size=32768,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="",
        variable_input_shapes=True,
        tile_delta=tile_delta,
        network_output_stride=network_output_stride,
        filetype=".tif",
        read_level=1,
    )

    test_dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(test_csv),
        tile_size=9984,
        img_size=32768,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="",
        variable_input_shapes=True,
        tile_delta=tile_delta,
        network_output_stride=network_output_stride,
        filetype=".tif",
        read_level=1,
    )

    sampler = weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, num_workers=3, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, num_workers=3, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="streamingclam-derma-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=False,
        mode="min",
        every_n_epochs=1,
        verbose=True,
    )

    # train model
    trainer = pl.Trainer(
        default_root_dir="/opt/ml/checkpoints",
        accelerator="gpu",
        max_epochs=2,
        check_val_every_n_epoch=1,
        devices=8,
        strategy="ddp",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def configure_parser_with_options(self):
        """Create an argparser based on the attributes"""
        parser = argparse.ArgumentParser(description="MultiGPU streaming")
        for name, default in dataclasses.asdict(self).items():
            argname = "--" + name
            tp = type(default)
            if tp is bool:
                if default == True:
                    argname = "--no_" + name
                    parser.add_argument(argname, action="store_false", dest=name)
                else:
                    parser.add_argument(argname, action="store_true")
            else:
                parser.add_argument(argname, default=default, type=tp)
        return parser

    def parser_to_options(self, parsed_args: dict):
        """Parse an argparser"""
        for name, value in parsed_args.items():
            self.__setattr__(name, value)
