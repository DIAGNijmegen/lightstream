from typing import Any

import torch
import lightning as L
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT

from src.stream import StreamingModule
from torchvision.models import resnet18, resnet34, resnet50

class StreamingResNet(L.LightningModule):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(self, model_name, tile_size, *args, **kwargs):
        super().__init__()
        assert model_name in list(StreamingResNet.model_choices.keys())
        self.model = StreamingResNet.model_choices[model_name](weights="IMAGENET1K_V1")
        self.stream_net, self.head = self.split_model()
        self.stream_net = StreamingModule(self.stream_net, tile_size)


    def split_model(self):
        stream_net = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )
        head = nn.Sequential(self.model.avgpool, nn.Flatten(), self.model.fc)
        return stream_net, head

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass


    def forward(self, x, *args, **kwargs):
        x = self.stream_net(x)
        return self.head(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)



if __name__ == "__main__":
    model = StreamingResNet("resnet18", tile_size=1600)
    model.eval()
    inputs = torch.ones((1, 3, 3200, 3200))

    output_start = model(inputs)

    test_model = resnet18(weights="IMAGENET1K_V1")
    test_model.eval()
    output_test = test_model(inputs)

    print(torch.sum(output_test - output_start))


    # Check backward loop