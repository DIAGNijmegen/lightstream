import torch
import pytest
import numpy as np
from tests.modelcheck import ModelCheck
from torchvision.models import resnet18, resnet34, resnet50
from models.resnet.resnet import split_resnet


test_cases = [resnet18, resnet34, resnet50]


def make_dummy_data():
    img_size = 1600 + 640
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.FloatTensor(3, img_size, img_size).normal_(0, 1)
    image = image.type(dtype).to(device)

    target = torch.tensor(50.0)  # large value so we get larger gradients
    target = target.type(dtype).to(device)

    batch = (image[None], target)
    return batch


@pytest.fixture(scope="module", params=test_cases)
def streaming_outputs(request):
    print("model fn", request.param)
    model = request.param()

    tile_size = 1920

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = make_dummy_data()

    stream_net, head = split_resnet(model)
    model_check = ModelCheck(stream_net, tile_size, loss_fn=torch.nn.MSELoss(), verbose=False, saliency=True)
    model_check.to(device)
    model_check.to(dtype)
    streaming, normal = model_check.run(batch)

    return [streaming, normal]


def test_forward_output(streaming_outputs):
    stream_outputs, normal_outputs = streaming_outputs
    max_error = np.abs(stream_outputs["forward_output"] - normal_outputs["forward_output"]).max()

    # if max_error < 1e-7:
    #    print("Equal output to streaming")
    # else:
    #    print("NOT equal output to streaming"),
    #    print("error:", max_error)
    assert max_error < 1e-2


def test_input_gradients(streaming_outputs):
    stream_outputs, normal_outputs = streaming_outputs

    diff = np.abs(stream_outputs["input_gradient"] - normal_outputs["input_gradient"])
    assert diff.max() < 1e-2


def test_kernel_gradients(streaming_outputs):
    stream_outputs, normal_outputs = streaming_outputs
    streaming_kernel_gradients = stream_outputs["kernel_gradients"]
    conventional_kernel_gradients = normal_outputs["kernel_gradients"]

    for i in range(len(streaming_kernel_gradients)):
        diff = np.abs(streaming_kernel_gradients[i] - conventional_kernel_gradients[i])
        max_diff = diff.max()
        # print(f"Conv layer {i} \t max difference between kernel gradients: {max_diff}")
        assert max_diff < 1e-2


def get_kernel_sizes(kernel_gradients):
    for i in range(len(kernel_gradients)):
        print(
            "Conv layer",
            i,
            "\t average gradient size:",
            float(torch.mean(torch.abs(kernel_gradients[i].cpu().numpy()))),
        )
