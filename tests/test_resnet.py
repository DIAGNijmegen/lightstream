import torch
import pytest
import numpy as np
from stream.utils.modelcheck import ModelCheck
from torchvision.models import resnet18, resnet34, resnet50
from models.resnet import split_resnet


test_cases = [resnet18, resnet34, resnet50]


@pytest.fixture(scope="module", params=test_cases)
def streaming_outputs(request):
    print("model fn", request.param)
    model = request.param()
    stream_net, head = split_resnet(model)
    model_check = ModelCheck(stream_net, verbose=False)
    model_check.gather_outputs()
    return [model_check.streaming_outputs, model_check.normal_outputs]


def test_forward_output(streaming_outputs):
    stream_outputs, normal_outputs = streaming_outputs
    max_error = np.abs(
        stream_outputs["forward_output"] - normal_outputs["forward_output"]
    ).max()

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
