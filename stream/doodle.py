import numpy as np
import pytest


@pytest.fixture
def tuple_params(request):
    yield sum(request.param)


@pytest.mark.parametrize("tuple_params", [(1, 2, 3)], indirect=True)
def test_tuple_params(tuple_params):
    print("meowing here")
    print(tuple_params)  # 6


@pytest.fixture
def dict_params(request):
    yield f"{request.param['a']}_{request.param['b']}"


@pytest.mark.parametrize("dict_params", [{"a": "foo", "b": "bar"}], indirect=True)
def test_dict_params(dict_params):
    print(dict_params)  # foo_bar
