import pytest


@pytest.fixture(params=[0, 1], ids=["spam", "ham"])
def a(request):
    print("request", request)
    return request.param


def test_a(a):
    pass


def test_a2(a):
    pass


def idfn(fixture_value):
    if fixture_value == 0:
        return "eggs"
    else:
        return None


@pytest.fixture(params=[0, 1], ids=idfn)
def b(request):
    return request.param


def test_b(b):
    pass
