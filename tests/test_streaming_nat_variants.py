import pytest
import torch

pytest.importorskip("timm")
pytest.importorskip("natten")

from lightstream.models.nat import NCHWNatNano, NCHWNatPico
from lightstream.models.nat.nat import NAT
from lightstream.models.nat.nchw import convert_nhwc_nat_state_dict
from lightstream.models.nat.streaming import _VARIANTS, _checkpoint_state


@pytest.mark.parametrize(
    ("variant", "factory", "embed_dim"),
    [
        ("nat_nano", NCHWNatNano, 32),
        ("nat_pico", NCHWNatPico, 16),
    ],
)
def test_synthetic_variant_construction_and_nhwc_parity(
    variant, factory, embed_dim
):
    config = _VARIANTS[variant]
    assert config["checkpoint"] is None
    assert config["nchw"] is factory
    assert config["reference"] == {
        "depths": [3, 4, 6, 5],
        "num_heads": [1, 2, 4, 8],
        "embed_dim": embed_dim,
        "mlp_ratio": 2,
        "kernel_size": 7,
        "layer_scale": None,
    }

    torch.manual_seed(7)
    reference = NAT(
        **config["reference"],
        num_classes=1000,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ).eval()
    model = factory().eval()
    feature_state = {
        key: value
        for key, value in reference.state_dict().items()
        if not key.startswith("head.")
    }
    model.load_state_dict(convert_nhwc_nat_state_dict(feature_state), strict=True)

    inputs = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = reference.forward_feature_map(inputs).permute(0, 3, 1, 2)
        actual = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("variant", ["nat_nano", "nat_pico"])
def test_synthetic_variants_reject_nonexistent_official_checkpoints(variant):
    message = "synthetic variant with no official checkpoint"
    with pytest.raises(ValueError, match=message):
        _checkpoint_state(variant, True)
    with pytest.raises(ValueError, match=message):
        _checkpoint_state(variant, f"{variant}_1k")


@pytest.mark.parametrize("variant", ["nat_nano", "nat_pico"])
def test_synthetic_variants_accept_supplied_state_dict(variant):
    state = {"weight": torch.ones(1)}
    resolved = _checkpoint_state(variant, state)
    assert resolved is state
