import pytest
import torch

pytest.importorskip("timm")
pytest.importorskip("natten")

from lightstream.models.nat import NCHWNatNano, NCHWNatPico, nat_nano, nat_pico
from lightstream.models.nat.nat import NAT
from lightstream.models.nat.nchw import (
    convert_nhwc_nat_state_dict,
    nchw_nat_nano,
    nchw_nat_pico,
)
from lightstream.models.nat.streaming import _VARIANTS, _checkpoint_state


@pytest.mark.parametrize(
    ("reference_factory", "nchw_factory", "embed_dim"),
    [
        (nat_nano, nchw_nat_nano, 32),
        (nat_pico, nchw_nat_pico, 16),
    ],
)
def test_synthetic_nhwc_factories_match_nchw_configuration(
    reference_factory, nchw_factory, embed_dim
):
    reference = reference_factory()
    model = nchw_factory()

    assert reference.embed_dim == model.embed_dim == embed_dim
    assert reference.mlp_ratio == model.mlp_ratio == 2
    assert [level.depth for level in reference.levels] == [3, 4, 6, 5]
    assert [level.blocks[0].num_heads for level in reference.levels] == [1, 2, 4, 8]
    assert all(
        level.blocks[0].attn.kernel_size in (7, (7, 7))
        for level in reference.levels
    )
    assert all(not level.blocks[0].layer_scale for level in reference.levels)
    assert all(
        isinstance(block.drop_path, torch.nn.Identity)
        for level in reference.levels
        for block in level.blocks
    )


@pytest.mark.parametrize("factory", [nat_nano, nat_pico])
def test_synthetic_nhwc_factories_reject_pretrained(factory):
    with pytest.raises(ValueError, match="no pretrained checkpoint exists"):
        factory(pretrained=True)


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
