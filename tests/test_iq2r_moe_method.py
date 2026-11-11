# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

# The IQ2R config parser tests live in tests/test_quant_config.py, which runs
# without AITER.
try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


@pytest.fixture
def aiter_iq2r():
    """AITER's GLM-5.3 IQ2R-v2 format code; older AITER builds lack it."""
    pytest.importorskip("aiter.ops.iq2r_v2")
    return pytest.importorskip("aiter.iq2r_glm53")


def test_iq2r_weight_loader_requires_exact_shapes():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.tp_size = 1
    parameter = torch.nn.Parameter(
        torch.empty((2, 8), dtype=torch.uint8), requires_grad=False
    )
    parameter.iq2r_name = "w13_weight"
    loaded = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    method.load_weight(parameter, loaded)
    assert torch.equal(parameter, loaded)

    with pytest.raises(ValueError, match="shape mismatch"):
        method.load_weight(parameter, loaded[:, :-1])


def test_iq2r_weight_loader_checks_the_tile_n():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.tp_size = 4
    method.layout = "glm53-packed-v2"
    parameter = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.int32), requires_grad=False
    )
    parameter.iq2r_name = "w2_iq2r_tile_n"
    method.load_weight(parameter, torch.tensor([128], dtype=torch.int32))
    assert parameter.item() == 128

    for loaded in ([64], [128, 128]):
        with pytest.raises(ValueError, match="kernels need 128"):
            method.load_weight(parameter, torch.tensor(loaded, dtype=torch.int32))


@pytest.mark.parametrize("layout", ["glm53-packed-v2", "glm5next-packed-v2"])
def test_iq2r_accepts_the_v2_layouts(layout):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    moe_mod.Iq2rMoEMethod.__init__(method, None, SimpleNamespace(), layout=layout)
    assert method.layout == layout


@pytest.mark.parametrize("layout", [None, "bogus", "glm53-packed-v1"])
def test_iq2r_rejects_other_layouts(layout):
    moe = SimpleNamespace()
    with pytest.raises(ValueError, match="unsupported IQ2R layout"):
        moe_mod.Iq2rMoEMethod.__init__(
            object.__new__(moe_mod.Iq2rMoEMethod), None, moe, layout=layout
        )


# (hidden, experts) of each packed layout.
_LAYOUT_SHAPES = {"glm53-packed-v2": (6144, 256), "glm5next-packed-v2": (4096, 288)}


def _glm53_packed_method(
    tp_size: int, tp_rank: int, intermediate: int, layout: str = "glm53-packed-v2"
):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, ep_size=1
        ),
        max_num_tokens=8,
    )
    method.layout = layout
    method.classes = 4
    method.hidden_size = _LAYOUT_SHAPES[layout][0]
    method.intermediate_size = intermediate
    method.tp_size = tp_size
    method.tp_rank = tp_rank
    try:
        import aiter.iq2r_glm53 as glm53
    except ImportError:
        pass
    else:
        method.shape = getattr(glm53, moe_mod._IQ2R_LAYOUT_SHAPES[layout])
    return method


def _glm53_packed_layer(method, device="meta", num_experts=None):
    layer = torch.nn.Module()
    layer.has_bias = False
    # The shared expert stays FP8, outside the MoE.
    layer.num_fused_shared_experts = 0
    hidden, experts = _LAYOUT_SHAPES[method.layout]
    with torch.device(device):
        method.create_weights(
            layer,
            num_experts=experts if num_experts is None else num_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=method.intermediate_size,
            params_dtype=torch.bfloat16,
        )
    return layer


def test_iq2r_glm53_packed_create_weights_at_tp4(aiter_iq2r):
    from aiter.iq2r_glm53 import iq2r_glm53_gate_bytes, iq2r_glm53_v2_aux_bytes
    from aiter.ops.iq2r_format import IQ2RMetadata

    method = _glm53_packed_method(tp_size=4, tp_rank=1, intermediate=512)
    layer = _glm53_packed_layer(method)

    down = IQ2RMetadata(logical_n=6144, logical_k=512)
    assert tuple(layer.w13_weight.shape) == (256, iq2r_glm53_gate_bytes(1024))
    assert tuple(layer.w13_weight_scale.shape) == (
        256,
        iq2r_glm53_v2_aux_bytes(1024, 6144, 4, gate=True),
    )
    assert tuple(layer.w2_weight.shape) == (256, down.data_bytes)
    assert tuple(layer.w2_weight_scale.shape) == (
        256,
        iq2r_glm53_v2_aux_bytes(6144, 512, 4, gate=False),
    )
    for name in ("w13_iq2r_tile_n", "w2_iq2r_tile_n"):
        parameter = getattr(layer, name)
        assert (tuple(parameter.shape), parameter.dtype) == ((1,), torch.int32)
    assert (method.tp_size, method.tp_rank, method.classes) == (4, 1, 4)


@pytest.mark.parametrize("load_dummy", [None, "empty", "zero", "xavier"])
def test_iq2r_glm53_packed_post_load_checks_codebook_range(
    aiter_iq2r, monkeypatch, load_dummy
):
    method = _glm53_packed_method(tp_size=8, tp_rank=0, intermediate=256)
    layer = _glm53_packed_layer(method, device="cpu")
    for aux in (layer.w13_weight_scale, layer.w2_weight_scale):
        aux.data[:, :4096] = 0x38  # E4M3 1.0, inside the kernels' range
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(load_dummy=load_dummy),
    )
    monkeypatch.setattr(moe_mod, "_IQ2R_GLM53_WORKSPACES", {})
    method.process_weights_after_loading(layer)

    layer.w2_weight_scale.data[0, 5] = 0x05  # subnormal: copies form wrongly
    if load_dummy:  # dummy weights carry no codebooks
        method.process_weights_after_loading(layer)
    else:
        with pytest.raises(ValueError, match="codebook bytes"):
            method.process_weights_after_loading(layer)


@pytest.mark.parametrize(
    "layout,limit,ok",
    [
        ("glm53-packed-v2", None, True),
        ("glm53-packed-v2", 10.0, False),
        ("glm5next-packed-v2", 10.0, True),
        ("glm5next-packed-v2", 7.0, False),
        ("glm5next-packed-v2", None, False),
    ],
)
def test_iq2r_packed_post_load_checks_the_swiglu_limit(
    aiter_iq2r, monkeypatch, layout, limit, ok
):
    """The kernels build in the model's SwiGLU: GLM-5.3-Flash's clamp at 10."""
    method = _glm53_packed_method(8, 0, 256, layout)
    layer = SimpleNamespace(
        w13_weight=torch.empty((2, 1), dtype=torch.uint8),
        w13_weight_scale=torch.full((2, 4096), 0x38, dtype=torch.uint8),
        w2_weight_scale=torch.full((2, 4096), 0x38, dtype=torch.uint8),
    )
    if limit is not None:
        layer.swiglu_limit = limit
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(load_dummy=None)
    )
    monkeypatch.setattr(method, "_glm53_workspace", lambda device: None)
    if ok:
        method.process_weights_after_loading(layer)
    else:
        with pytest.raises(ValueError, match="clamp the SwiGLU"):
            method.process_weights_after_loading(layer)


def test_iq2r_glm5next_packed_create_weights_at_tp8(aiter_iq2r):
    from aiter.iq2r_glm53 import (
        GLM5_NEXT,
        iq2r_glm53_gate_bytes,
        iq2r_glm53_v2_aux_bytes,
    )
    from aiter.ops.iq2r_format import IQ2RMetadata

    method = _glm53_packed_method(8, 3, 256, "glm5next-packed-v2")
    layer = _glm53_packed_layer(method)
    # Down rows are padded to 4608 for the kernels' tiles.
    down = IQ2RMetadata(logical_n=4608, logical_k=256)
    assert tuple(layer.w13_weight.shape) == (
        288,
        iq2r_glm53_gate_bytes(512, GLM5_NEXT),
    )
    assert tuple(layer.w13_weight_scale.shape) == (
        288,
        iq2r_glm53_v2_aux_bytes(512, 4096, 4, gate=True),
    )
    assert tuple(layer.w2_weight.shape) == (288, down.data_bytes)
    assert tuple(layer.w2_weight_scale.shape) == (
        288,
        iq2r_glm53_v2_aux_bytes(4608, 256, 4, gate=False),
    )
    assert method.shape is GLM5_NEXT


@pytest.mark.parametrize("hidden,experts", [(6144, 288), (4096, 256)])
def test_iq2r_glm5next_packed_rejects_other_shapes(aiter_iq2r, hidden, experts):
    method = _glm53_packed_method(8, 0, 256, "glm5next-packed-v2")
    layer = torch.nn.Module()
    layer.has_bias = False
    layer.num_fused_shared_experts = 0
    with pytest.raises(ValueError, match="glm5next-packed-v2 IQ2R layout is for"):
        method.create_weights(
            layer,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=256,
            params_dtype=torch.bfloat16,
        )


def test_glm5next_weights_mapping_targets_every_packed_tensor(aiter_iq2r):
    compiler = pytest.importorskip("aiter.quantization.iq2r.glm5.compile")
    from atom.model_loader.weight_names import CheckpointNameRewriter
    from atom.models.glm5_next import Glm5NextForConditionalGeneration

    layer = _glm53_packed_layer(_glm53_packed_method(8, 0, 256, "glm5next-packed-v2"))
    rewriter = CheckpointNameRewriter(
        weights_mapping=Glm5NextForConditionalGeneration.weights_mapping
    )
    # A glm5next-packed-v2 checkpoint uses the packed names under the
    # multimodal checkpoint's text prefix.
    mapped = sorted(
        rewriter.rewrite(key)
        for projection in ("gate_up", "down")
        for key in compiler.iq2r_compiled_tensor_keys(
            3, projection, prefix="model.language_model."
        ).values()
    )
    assert mapped == sorted(
        f"model.layers.3.mlp.experts.{name}" for name, _ in layer.named_parameters()
    )


def test_glm53_weights_mapping_targets_every_compiled_tensor(aiter_iq2r):
    compiler = pytest.importorskip("aiter.quantization.iq2r.glm5.compile")
    from atom.model_loader.weight_names import CheckpointNameRewriter
    from atom.models.deepseek_v2 import GlmMoeDsaForCausalLM

    layer = _glm53_packed_layer(_glm53_packed_method(4, 0, 512))
    rewriter = CheckpointNameRewriter(
        weights_mapping=GlmMoeDsaForCausalLM.weights_mapping
    )
    mapped = sorted(
        rewriter.rewrite(key)
        for projection in ("gate_up", "down")
        for key in compiler.iq2r_compiled_tensor_keys(3, projection).values()
    )
    assert mapped == sorted(
        f"model.layers.3.mlp.experts.{name}" for name, _ in layer.named_parameters()
    )


@pytest.mark.parametrize(
    "tp_size,ep_size,fused_shared",
    [(2, 1, 0), (1, 1, 0), (4, 2, 0), (8, 1, 1)],
)
def test_iq2r_glm53_packed_rejects_unsupported_parallelism(
    tp_size, ep_size, fused_shared
):
    method = _glm53_packed_method(tp_size, 0, 2048 // tp_size)
    method.moe.moe_parallel_config.ep_size = ep_size
    layer = torch.nn.Module()
    layer.has_bias = False
    layer.num_fused_shared_experts = fused_shared
    with (
        pytest.raises(NotImplementedError, match="glm53-packed-v2"),
        torch.device("meta"),
    ):
        method.create_weights(
            layer,
            num_experts=256 + fused_shared,
            hidden_size=6144,
            intermediate_size_per_partition=2048 // tp_size,
            params_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("layout", ["glm53-packed-v2", "glm5next-packed-v2"])
@pytest.mark.parametrize("tp_size,tp_rank", [(4, 3), (8, 5)])
def test_iq2r_glm53_packed_loader_slices_the_rank_shard(
    aiter_iq2r, tp_size, tp_rank, layout
):
    from aiter.iq2r_glm53 import (
        iq2r_glm53_gate_bytes,
        iq2r_glm53_slice_gate,
        iq2r_glm53_v2_aux_bytes,
        iq2r_glm53_v2_slice_down_aux,
        iq2r_glm53_v2_slice_gate_aux,
    )
    from aiter.ops.iq2r_format import IQ2RMetadata, iq2r_slice_input_data

    shard = 2048 // tp_size
    method = _glm53_packed_method(tp_size, tp_rank, shard, layout)
    shape = method.shape
    hidden, rows = shape.hidden, shape.down_rows
    down = IQ2RMetadata(logical_n=rows, logical_k=2048)
    generator = torch.Generator().manual_seed(tp_size)

    def full(columns):
        return torch.randint(0, 256, (1, columns), generator=generator).to(torch.uint8)

    weights = {
        "w13_weight": full(iq2r_glm53_gate_bytes(4096, shape)),
        "w13_weight_scale": full(iq2r_glm53_v2_aux_bytes(4096, hidden, 4, gate=True)),
        "w2_weight": full(down.data_bytes),
        "w2_weight_scale": full(iq2r_glm53_v2_aux_bytes(rows, 2048, 4, gate=False)),
    }
    start = 2 * shard * tp_rank
    expected = {
        "w13_weight": iq2r_glm53_slice_gate(
            weights["w13_weight"], start, 2 * shard, shape
        ),
        "w13_weight_scale": iq2r_glm53_v2_slice_gate_aux(
            weights["w13_weight_scale"], start, 2 * shard, 4096, 4
        ),
        "w2_weight": iq2r_slice_input_data(
            weights["w2_weight"], down, shard * tp_rank, shard
        ),
        "w2_weight_scale": iq2r_glm53_v2_slice_down_aux(
            weights["w2_weight_scale"], shard * tp_rank, shard, 2048, 4, shape
        ),
    }
    # The rank's shard has the shape create_weights allocates.
    assert expected["w13_weight"].shape[1] == iq2r_glm53_gate_bytes(2 * shard, shape)
    assert expected["w13_weight_scale"].shape[1] == iq2r_glm53_v2_aux_bytes(
        2 * shard, hidden, 4, gate=True
    )
    assert expected["w2_weight_scale"].shape[1] == iq2r_glm53_v2_aux_bytes(
        rows, shard, 4, gate=False
    )
    for name, value in expected.items():
        parameter = torch.nn.Parameter(torch.empty_like(value), requires_grad=False)
        parameter.iq2r_name = name
        method.load_weight(parameter, weights[name])
        assert torch.equal(parameter, value), name


def test_iq2r_glm53_packed_apply_runs_the_glm53_moe(aiter_iq2r, monkeypatch):
    method = _glm53_packed_method(tp_size=4, tp_rank=0, intermediate=512)
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        w13_weight=torch.empty((256, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((256, 5), dtype=torch.uint8),
        w2_weight=torch.empty((256, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((256, 7), dtype=torch.uint8),
    )
    hidden = torch.randn(2, 6144, dtype=torch.bfloat16)
    # FusedMoE routes before apply: top-8 over the 256 routed experts. The
    # kernels take int32 ids and FP32 weights.
    topk_weights = torch.full((2, 8), 0.1, dtype=torch.bfloat16)
    topk_ids = torch.arange(16, dtype=torch.int64).reshape(2, 8)
    calls = []
    monkeypatch.setattr(
        method, "_glm53_workspace", lambda device: ("workspace", device)
    )
    monkeypatch.setattr(
        aiter_iq2r, "iq2r_glm53_moe_out", lambda *args: calls.append(args)
    )
    arguments = {
        "layer": layer,
        "x": hidden,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "global_num_experts": 256,
        "activation": moe_mod.ActivationType.Silu,
    }

    output = method.apply(**arguments)

    assert len(calls) == 1
    assert torch.equal(calls[0][0], hidden)
    assert calls[0][1] is layer.w13_weight
    assert calls[0][5].dtype == torch.float32
    assert torch.equal(calls[0][5], topk_weights.float())
    assert calls[0][6].dtype == torch.int32
    assert torch.equal(calls[0][6], topk_ids.int())
    assert calls[0][5].shape == calls[0][6].shape == (2, 8)
    assert calls[0][7] is output
    assert calls[0][8] == ("workspace", hidden.device)
    with pytest.raises(ValueError, match="does not support"):
        method.apply(**{**arguments, "activation": moe_mod.ActivationType.Swiglu})
    with pytest.raises(ValueError, match="BF16 rows"):
        method.apply(**{**arguments, "x_scale": torch.ones(2, 1)})
