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


# Per model: aiter.iq2r_glm53 shape name, hidden, routed experts, padded down
# rows, and the layout per codebook size.
_MODELS = {
    "glm53": ("GLM53", 6144, 256, 6144, {512: "glm53-packed-v2", 1024: "glm53-packed-v3"}),
    "glm5next": (
        "GLM5_NEXT",
        4096,
        288,
        4608,
        {512: "glm5next-packed-v2", 1024: "glm5next-packed-v3"},
    ),
}


@pytest.fixture
def aiter_iq2r():
    """AITER's GLM-5.3 IQ2R-v2 format code; older AITER builds lack it."""
    pytest.importorskip("aiter.ops.iq2r_v2")
    return pytest.importorskip("aiter.iq2r_glm53")


def test_iq2r_weight_loader_requires_exact_shapes():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.tp_size = 1
    method.shards = 1
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


@pytest.mark.parametrize("model", sorted(_MODELS))
@pytest.mark.parametrize("entries", [512, 1024])
def test_iq2r_accepts_the_v2_and_v3_layouts(model, entries):
    shape_name, *_, layouts = _MODELS[model]
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    moe_mod.Iq2rMoEMethod.__init__(
        method, None, SimpleNamespace(), layout=layouts[entries]
    )
    assert (method.shape_name, method.codebook_entries) == (shape_name, entries)


@pytest.mark.parametrize("layout", [None, "bogus", "glm53-packed-v1"])
def test_iq2r_rejects_other_layouts(layout):
    moe = SimpleNamespace()
    with pytest.raises(ValueError, match="unsupported IQ2R layout"):
        moe_mod.Iq2rMoEMethod.__init__(
            object.__new__(moe_mod.Iq2rMoEMethod), None, moe, layout=layout
        )


def _glm53_packed_method(
    tp_size: int,
    tp_rank: int,
    intermediate: int,
    entries: int = 512,
    model: str = "glm53",
):
    shape_name, hidden, *_, layouts = _MODELS[model]
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.codebook_entries = entries
    method.layout = layouts[entries]
    method.shape_name = shape_name
    glm53 = pytest.importorskip("aiter.iq2r_glm53")
    method.shape = getattr(glm53, shape_name)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, ep_size=1
        ),
        max_num_tokens=8,
    )
    method.classes = 4
    method.hidden_size = hidden
    method.intermediate_size = intermediate
    method.shards = max(intermediate // 512, 1)
    method.tp_size = tp_size
    method.tp_rank = tp_rank
    return method


def _glm53_packed_layer(method, device="meta", num_experts=None):
    """The method's layer; ``num_experts`` (small CPU layers) runs it under a
    copy of the model shape with that many experts."""
    import dataclasses

    glm53 = pytest.importorskip("aiter.iq2r_glm53")
    layer = torch.nn.Module()
    layer.has_bias = False
    # The shared expert stays FP8, outside the MoE.
    layer.num_fused_shared_experts = 0
    layer.swiglu_limit = method.shape.swiglu_limit
    shape = method.shape
    if num_experts is not None:
        setattr(glm53, method.shape_name, dataclasses.replace(shape, experts=num_experts))
    try:
        with torch.device(device):
            method.create_weights(
                layer,
                num_experts=num_experts or shape.experts,
                hidden_size=method.hidden_size,
                intermediate_size_per_partition=method.intermediate_size,
                params_dtype=torch.bfloat16,
            )
    finally:
        setattr(glm53, method.shape_name, shape)
    return layer


@pytest.mark.parametrize("model", sorted(_MODELS))
@pytest.mark.parametrize("entries", [512, 1024])
def test_iq2r_glm53_packed_create_weights_at_tp4(aiter_iq2r, entries, model):
    from aiter.iq2r_glm53 import iq2r_glm53_gate_bytes, iq2r_glm53_v2_aux_bytes
    from aiter.ops.iq2r_format import IQ2RMetadata

    _, hidden, experts, rows, _ = _MODELS[model]
    method = _glm53_packed_method(
        tp_size=4, tp_rank=1, intermediate=512, entries=entries, model=model
    )
    layer = _glm53_packed_layer(method)
    shape = method.shape

    # GLM-5.3-Flash's down rows are padded to the kernels' 4,608.
    down = IQ2RMetadata(logical_n=rows, logical_k=512, codebook_entries=entries)
    assert tuple(layer.w13_weight.shape) == (
        experts,
        iq2r_glm53_gate_bytes(1024, entries, shape),
    )
    assert tuple(layer.w13_weight_scale.shape) == (
        experts,
        iq2r_glm53_v2_aux_bytes(1024, hidden, 4, gate=True, codebook_entries=entries),
    )
    assert tuple(layer.w2_weight.shape) == (experts, down.data_bytes)
    assert tuple(layer.w2_weight_scale.shape) == (
        experts,
        iq2r_glm53_v2_aux_bytes(rows, 512, 4, gate=False, codebook_entries=entries),
    )
    for name in ("w13_iq2r_tile_n", "w2_iq2r_tile_n"):
        parameter = getattr(layer, name)
        assert (tuple(parameter.shape), parameter.dtype) == ((1,), torch.int32)
    assert (method.tp_size, method.tp_rank, method.classes) == (4, 1, 4)


@pytest.mark.parametrize("model", sorted(_MODELS))
@pytest.mark.parametrize("tp_size", [1, 2])
def test_iq2r_glm53_packed_create_weights_stacks_wide_ranks(aiter_iq2r, tp_size, model):
    """A rank wider than the kernels (TP1: 2048, TP2: 1024) stacks its
    512-wide shards, each the size of a TP4 rank's stack."""
    tp4 = _glm53_packed_layer(_glm53_packed_method(4, 0, 512, 1024, model))
    method = _glm53_packed_method(tp_size, 0, 2048 // tp_size, 1024, model)
    layer = _glm53_packed_layer(method)
    shards = 4 // tp_size
    assert method.shards == shards
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        assert tuple(getattr(layer, name).shape) == (
            shards,
            *getattr(tp4, name).shape,
        ), name


@pytest.mark.parametrize("model", sorted(_MODELS))
@pytest.mark.parametrize("entries", [512, 1024])
@pytest.mark.parametrize("load_dummy", [None, "empty", "zero", "xavier"])
def test_iq2r_glm53_packed_post_load_checks_codebook_range(
    aiter_iq2r, monkeypatch, load_dummy, entries, model
):
    method = _glm53_packed_method(
        tp_size=8, tp_rank=0, intermediate=256, entries=entries, model=model
    )
    layer = _glm53_packed_layer(method, device="cpu", num_experts=2)
    for aux in (layer.w13_weight_scale, layer.w2_weight_scale):
        aux.data[:, : entries * 8] = 0x38  # E4M3 1.0, inside the kernels' range
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(load_dummy=load_dummy),
    )
    monkeypatch.setattr(moe_mod, "_IQ2R_GLM53_WORKSPACES", {})
    method.process_weights_after_loading(layer)

    # subnormal: copies form wrongly (the last entry of copy 0 is checked too)
    layer.w2_weight_scale.data[0, 5 if load_dummy else entries * 8 - 1] = 0x05
    if load_dummy:  # dummy weights carry no codebooks
        method.process_weights_after_loading(layer)
    else:
        with pytest.raises(ValueError, match="codebook bytes"):
            method.process_weights_after_loading(layer)


@pytest.mark.parametrize("model", sorted(_MODELS))
def test_iq2r_post_load_checks_the_swiglu_clamp(aiter_iq2r, monkeypatch, model):
    """The kernels build the model's SwiGLU in: Flash's clamp at 10, none for GLM-5.3."""
    method = _glm53_packed_method(8, 0, 256, model=model)
    layer = _glm53_packed_layer(method, device="cpu", num_experts=2)
    for aux in (layer.w13_weight_scale, layer.w2_weight_scale):
        aux.data[:, : 512 * 8] = 0x38
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(load_dummy=None)
    )
    monkeypatch.setattr(moe_mod, "_IQ2R_GLM53_WORKSPACES", {})
    method.process_weights_after_loading(layer)
    layer.swiglu_limit = 7.0
    with pytest.raises(ValueError, match="clamp the SwiGLU"):
        method.process_weights_after_loading(layer)
    if model == "glm5next":
        del layer.swiglu_limit
        with pytest.raises(ValueError, match="clamp the SwiGLU"):
            method.process_weights_after_loading(layer)


@pytest.mark.parametrize("model,other", [("glm53", "glm5next"), ("glm5next", "glm53")])
def test_iq2r_rejects_the_other_models_shape(aiter_iq2r, model, other):
    method = _glm53_packed_method(4, 0, 512, model=model)
    _, hidden, experts, *_ = _MODELS[other]
    layer = torch.nn.Module()
    layer.has_bias = False
    layer.num_fused_shared_experts = 0
    with pytest.raises(ValueError, match="layout is for hidden"), torch.device("meta"):
        method.create_weights(
            layer,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=512,
            params_dtype=torch.bfloat16,
        )


def test_glm5_next_weights_mapping_targets_every_compiled_tensor(aiter_iq2r):
    compiler = pytest.importorskip("aiter.quantization.iq2r.glm5.compile")
    from atom.model_loader.weight_names import CheckpointNameRewriter
    from atom.models.glm5_next import Glm5NextForConditionalGeneration

    layer = _glm53_packed_layer(_glm53_packed_method(4, 0, 512, model="glm5next"))
    rewriter = CheckpointNameRewriter(
        weights_mapping=Glm5NextForConditionalGeneration.weights_mapping
    )
    mapped = sorted(
        rewriter.rewrite(key)
        for projection in ("gate_up", "down")
        for key in compiler.iq2r_compiled_tensor_keys(
            3, projection, "model.language_model."
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
    [(16, 1, 0), (4, 2, 0), (1, 2, 0), (8, 1, 1)],
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


@pytest.mark.parametrize("model", sorted(_MODELS))
@pytest.mark.parametrize("entries", [512, 1024])
@pytest.mark.parametrize("tp_size,tp_rank", [(4, 3), (8, 5), (2, 1), (1, 0)])
def test_iq2r_glm53_packed_loader_slices_the_rank_shard(
    aiter_iq2r, tp_size, tp_rank, entries, model
):
    from aiter.iq2r_glm53 import (
        iq2r_glm53_gate_bytes,
        iq2r_glm53_slice_gate,
        iq2r_glm53_v2_aux_bytes,
        iq2r_glm53_v2_slice_down_aux,
        iq2r_glm53_v2_slice_gate_aux,
    )
    from aiter.ops.iq2r_format import IQ2RMetadata, iq2r_slice_input_data

    _, hidden, _, rows, _ = _MODELS[model]
    shard = 2048 // tp_size
    method = _glm53_packed_method(tp_size, tp_rank, shard, entries, model)
    shape = method.shape
    down = IQ2RMetadata(logical_n=rows, logical_k=2048, codebook_entries=entries)
    generator = torch.Generator().manual_seed(tp_size)

    def full(columns):
        return torch.randint(0, 256, (1, columns), generator=generator).to(torch.uint8)

    def aux_bytes(n, k, gate):
        return iq2r_glm53_v2_aux_bytes(n, k, 4, gate=gate, codebook_entries=entries)

    weights = {
        "w13_weight": full(iq2r_glm53_gate_bytes(4096, entries, shape)),
        "w13_weight_scale": full(aux_bytes(4096, hidden, True)),
        "w2_weight": full(down.data_bytes),
        "w2_weight_scale": full(aux_bytes(rows, 2048, False)),
    }
    start = 2 * shard * tp_rank
    expected = {
        "w13_weight": iq2r_glm53_slice_gate(
            weights["w13_weight"], start, 2 * shard, entries, shape
        ),
        "w13_weight_scale": iq2r_glm53_v2_slice_gate_aux(
            weights["w13_weight_scale"], start, 2 * shard, 4096, 4, entries
        ),
        "w2_weight": iq2r_slice_input_data(
            weights["w2_weight"], down, shard * tp_rank, shard
        ),
        "w2_weight_scale": iq2r_glm53_v2_slice_down_aux(
            weights["w2_weight_scale"], shard * tp_rank, shard, 2048, 4, entries, shape
        ),
    }
    # The rank's shard has the shape create_weights allocates.
    assert expected["w13_weight_scale"].shape[1] == aux_bytes(2 * shard, hidden, True)
    assert expected["w2_weight_scale"].shape[1] == aux_bytes(rows, shard, False)
    assert expected["w13_weight"].shape[1] == iq2r_glm53_gate_bytes(
        2 * shard, entries, shape
    )
    if shard > 512:
        # A wide rank loads stacked 512-wide shards: those of TP4 ranks
        # tp_rank * shards .. + shards - 1.
        tp4 = _glm53_packed_method(4, 0, 512, entries, model)
        expected = {}
        for name in weights:
            parts = []
            for rank in range(tp_rank * shard // 512, (tp_rank + 1) * shard // 512):
                tp4.tp_rank = rank
                parts.append(tp4._glm53_tp_slice(name, weights[name]))
            expected[name] = torch.stack(parts)
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
    shared_calls = []

    def moe_out(*args, shared=None):
        calls.append(args)
        shared_calls.append(shared)

    monkeypatch.setattr(aiter_iq2r, "iq2r_glm53_moe_out", moe_out)
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
    assert shared_calls == [None]  # no shared expert attached
    with pytest.raises(ValueError, match="does not support"):
        method.apply(**{**arguments, "activation": moe_mod.ActivationType.Swiglu})
    with pytest.raises(ValueError, match="BF16 rows"):
        method.apply(**{**arguments, "x_scale": torch.ones(2, 1)})
    with pytest.raises(ValueError, match="router weights"):
        method.apply(**{**arguments, "apply_router_weight_on_input": True})


@pytest.mark.parametrize("model", sorted(_MODELS))
def test_iq2r_glm53_apply_fuses_the_attached_shared_expert(
    aiter_iq2r, monkeypatch, model
):
    """An attached shared expert goes to the IQ2R launches for small batches only."""
    _, hidden, experts, rows, _ = _MODELS[model]
    method = _glm53_packed_method(tp_size=4, tp_rank=0, intermediate=512, model=model)
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        w13_weight=torch.empty((experts, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((experts, 5), dtype=torch.uint8),
        w2_weight=torch.empty((experts, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((experts, 7), dtype=torch.uint8),
    )
    generator = torch.Generator().manual_seed(0)
    gate, up = (torch.randn((512, hidden), generator=generator) * 0.02 for _ in range(2))
    down = torch.randn((hidden, 512), generator=generator) * 0.02
    method.attach_shared_expert(layer, gate, up, down)
    packed = layer.iq2r_shared
    assert packed.shape == method.shape
    # GLM-5.3-Flash's shared down rows are zero-padded to the kernels' 4,608.
    assert packed.gate.numel() == 2 * 512 * hidden and packed.down.numel() == 512 * rows
    assert packed.gate_scales.numel() == 2 * 512 * hidden // 32
    assert packed.down_scales.numel() == 512 * rows // 32

    seen = []
    monkeypatch.setattr(method, "_glm53_workspace", lambda device: None)
    monkeypatch.setattr(
        aiter_iq2r, "iq2r_glm53_moe_out", lambda *args, shared=None: seen.append(shared)
    )
    limit = method.fused_shared_tokens()
    assert limit == aiter_iq2r.FUSED_SHARED_TOKENS
    for tokens in (1, limit, limit + 1):
        method.apply(
            layer=layer,
            x=torch.zeros((tokens, hidden), dtype=torch.bfloat16),
            topk_weights=torch.zeros((tokens, 8)),
            topk_ids=torch.zeros((tokens, 8), dtype=torch.int32),
            global_num_experts=experts,
            activation=moe_mod.ActivationType.Silu,
        )
    assert seen == [packed, packed, None]


def test_dequantize_linear_block_and_row_scales():
    from atom.models.deepseek_v2 import _dequantize_linear

    weight = torch.randn((256, 384)).to(torch.float8_e4m3fn)
    block = torch.rand((2, 3)) + 0.5
    linear = SimpleNamespace(weight=weight, weight_scale=block)
    expected = weight.float() * block.repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    assert torch.equal(_dequantize_linear(linear), expected)
    rows = torch.rand((256, 1))
    linear = SimpleNamespace(weight=weight, weight_scale=rows)
    assert torch.equal(_dequantize_linear(linear), weight.float() * rows)
    linear = SimpleNamespace(weight=weight.float())
    assert torch.equal(_dequantize_linear(linear), weight.float())


def test_glm5_next_moe_attaches_the_shared_expert_to_iq2r(monkeypatch):
    """Glm5NextMoE hands its FP8 shared expert to the IQ2R method after loading,
    unless ATOM_IQ2R_FUSE_SHARED=0 or the experts are not IQ2R."""
    glm5_next = pytest.importorskip("atom.models.glm5_next")
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    attached = []
    method.attach_shared_expert = lambda *args: attached.append(args)
    method.fused_shared_tokens = lambda: 256
    weight = torch.randn((2 * 64, 128)).to(torch.float8_e4m3fn)
    module = object.__new__(glm5_next.Glm5NextMoE)
    torch.nn.Module.__init__(module)
    module.experts = SimpleNamespace(quant_method=method)
    module.shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=weight, weight_scale=torch.ones((1, 1))),
        down_proj=SimpleNamespace(weight=weight[:64].T.contiguous()),
    )
    module._iq2r = True
    monkeypatch.setattr(
        glm5_next, "get_current_atom_config", lambda: SimpleNamespace(load_dummy=None)
    )
    module.process_weights_after_loading()
    assert module._iq2r_fused_shared_tokens == 256
    (experts, gate, up, down), = attached
    assert experts is module.experts
    assert torch.equal(gate, weight[:64].float()) and torch.equal(up, weight[64:].float())
    assert torch.equal(down, weight[:64].T.float())

    monkeypatch.setenv("ATOM_IQ2R_FUSE_SHARED", "0")
    module.process_weights_after_loading()
    assert module._iq2r_fused_shared_tokens == 0
    monkeypatch.delenv("ATOM_IQ2R_FUSE_SHARED")
    module._iq2r = False
    module.process_weights_after_loading()
    assert module._iq2r_fused_shared_tokens == 0 and len(attached) == 1
