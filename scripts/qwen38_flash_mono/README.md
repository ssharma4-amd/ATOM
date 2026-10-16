# Qwen3.8-Flash-Next monokernel decode in ATOM

This branch adds a fused decode path for Qwen3.8-Flash-Next to ATOM. It runs on one AMD Instinct MI355X
(gfx950) with MXFP4 weights. Each of the model's 48 decoder layers runs as one FlyDSL kernel per decode
step. The kernel covers the layer's attention (QSA sparse attention or GatedDeltaNet), the PLE embedding
mix, the MoE with its fused shared expert, and the hyper-connection streams. The embedding, the final
mixer and `lm_head` stay ATOM's.

The fused path serves:
- decode steps of up to 4 requests;
- MTP speculative decoding with 1, 2 or 3 draft tokens. The verify step and the drafter's decoder layer
  both run on the same kernels.

Any other step falls back to ATOM's regular model.

What is compared below: both paths run in the same ATOM server on the same MXFP4 checkpoint, with the
same server line, prompts and client.
- **Fused path (`DECODE=mono`):** the per-layer kernels above.
- **Regular path (`DECODE=native`), the baseline:** ATOM's and aiter's own kernels, unchanged: aiter's
  FlyDSL MoE, ATOM's Triton W4A8 GEMM, GatedDeltaNet and QSA kernels, aiter's norms and `lm_head` GEMM.
  The only ATOM changes it runs are the few lines that load this checkpoint and the inline-send option
  below.
- **Both:** `ATOM_RUNNER_INLINE_SEND=1` (ours: ATOM's model runner sends each step's output from
  its main thread instead of handing it to another thread), and `ATOM_ENABLE_REPLAYSSM=0` when MTP
  is on (the fused verify needs it off; ATOM's own default turns it on with MTP).

The baseline is ATOM at this branch's base; ATOM's regular path keeps improving in later releases.

On GSM8K, HumanEval and MT-Bench prompts (one request at a time, up to 2,048 output tokens) the fused
path decodes about 3.1× faster than ATOM's regular path without MTP, and about 2.7× faster with three
draft tokens:

| Draft tokens | Set | Fused (tokens/s per request) | Regular path | Speedup |
|---|---|---|---|---|
| none | GSM8K | 331 | 107 | 3.10× |
| none | HumanEval | 331 | 107 | 3.09× |
| none | MT-Bench | 331 | 107 | 3.09× |
| 3 | GSM8K | 583 | 220 | 2.65× |
| 3 | HumanEval | 522 | 195 | 2.67× |
| 3 | MT-Bench | 500 | 186 | 2.70× |

GSM8K accuracy (lm_eval, 1,319 questions, 5-shot) is 96.0 % on the fused path against 96.2 % on the
regular path, which is within one standard error.

## Requirements

- One MI355X and a host with Docker. The scripts pull a pinned `rocm/pytorch` image (ROCm 10.0,
  PyTorch 2.13) and build their environment inside it.
- The MXFP4 checkpoint, built from the public BF16 one on CPU (below).
- Network access on the first run, for pip packages, aiter and the evaluation datasets.

## The checkpoint

`scripts/qwen38_flash_mxfp4/convert_mxfp4.py` turns `Qwen/Qwen3.8-Flash-Next` (BF16, 335 GiB) into the
MXFP4 checkpoint ATOM loads (163 GiB). It needs only `torch` and `safetensors`, and takes about a minute
on a large CPU host:

- linear and expert weights are stored as MXFP4: E2M1 values with one E8M0 scale per 32 values;
- activations are declared as MXFP8;
- `lm_head` and the hyper-connection mix weights are rounded to MXFP4 values and kept in BF16;
- everything else is copied unchanged.

```bash
python scripts/qwen38_flash_mxfp4/convert_mxfp4.py --hf Qwen3.8-Flash-Next --out Qwen3.8-Flash-Next-MXFP4
# or, in the container (downloads the pinned revision first unless BF16 points at a copy):
GPU=0 BF16=/path/to/Qwen3.8-Flash-Next bash scripts/qwen38_flash_mono/repro.sh convert
```

## Everything in one command

```bash
GPU=0 SMOKE=1 bash scripts/qwen38_flash_mono/build_and_repro.sh   # every run with a few requests
GPU=0 bash scripts/qwen38_flash_mono/build_and_repro.sh           # everything, about 20 h
```

It builds the checkpoint unless `MODEL` points at one. It then runs every test below for both decode
paths (`ARMS="mono native"`) and every draft count (`KS="0 1 2 3"`), and writes one table to
`$WORK/results/table.txt` with the regular path, the fused path and the speedup:

| Step | What it measures |
|---|---|
| `speed` | Decode tokens/s per request on the GSM8K, HumanEval and MT-Bench prompt sets (128 / 128 / 80), reasoning effort xhigh, up to 2,048 tokens, at 1–4 requests in flight (`CONCS`). |
| `accuracy` | GSM8K (lm_eval, 5-shot, chat template), HumanEval pass@1, GPQA-Diamond and MMLU-Pro, greedy, 4 requests in flight. |
| `context` | One request, 1,024 output tokens after random 1K, 8K, 128K and 250K-token prompts. |
| `serving` | 1–4 requests in flight, 1,024 output tokens after random 1K and 8K prompts. |

`STEPS`, `ARMS`, `KS` and `CONCS` select a subset. If a run's results directory already has a summary,
the run is skipped, so rerunning picks up where it stopped. `DRY_RUN=1` lists the runs without starting
them.

## One test at a time

`scripts/qwen38_flash_mono/repro.sh` runs one test. It starts the container and the server, runs the
client, prints a summary and stops the server. Each run writes `$WORK/runs/<time>-<command>-<decode>-k<K>/`
with the summary, the logs and the settings.

```bash
export MODEL=/path/to/Qwen3.8-Flash-Next-MXFP4 GPU=0
bash scripts/qwen38_flash_mono/repro.sh setup                          # once, a few minutes
K=0 bash scripts/qwen38_flash_mono/repro.sh speed                      # fused path, no MTP
DECODE=native K=3 bash scripts/qwen38_flash_mono/repro.sh speed        # regular path, 3 drafts
K=0 ISL="1024 8192" PROMPTS=10 REPS=3 bash scripts/qwen38_flash_mono/repro.sh bench
K=0 CONC=4 bash scripts/qwen38_flash_mono/repro.sh gsm8k
K=0 CONC=4 bash scripts/qwen38_flash_mono/repro.sh accuracy
bash scripts/qwen38_flash_mono/repro.sh help                           # every command and setting
```

The commands:
- `speed`: per-request tokens/s;
- `bench`: `benchmark_serving` with random prompts;
- `gsm8k`: lm_eval GSM8K, 5-shot with the chat template;
- `accuracy`: HumanEval, GPQA-Diamond and MMLU-Pro;
- `agentx`: aiperf's AgentX coding-agent traces;
- `server`: the server alone.

The first server start JIT-compiles aiter and the layer kernels, which takes about 6 minutes. Later
starts reuse the cache in `$WORK`. Run on an otherwise idle GPU; the scripts do not check for other
processes.

## Serving it directly

The fused path is part of ATOM's normal server. `ATOM_MONO_ENABLE` is on by default, and any
configuration the fused path supports is routed to it:

```bash
ATOM_RUNNER_INLINE_SEND=1 \
python -m atom.entrypoints.openai_server --model Qwen3.8-Flash-Next-MXFP4 --trust-remote-code -tp 1 \
  --max-num-seqs 4 --max-model-len 10240 --hf-overrides '{"_multimodal_config": null}'

# with MTP, 3 draft tokens (verify steps of up to 4 requests run on the fused path):
ATOM_RUNNER_INLINE_SEND=1 ATOM_ENABLE_REPLAYSSM=0 \
python -m atom.entrypoints.openai_server --model Qwen3.8-Flash-Next-MXFP4 --trust-remote-code -tp 1 \
  --max-num-seqs 4 --max-model-len 10240 --hf-overrides '{"_multimodal_config": null}' \
  --method mtp --num-speculative-tokens 3
```

| Variable | Default | Effect |
|---|---|---|
| `ATOM_MONO_ENABLE` | 1 | `0` serves ATOM's regular path. |
| `ATOM_RUNNER_INLINE_SEND` | 0 | `1` sends each step's output from the worker's main thread, which removes a thread hand-off between steps. |

The fused path needs:
- TP1 and a text-only deployment (the `--hf-overrides` above);
- the converted checkpoint;
- the bf16 KV cache with block size 16;
- `max_model_len` of at most 262,144 (less the draft count with MTP);
- `ATOM_ENABLE_REPLAYSSM=0` with MTP;
- no EP, DP, PP or context parallelism.

A configuration it does not serve logs the reason once and runs ATOM's regular path.
`docs/environment_variables.md` lists every condition.

## Code

| Path | What |
|---|---|
| `atom/models/qwen4_exp_mono/` | the decode path: routing (`dispatch.py`), binding ATOM's caches and weights (`state.py`, `weights.py`), the per-step tables, the single-request, grouped and draft runners |
| `atom/models/qwen4_exp_mono/kernels/` | the FlyDSL layer kernels: `kernel.py` builds a layer from the stages in `stages/` (QSA, GatedDeltaNet, PLE, MoE, hyper-connections) |
| `scripts/qwen38_flash_mxfp4/convert_mxfp4.py` | BF16 → MXFP4 checkpoint |
| `scripts/qwen38_flash_mono/` | `repro.sh`, `build_and_repro.sh`, the pinned pip constraints |
| `atom/benchmarks/qwen38_flash_mono/` | the per-request speed client (`single_request_bench.py`) and the accuracy evals (`accuracy_eval.py`); both work with any OpenAI-compatible server |
| `tests/test_qwen4_exp_mono_*.py` | routing, cache binding, grouped steps, build keys (`pytest tests/test_qwen4_exp_mono_*.py`) |
