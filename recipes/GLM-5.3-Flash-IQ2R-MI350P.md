# GLM-5.3-Flash IQ2R v3 on one MI350P

MI350P: gfx950 (the MI350X/MI355X ISA), 128 CUs, 144 GiB. The IQ2R v3 checkpoint (`glm5next-packed-v3`,
120 GB) serves on one MI350P at TP1; MXFP4 does not fit (its routed experts alone are ~162 GB).

## Serving

```bash
python -m atom.entrypoints.openai_server \
  --model /models/GLM-5.3-Flash-IQ2Rv3 --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 1 --kv_cache_dtype fp8 --max-model-len 32768 \
  --enable_prefix_caching --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92
```

Set `AITER_FDA_USE_GLUON=0` when the image's Triton cannot build the Gluon FlashKDA kernels, as in the
GLM-5.3-Flash IQ2R serving image.

At TP1 the rank's intermediate is the full 2048. `Iq2rMoEMethod` loads it as four stacked 512-wide kernel
shards (each the slice a TP4 rank would load) and aiter's IQ2R MoE runs every shard per launch; the
result equals four TP4 ranks summed. TP2 (two shards) loads the same way. The weights take 110.6 GB of
the GPU at load; the shared expert runs inside the IQ2R launches up to 256 tokens
(`ATOM_IQ2R_FUSE_SHARED=0` keeps it a separate FP8 MLP). aiter's IQ2R rows for 128 CUs
(`gfx950,128`) are MI350P's launch choices.

## Measured

ATOM `benchmark_serving`, one MI350P, TP1, random 1024 in / 1024 out, ignore EOS, 8 x concurrency
prompts, FP8 KV cache, the server above. aiter's `gfx950,128` rows for GLM-5.3-Flash: IQ2R MoE, BF16
GEMMs (the KDA projections and the LM head stay BF16 in the checkpoint) and FP8 block-scale GEMMs.

| Concurrency | Output tok/s | Mean TPOT (ms) | Mean TTFT (ms) |
|---|---|---|---|
| 1 | 75.2 | 13.13 | 188 |
| 2 | 142.6 | 13.82 | 220 |
| 4 | 250.7 | 15.67 | 308 |
| 8 | 416.0 | 18.49 | 783 |
| 16 | 645.3 | 23.13 | 1722 |
| 32 | 953.3 | 30.38 | 3285 |
| 64 | 1344.0 | 42.32 | 5462 |

With the IQ2R rows alone (default BF16 and FP8 GEMM kernels) the same sweep gave 69.3, 128.3, 227.9,
393.5, 605.6, 945.3 and 1327.5 tok/s. Before the GEMM rows, a decode step at concurrency 1 (15.0 ms)
spent 20% in the IQ2R MoE, 35% in BF16 GEMMs and 11% in FP8 block-scale GEMMs.
