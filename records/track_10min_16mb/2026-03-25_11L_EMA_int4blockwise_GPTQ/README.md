## Record: 11L EMA + int4 blockwise GPTQ-lite (B=64) — *provisional / multi-seed TBD*

**Status:** Work in progress. Parent record: `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`. This variant replaces **per-row percentile clip search** (int6-style grid) with **4-bit blockwise quantization along each row** (`GPTQ_BLOCK_SIZE=64`, scales from per-block max |w|, codes in **−8…7**). Training stack is otherwise aligned with that record (EMA, tight SWA, late QAT, warmdown, etc.).

**Headline metrics (to finalize after 3-seed mean):** replace `TBD` in `submission.json` once `SEED=42` and `SEED=2042` logs are added.

### Change vs parent GPTQ-lite

| Aspect | Parent (2026-03-22) | This submission |
|--------|---------------------|-----------------|
| Attn/MLP export quant | int6 per row, 5 clip percentiles, min MSE | int4 blockwise per row, block size 64, scale = (block max abs) / 7 |
| Meta type | `int6` (legacy) / per-row scales | `int4_block` + `block_size` in `mixed_quantize_int6` |
| Training cost | Same | Same (quant runs **after** training on CPU) |

### Results (8×H100, `MAX_WALLCLOCK_SECONDS=600`, sliding eval stride=64)

Fill in the table after additional seeds; **seed 1337** below is from the run you captured (`RUN_ID=baseline_sp1024`).

| Seed | Steps @ cap | val_bpb (wallclock val) | post_ema val_bpb | Sliding BPB (s64) | int6+zlib bytes (code+model) |
|------|-------------|-------------------------|------------------|-------------------|--------------------------------|
| **1337** | 6851 | 1.1405 | 1.1395 | **1.1512** | 13,299,534 |
| 42 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| 2042 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

**Mean sliding BPB (3 seeds):** *TBD* | **Std:** *TBD*

Notes from seed **1337** run:

- `final_int6_roundtrip` val_bpb ≈ **1.1749** (dense roundtrip after int4-block export; larger gap vs sliding than parent — worth tuning block size / scheme).
- Peak VRAM ≈ **20.7 GiB** per GPU (log line).

### Architecture & training

Same as parent record (11L, XSA last 4, Partial RoPE, LN scale, VE, SmearGate, BigramHash, FA3, Muon+AdamW, EMA 0.997, SWA every 50 when scale&lt;0.2, late QAT @ 0.15, warmdown 3500 with wallclock LR). See parent README for full list.

### Quantization (this record’s `train_gpt.py`)

- **GPTQ-lite (int4 blockwise):** `GPTQ_BLOCK_SIZE` (default **64**); attn + MLP weight matrices only (same categories as parent int6 path).
- int8 path unchanged for other large tensors; control tensors fp32; zstd/zlib as in script.

### Reproducibility

```bash
SEED=1337 \
RUN_ID=replaceme \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=42` and `SEED=2042`, then update this README and `submission.json` with means.

### Files

| File | Purpose |
|------|---------|
| `train_gpt.py` | Frozen copy of training + export script (must run from this folder per challenge rules). |
| `train_seed1337.log` | Seed 1337 log (paste full file if you replace with a longer capture). |
| `train_seed42.log` | Placeholder — replace with full log. |
| `train_seed2042.log` | Placeholder — replace with full log. |
| `submission.json` | Metadata; **replace** `author`, `github_id`, and multi-seed aggregates when ready. |
