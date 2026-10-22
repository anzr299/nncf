# Precision guide

## Priors per weight family
| family | prior | why |
|---|---|---|
| router `mlp.gate`, tiny gates (`shared_expert_gate`, `linear_attn.in_proj_a/b`) | INT8_ASYM (not BF16) | largest INT4 error, <0.1% of weights; INT8 per-channel is ~1-2% weight error, enough |
| linear attention / SSM (`linear_attn.in_proj_qkv/z`, `out_proj`) | INT8 if a small share | errors persist in the recurrent state; +3.4 MMLU-Pro on Qwen3.6-35B |
| full attention `self_attn.*` | INT8 (k/v always: tiny under GQA) | most sensitive inputs by NNCF score |
| shared expert (`mlp.shared_expert.*`) | INT8 | sees every token; on Qwen3.6-35B +0.06 GiB cut proxy KL 31% and added +0.52 LAMBADA (p=0.045) |
| routed experts (packed 3D, ~90% of an MoE) | base; first to demote | each expert sees few tokens; on the 35B, asym (+0.12 GiB) and INT8 on the UD-bumped layers (+0.73 GiB) gave nothing on LAMBADA; only g64 (+0.7 GiB) closed the last ~0.4 pt |
| dense MLP | base; g64 if the base is g128 | dense models lose more at g128 than MoE |
| embeddings, lm_head | INT8_ASYM | demoting embeddings to INT4 hurt on Qwen3.8-27B |

Check the share in `arch.py` before copying a prior: SSM is 2.9% of Qwen3.6-35B but 20.6% of Qwen3.8-27B.
Prefer INT4_ASYM over INT4_SYM at the same group size unless the budget is tight (asym won 60/61
weights on the 27B). Reference quants often bump the first block and the last few blocks; NNCF has no
5/6-bit integer modes, so copy a bump as a finer group or INT8.

## Measured on Qwen3.6-35B-A3B (thinking-on MMLU-Pro, 3617 questions)
| recipe | GiB | MMLU-Pro | proxy KL |
|---|---|---|---|
| BF16 | 64.6 | 83.74 | 0 |
| INT8_ASYM | 32.4 | 83.63 | 0.0025 |
| INT4_ASYM g64, SSM/attn/shared INT8, gates BF16 | 18.54 | 82.00 | 0.0070 |
| INT4_SYM g128, attn/SSM/router/emb INT8 (`assets/` minus shared expert) | 17.70 | 82.97 | 0.0139 |
| same but SSM left INT4 | 17.24 | 79.54 | 0.0302 |
| INT4_ASYM g64 everywhere | 17.84 | 78.10 | 0.0356 |

LAMBADA (0-shot, deterministic): BF16 74.67 | INT4_ASYM g64 + SSM/attn/shared INT8 (18.54 GiB) 74.50, no
significant loss | `assets/` recipe = row 4 + shared expert INT8 (17.75 GiB) 74.07 | row 4 (17.70 GiB) 73.55 |
INT4_ASYM g64 everywhere 72.79. LAMBADA agrees with the proxy order exactly.

Proxy KL ranks the MMLU-Pro rows with Spearman 0.94. The INT4 losses come from long reasoning that never finishes
within the token budget; thinking-off GSM8K put every INT4 recipe within noise of BF16, so never choose
recipes with a short thinking-off eval.

## Pitfalls
- Load on one device: `device_map` sharding gives constant logits on Qwen3.5/3.6.
- Torch NNCF has no data-aware compression (AWQ, scale estimation, data-based mixed precision) for grouped_mm MoE experts.
- `nncf.strip` imports torchao; `save_pretrained` on GPU can OOM on packed experts (move to CPU first).
