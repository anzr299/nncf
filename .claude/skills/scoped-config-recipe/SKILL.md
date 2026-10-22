---
name: scoped-config-recipe
description: Choose per-layer precision and group size for NNCF data-free weight compression of a PyTorch/HF LLM and write it as a recipe.json of nncf.compress_weights() parameters (scoped_configs + ignored_scope). Uses the model architecture, an optional local reference quantization (e.g. an Unsloth UD GGUF), a base precision, and a KL-vs-BF16 proxy eval. Use whenever the user asks which layers need higher precision, wants a mixed INT4/INT8 or "Unsloth-style" recipe, mentions scoped_configs or per-layer sensitivity, or wants roughly INT4 size without losing accuracy.
compatibility: Python with torch, transformers, datasets and NNCF develop; a CUDA GPU for models beyond a few B params; vLLM only to build the proxy data.
---

# Scoped-config recipe

Deliverable: `recipe.json`, the keyword arguments for `nncf.compress_weights()`:
```json
{
  "mode": "int4_sym", "group_size": 128, "ratio": 1.0,
  "scoped_configs": [
    {"patterns": ["model/(self_attn|linear_attn)/", "model/mlp/(gate|shared_expert|shared_expert_gate)/",
                  "model/embed_tokens/", "lm_head/"],
     "mode": "int8_asym", "group_size": -1}
  ],
  "ignored_scope": {"names": []}
}
```
Each scoped config and the ignored scope take nncf.Scope fields: `names` (exact node names), `patterns`
(`re.match` on node names), `types` (op type, e.g. `linear`, `_grouped_mm`, `embedding`), `subgraphs`
(`[{"inputs": [...], "outputs": [...]}]`), `validate`. Node names come from `nodes.json` and have **no layer
index** (`model/mlp/experts/_grouped_mm/72`), so per-layer choices use `names`. The last matching scoped
config wins and a scoped config overrides `ignored_scope`. See `assets/` for the
best measured cheap recipe on that model (17.75 GiB, LAMBADA 74.07 vs BF16 74.67; see `references/guide.md`).

## Inputs
- **Model**: BF16 checkpoint path or HF id.
- **Base precision**: e.g. `int4_asym` g128. Everything is measured against it. Layers may go *below* it
  when that pays for promoting more important ones; say why in your summary.
- **Size limit** (optional): if the user gives one (GiB or % over the base), stay within it. Otherwise
  keep upgrades that clearly pay off for their size and report the size vs the base.
- **Reference GGUF** (optional): local file, e.g. an Unsloth UD quant of the same model.

## Setup
```bash
S=<this skill's directory>/scripts
PY=python                      # any env with torch, transformers, datasets (+ gguf for step 2, vLLM for step 4 data)
export PYTHONPATH=<nncf checkout>/src   # skip if NNCF develop is pip-installed
```
Ask the user for the Python env if `$PY -c "import torch, transformers, nncf"` fails; don't install into
an env they didn't name. Scripts default to the first GPU (CPU if none); pass `--device` to override, and
check `nvidia-smi` for free GPUs on a shared machine.

## Steps
1. **Architecture**: `$PY $S/arch.py $MODEL --out nodes.json`. Prints each weight family's share of the
   weights. Read `references/guide.md` for the priors on the families you see.
2. **Reference** (if given): `$PY $S/reference.py <file.gguf> --nodes nodes.json`. Shows each family's
   type and the layers the reference bumped up or down. Translate its choices to what NNCF offers instead
   of copying them: F32 / BF16 / F16 / Q8_0 / Q6_K -> **INT8_ASYM** per-channel (near-lossless, and half the
   size of BF16; keep a weight uncompressed only if the proxy shows INT8 hurts it); one level above its
   family's base type (e.g. Q5_K over Q4_K) -> INT8 on small families, a finer INT4 group on large ones.
3. **Reason, then write candidates.** The priors in `guide.md` cover common families only; for every family
   in `arch.py`'s output, work out its precision from what the layer does, and write that reasoning down:
   - **How often it runs**: on every token (attention, shared expert, router, dense MLP) or only when
     routed (MoE experts, each seeing a small fraction of tokens). Always-on layers deserve bits first.
   - **Whether its error persists**: into a recurrent state (SSM / linear attention), the KV cache
     (k/v), or the routing decision (router logits pick the experts). Persistent errors compound.
   - **Where it writes**: output projections (o_proj, down_proj, out_proj) write straight into the residual
     stream and read heavy-tailed inputs; input projections feed a nonlinearity or attention.
   - **What promoting it costs**: its share from `arch.py`. Tiny, always-on families (gates, router, k/v
     under GQA) are almost free to put in INT8; huge routed families are where demotions pay for them.
   - **Position**: the first and last blocks are more fragile in most models; the reference shows where.
   **Start with the cheap upgrades.** Before anything costly, list every family that runs on every token
   or whose error persists and is a small share of the weights (router, gates, k/v, attention, shared
   expert, SSM projections when small, embeddings / lm_head). Moving all of them to INT8_ASYM usually costs
   a few percent of size and is the first candidate after the base; on Qwen3.6-35B it gave most of the
   achievable gain. INT8 is the upgrade target: it is near-lossless per channel, so BF16 (ignored_scope)
   is almost never worth its 2x size. Only then rank the remaining, expensive promotions (large or routed
   families, finer groups on the bulk of the weights) by expected benefit per GiB.
   Write a plain-base recipe plus 2–4 candidates that each test one idea (e.g. "cheap upgrades",
   "+ shared expert", "+ last-block bumps"), so the proxy can tell you which idea paid off. Check sizes:
   `$PY $S/size.py base.json a.json b.json --nodes nodes.json --base int4_asym:128`.
   It ends with a markdown table: one row per layer type with its share of the weights, one column per
   recipe with the precision and stored bits each layer type gets, and a total GiB / bpw row.
4. **Proxy**: build the trace set once, then score every candidate in one run (teacher loaded once):
   ```bash
   $PY $S/proxy_kl.py data $MODEL --out data.jsonl     # --tp N if the model needs several GPUs
   $PY $S/proxy_kl.py eval $MODEL --data data.jsonl \
       --candidate base=base.json --candidate a=a.json --candidate b=b.json
   ```
   `eval` keeps the BF16 teacher and one candidate in memory, on the first two GPUs by default (set
   `--teacher-device`/`--device`). If they don't fit, ask the user before falling back to CPU.
   Lower `kl` (and `kl_late_half`) is better. Differences under ~2x need a real eval to separate.
5. **Refine from the proxy** (1–2 rounds). For each candidate, compare its KL drop vs the base with the
   GiB it added, and update your reasoning:
   - a change that cut KL a lot for little size is a lever: push it further (more of that family, more
     layers, INT8 instead of a finer group) and look for similar layers you have not promoted yet;
   - a change that barely moved KL is wasted size: revert it and spend the bits elsewhere, or demote it;
   - if `kl_late_half` is much higher than `kl`, errors compound over long outputs: favour layers whose
     error persists (recurrent state, k/v, router) over one-shot ones;
   - if the best candidate is over the user's size limit, try the cheapest demotion of a large,
     low-impact family to bring it back.
   Write the next round's candidates from these conclusions, re-run `size.py` and the proxy on them (keep
   the base and the current best for comparison), and stop when a round no longer gives a clear gain.
6. **Deliver** the best `recipe.json` (within the size limit, if one was given), with:
   - the per-layer-type precision table from `size.py`, run on the base and the chosen recipe, pasted as is:
     ```
     | Layer type | Share | base | recipe |
     |---|---|---|---|
     | mlp.experts.gate_up_proj | 62% | INT4_SYM g128 (4.125) | INT4_SYM g128 (4.125) |
     | self_attn.q/k/v/o_proj | 0.79% | INT4_SYM g128 (4.125) | INT8_ASYM (8.01) |
     | mlp.experts.down_proj | 31% | ... | INT4_ASYM g64 (4.312) layers 1-38; INT8_ASYM (8.01) layer 39 |
     | Total | | 17.11 GiB, 4.240 bpw | 17.75 GiB, 4.400 bpw |
     ```
   - for each layer type that differs from the base, one line of reasoning and evidence (why the layer
     matters, the reference signal, and what the proxy showed when it was changed);
   - proxy KL of every candidate tried, per round, vs the base, and what each round changed.
   Tell the user to confirm with a thinking-on eval.

Apply it with:
```python
from common import load_recipe   # $S on sys.path
model = nncf.compress_weights(model, dataset=nncf.Dataset([example_input]), **load_recipe("recipe.json"))
```
