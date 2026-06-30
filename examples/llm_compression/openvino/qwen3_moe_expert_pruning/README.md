# MoE Expert Pruning (REAP) of an LLM with OpenVINO

This example reduces the number of experts in a Mixture-of-Experts (MoE) LLM using
**REAP** (Router-weighted Expert Activation Pruning) and the OpenVINO backend, then
evaluates the result with [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
on GSM8K.

Unlike weight quantization, expert pruning reduces the *number of experts* per MoE layer
(e.g. 8 → 6 at `RATIO = 0.25`). REAP scores each expert by its router-weighted output
magnitude over a calibration set and drops the lowest-saliency experts, renormalizing the
router over the survivors. Because it is data-aware, a calibration dataset is required.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script defaults to `optimum-internal-testing/tiny-random-qwen3_moe` for a fast smoke
test. To prune a real model, edit the constants at the top of `main()` (e.g.
`MODEL_ID = "Qwen/Qwen3-30B-A3B"`, `RATIO`, `SUBSET_SIZE`, `EVAL_TASKS`).

## Steps in `main.py`

1. **Export** the MoE model to OpenVINO via optimum-intel.
2. **Calibration dataset**: tokenizes text into the OV causal-LM input layout
   (`input_ids`, `attention_mask`, `position_ids`, `beam_idx`) wrapped in an
   `nncf.Dataset`. NNCF runs the model over these samples to collect router
   probabilities and per-expert output norms.
3. **Pruning**: `nncf.experimental.reduce_experts(model, dataset=..., ratio=..., method="reap")`.
4. **Save + generate**: saves the pruned model and runs a short generation.
5. **Evaluate**: runs lm-eval on the pruned model (GSM8K by default).
