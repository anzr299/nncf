# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example: REAP expert pruning of a Mixture-of-Experts (MoE) LLM for the OpenVINO backend.

REAP (Router-weighted Expert Activation Pruning) reduces the *number of experts* per MoE
layer (e.g. 8 -> 6 at ratio=0.25) by dropping the lowest-saliency experts, where saliency
``S_j = mean_{x active}( g_j(x) * ||E_j(x)||_2 )`` combines the router probability with the
expert output norm. It is data-aware, so a calibration dataset is required.
"""

import time
from functools import partial

import numpy as np
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf


def main():
    MODEL_ID = "optimum-internal-testing/tiny-random-qwen3_moe"
    OUTPUT_DIR = "qwen3_moe_reap"
    RATIO = 0.25  # fraction of experts to remove per MoE layer
    SUBSET_SIZE = 128  # number of calibration samples
    SEQ_LEN = 512
    EVAL_TASKS = ["gsm8k"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False)

    # ---- Calibration dataset ----
    # Each text sample is tokenized into the OpenVINO causal-LM input layout. NNCF runs the
    # model over these samples to collect router probabilities and per-expert output norms
    # that drive the REAP saliency score.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x["text"]) > 128).select(range(SUBSET_SIZE))

    def transform_fn(data, tokenizer):
        tokenized = tokenizer(data["text"], return_tensors="np", truncation=True, max_length=SEQ_LEN)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": np.arange(input_ids.shape[0], dtype=int),
        }

    calibration_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    # ---- REAP expert pruning ----
    start_t = time.time()
    model.model = nncf.experimental.reduce_experts(
        model.model,
        dataset=calibration_dataset,
        ratio=RATIO,
        method="reap",
        subset_size=SUBSET_SIZE,
    )
    print(f"Expert pruning done in {time.time() - start_t:.1f}s")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ---- Generation sanity check ----
    pruned = OVModelForCausalLM.from_pretrained(OUTPUT_DIR)
    messages = [{"role": "user", "content": "What is PyTorch?"}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    output = pruned.generate(input_ids, max_new_tokens=64)
    print(tokenizer.decode(output[0]))

    # ---- Evaluation with lm-eval (e.g. GSM8K) ----
    import lm_eval

    results = lm_eval.simple_evaluate(
        model="openvino",
        model_args=f"pretrained={OUTPUT_DIR}",
        tasks=EVAL_TASKS,
        batch_size="auto",
    )
    for task, metrics in results["results"].items():
        print(task, {k: v for k, v in metrics.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
