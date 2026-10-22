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
Proxy eval: KL(BF16 || candidate) and top-1 agreement, teacher-forced on the BF16 model's own reasoning traces.

    python proxy_kl.py data <model> --out data.jsonl [--tp N]            # once per model (needs vLLM)
    python proxy_kl.py eval <model> --data data.jsonl --candidate base=recipe_base.json \
        --candidate a=recipe_a.json [--candidate ckpt=/dir] [--teacher-device D --device D]

Devices default to the first two GPUs (both on one GPU if there is only one, CPU without a GPU); the
teacher and one candidate are resident at once.

A candidate is a recipe.json (compressed in memory, data-free) or a compressed checkpoint directory.
The teacher is loaded once for all candidates. Thinking traces are used because quantization damage on
reasoning models shows up as drift over long generations, which short thinking-off evals miss.
"""

import argparse
import json
import os
import random

import torch
from common import default_device
from common import load_model
from common import load_recipe


def build_data(args):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # CUDA is already initialized here
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm import SamplingParams

    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    questions = [
        mmlu[i]["question"] + "\n" + "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(mmlu[i]["options"]))
        for i in rng.sample(range(len(mmlu)), 36)
    ] + [gsm[i]["question"] for i in rng.sample(range(len(gsm)), 12)]
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q + "\nThink step by step."}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
        for q in questions
    ]  # fmt: skip
    llm = LLM(args.model, max_model_len=args.seqlen + 64, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=64)  # fmt: skip
    outs = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.seqlen, seed=0))
    with open(args.out, "w") as f:
        for p, o in zip(prompts, outs):
            prompt_ids = tok(p, add_special_tokens=False).input_ids
            ids = (prompt_ids + list(o.outputs[0].token_ids))[: args.seqlen]
            f.write(json.dumps({"start": len(prompt_ids), "ids": ids}) + "\n")
    print(f"{len(prompts)} reasoning traces -> {args.out}")


@torch.no_grad()
def kl_and_top1(teacher, student, ids, start):
    """Token-level KL(teacher || student) and top-1 agreement on the generated part of one sequence."""
    x = torch.tensor([ids])
    lt = teacher(x.to(teacher.device)).logits[0, start - 1 : -1].float()
    ls = student(x.to(student.device)).logits[0, start - 1 : -1].float().to(lt.device)
    kl, top1 = [], []
    for a in range(0, lt.shape[0], 256):  # chunks keep the vocab-size softmax small
        pt, ps = torch.log_softmax(lt[a : a + 256], -1), torch.log_softmax(ls[a : a + 256], -1)
        kl.append((pt.exp() * (pt - ps)).sum(-1))
        top1.append(pt.argmax(-1) == ps.argmax(-1))
    return torch.cat(kl).cpu(), torch.cat(top1).cpu()


def evaluate(args):
    import nncf

    with open(args.data) as f:
        data = [json.loads(line) for line in f]
    teacher, _ = load_model(args.model, args.teacher_device)
    for cand in args.candidate:
        label, _, path = cand.partition("=")
        if path.endswith(".json"):
            student, _ = load_model(args.model, args.device)
            example = {k: v.to(args.device) for k, v in student.dummy_inputs.items()}
            student = nncf.compress_weights(student, dataset=nncf.Dataset([example]), **load_recipe(path))
        else:
            student, _ = load_model(path, args.device)
        kls, top1s = zip(*(kl_and_top1(teacher, student, d["ids"], d["start"]) for d in data))
        kl, top1 = torch.cat(kls), torch.cat(top1s)
        late = torch.cat([k[len(k) // 2 :] for k in kls])
        result = {"label": label, "path": path, "kl": kl.mean().item(), "kl_late_half": late.mean().item(),
                  "kl_p99": kl.quantile(0.99).item(), "top1": top1.float().mean().item()}  # fmt: skip
        print(json.dumps(result))
        with open(args.results, "a") as f:
            f.write(json.dumps(result) + "\n")
        del student
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("model")
    d.add_argument("--out", default="proxy_data.jsonl")
    d.add_argument("--seqlen", type=int, default=2048)
    d.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    d.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    e = sub.add_parser("eval")
    e.add_argument("model", help="BF16 base model (the teacher)")
    e.add_argument("--data", required=True)
    e.add_argument("--candidate", action="append", required=True, help="LABEL=recipe.json or LABEL=<checkpoint dir>")
    e.add_argument("--teacher-device", default=default_device(0))
    e.add_argument("--device", default=default_device(1))
    e.add_argument("--results", default="proxy_results.jsonl")
    args = ap.parse_args()
    build_data(args) if args.cmd == "data" else evaluate(args)


if __name__ == "__main__":
    main()
