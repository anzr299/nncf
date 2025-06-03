# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import warnings
from functools import partial

import torch
from datasets import load_dataset
from modelling import FXAutoModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.models.llama import LlamaTokenizerFast
from modelling import convert_and_export_with_cache, FXAutoModelForCausalLM

import nncf

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# tokens_done = 0
#         no_token = 0
#         while self._has_unfinished_sequences(
#             this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
#         ):
#             # prepare model inputs
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

#             # prepare variable output controls (note: some models won't accept all output controls)
#             model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
#             model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

#             if is_prefill:
#                 s_time = time.time()
#                 outputs = self(**model_inputs, return_dict=True)
#                 time_taken = time.time() - s_time
#                 print(f'Prefill Time {time_taken*1000:.4f}')
#                 is_prefill = False
#             elif no_token <= 10:
#                 s_time = time.time()
#                 outputs = model_forward(**model_inputs, return_dict=True)
#                 time_taken = time.time() - s_time
#                 tokens_done += time_taken
#                 no_token += 1
#             elif no_token == 11:
#                 print(f'Decode per token average time: {tokens_done*100:.4f}')
#                 outputs = model_forward(**model_inputs, return_dict=True)
#                 no_token += 1
#             else:
#                 outputs = model_forward(**model_inputs, return_dict=True)

def transform_fn(data: str, tokenizer: LlamaTokenizerFast):
    tokenized_text = tokenizer(data["text"], return_tensors="pt")
    input_ids = tokenized_text["input_ids"]
    attention_mask = tokenized_text["attention_mask"]

    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1

    inputs = (
        input_ids,
        position_ids.squeeze(0),
    )

    return inputs

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, *args, **kwargs):
        return self.model(args[0], {"input_pos": args[1][0:1]}).unsqueeze(0)

def main():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, num_hidden_layers=1)
    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    model, model_config, gen_config = convert_and_export_with_cache(model_hf)
    # torch.export.save(model, "HF_llama.pt2")
    # model = torch.export.load("executorch_llama.pt2")
    # model = torch.export.load("HF_llama.pt2")
    model = model.module()
    # model = WrapperModel(model=model)
    # Comment this text to turn off model optimization and measure performance of baseline model
    model = nncf.compress_weights(
        model,
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    )
    compressed_model_hf = FXAutoModelForCausalLM(model, model_config, generation_config=gen_config)

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    input_ids = tokenizer("What is PyTorch?", return_tensors="pt")
    print(input_ids)
    # inps = transform_fn({"text": "What is PyTorch?"}, tokenizer)
    # model = torch.compile(model, backend='openvino', options={"aot_autograd": True})
    # model(*inps)
    # compressed_model_hf.forward(*inps)
    output = compressed_model_hf.generate(input_ids["input_ids"], generation_config=gen_config)

    start_t = time.time()
    output = compressed_model_hf.generate(input_ids["input_ids"], generation_config=gen_config)
    elapsed_time = time.time() - start_t
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    import re
    output_text = re.findall(r'\b\w+\b', output_text)
    print(len(output_text))
    print(f'{(elapsed_time/len(output_text))*1000:.4f}')

    return output_text


if __name__ == "__main__":
    main()
