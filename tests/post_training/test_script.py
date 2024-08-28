from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import nncf
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching

from torch._export import capture_pre_autograd_graph
import time
from optimum.intel.openvino import OVModelForCausalLM
from torch.export._trace import _export
from transformers import DataCollatorForLanguageModeling

model_id = 'tinyllama/tinyllama-1.1b-step-50k-105b'

model_hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu").eval()
model_hf_original = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu").eval()
preprocessor = AutoTokenizer.from_pretrained(model_id)
# preprocessor.pad_token = preprocessor.eos_token
inputs = preprocessor("Using DistilBERT with ONNX Runtime!", return_tensors="pt")
ex_input = inputs['input_ids']

preprocessor.pad_token = preprocessor.eos_token
# data_collator = DataCollatorForLanguageModeling(preprocessor, mlm=False)

print(ex_input)
with torch.no_grad():
    with disable_patching():
        exported_model = capture_pre_autograd_graph(model_hf, args=(ex_input,))

        start_time = time.perf_counter()
        exported_model(ex_input)
        fx_inference_time = time.perf_counter() - start_time

        compiled_model = torch.compile(exported_model, backend='openvino', options = {"device" : "CPU", "model_caching" : True})
        compiled_model(ex_input)
        start_time = time.perf_counter()
        compiled_model(ex_input)
        compiled_model_inference_time = time.perf_counter() - start_time

        compressed_exported_model = nncf.compress_weights(exported_model)
        start_time = time.perf_counter()
        compressed_exported_model(ex_input)
        compressed_fx_inference_time = time.perf_counter() - start_time

        compressed_compiled_exported_model = torch.compile(compressed_exported_model, backend='openvino', options = {"device" : "CPU", "model_caching" : True})
        infer_compiled_model = compressed_compiled_exported_model(ex_input)

        start_time = time.perf_counter()
        infer_compiled_model = compressed_compiled_exported_model(ex_input)
        compiled_compressed_fx_inference_time = time.perf_counter() - start_time

ov_model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False, compile=False, stateful=False)
ov_model(ex_input)
start_time = time.perf_counter()
infer_compiled_model = ov_model(ex_input)
OV_FP32_model_inference_time = time.perf_counter() - start_time

ov_model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True, compile=False, stateful=False)
ov_model(ex_input)
start_time = time.perf_counter()
infer_compiled_model = ov_model(ex_input)
OV_compressed_model_inference_time = time.perf_counter() - start_time

print("Torch FX model inference time: ", fx_inference_time)
print("Compressed Torch FX model inference time: ", compressed_fx_inference_time)
print("Compiled and compressed torch fx model inference time: ", compiled_compressed_fx_inference_time)
print("OV FP32 model inference time: ", OV_FP32_model_inference_time)
print("OV Compressed model inference time: ", OV_compressed_model_inference_time)
print("Fx Compiled Model inference time: ", compiled_model_inference_time)

from whowhatbench import Evaluator

compiled_model(ex_input)
model_hf.model = compiled_model
print(model_hf.model)
print(model_hf.generate(**inputs))
# evaluator = Evaluator(base_model=model_hf_original, tokenizer=preprocessor, metrics=("similarity",))
# evaluator.dump_gt('./data/')
# all_metrics = evaluator.score(model_hf)
# print(all_metrics)