from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nncf
from torch._export import capture_pre_autograd_graph
import time

model_id = 'tinyllama/tinyllama-1.1b-step-50k-105b'

model_hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")

preprocessor = AutoTokenizer.from_pretrained(model_id)
inputs = preprocessor("Using DistilBERT with ONNX Runtime!", return_tensors="pt")
ex_input = inputs['input_ids']
exported_model = capture_pre_autograd_graph(model_hf, args=(ex_input,))

infer_exported_model = exported_model(ex_input)

start_time = time.perf_counter()
compressed_exported_model = nncf.compress_weights(exported_model)
exported_inference_time = time.perf_counter() - start_time
print("Exported Torch FX model inference time: ", exported_inference_time)

compiled_exported_model = torch.compile(compressed_exported_model, backend='openvino', )
compiled_exported_model(ex_input)

print("starting measurement inference")
start_time = time.perf_counter()
infer_compiled_model = compiled_exported_model(ex_input)
compiled_inference_time = time.perf_counter() - start_time
print("Compiled torch fx model inference time: ", compiled_inference_time)
