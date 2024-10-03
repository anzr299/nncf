# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch._export import capture_pre_autograd_graph
import nncf
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from openvino.runtime import Core
import numpy as np
import os
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def _capture_model(model, inputs):
    with torch.no_grad():
        with disable_patching():
            return capture_pre_autograd_graph(model, (inputs,))

CHECKPOINT_URL = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"


def get_model(model_id: str, checkpoint_url: str, device: torch.device) -> torch.nn.Module:
    num_classes = 200  # 200 is for Tiny ImageNet, default is 1000 for ImageNet
    model = getattr(models, model_id)(weights=None)
    # Update the last FC layer for Tiny ImageNet number of classes.
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.to(device)
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def fx_quantize(weight, scale, zp, quant_min, quant_max):
    res = torch.clamp(
        torch.round(weight * (1.0 / scale)) + zp, quant_min, quant_max
    )
    res = res.to(torch.int8)
    return res

def get_fx_constants(quantized_model):
    for node in quantized_model.graph.nodes:
        if(node.name == '_param_constant0'):
            weight = get_tensor_constant_from_node(node, quantized_model)
        elif(node.name == 'conv2d_zero_point_0'):
            zp = get_tensor_constant_from_node(node, quantized_model)
        elif(node.name == 'conv2d_scale_0'):
            scale = get_tensor_constant_from_node(node, quantized_model)
        elif(node.name == 'quantize_per_channel_default'):
            quantized_value = node.target(weight, scale, zp, 0, -128, 127, torch.int8)
        elif(node.name == 'dequantize_per_channel_default'):
            dequantized_value = node.target(quantized_value, scale, zp, 0, -128, 127, torch.int8)

    scale = scale.reshape([64,1,1,1])
    zp = zp.reshape([64, 1, 1, 1])

    int8_weight = fx_quantize(weight, scale, zp, -128, 127)
    
    return weight, int8_weight

def compile_and_save(quantized_model, ex_input, cache_dir):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Removed existing cache directory: {cache_dir}")
    quantized_model = torch.compile(quantized_fx_model, backend="openvino", options = {"device" : "CPU", "model_caching" : True, "cache_dir": cache_dir})
    quantized_model(ex_input)

def ov_quantize(input_data_values, input_low, input_high, output_low, output_high, levels=256):
    quantum = (input_high-input_low)/255
    x_clipped = torch.clamp(input_data_values, input_low, input_high)

    # Perform the quantization step
    quantized = torch.round(((x_clipped - input_low) / (input_high - input_low)) * (levels - 1)-127)
    quantized = quantized.to(torch.int8)
    
    return quantized

def get_ov_constants(model):
    for node in model.get_ops():
        if node.get_friendly_name() == "quantize_per_channel/quantized_decomposed.quantize_per_channel.default/FakeQuantize":
            num_inputs = node.get_input_size()
            input_values = [] 
            for i in range(num_inputs):
                input_value = node.input_value(i)
                input_node = input_value.get_node()
                input_tensor = input_node.get_data()
                
                input_tensor = torch.from_numpy(input_tensor)
                input_values.append(input_tensor)

            input_data = input_values[0]
            input_low = input_values[1]
            input_high = input_values[2]
            output_low = input_values[3]
            output_high = input_values[4] 

            quantized_output_func = ov_quantize(input_data, input_low, input_high, output_low, output_high)
            break
    return input_data, quantized_output_func

def read_ov_model(cache_dir):
    cache_dir =  os.path.join(cache_dir,"model")
    if os.path.exists(cache_dir):
        model_file = None
        for file_name in os.listdir(cache_dir):
            if file_name.endswith(".xml"):
                model_file = os.path.join(cache_dir, file_name)
                break

        if model_file:
            print(f"Found model file: {model_file}")
            ov_model = core.read_model(model_file)
        else:
            print("No model XML file found in the cache directory.")
    else:
        print(f"Cache directory does not exist: {cache_dir}")
    return ov_model


# Torch FX model retrieval
torch.manual_seed(42)
device = torch.device("cpu")
pt_model = get_model("resnet18", CHECKPOINT_URL, device)
with torch.no_grad():
    ex_input = torch.ones((1, 3, 224, 224))
    calibration_dataset = nncf.Dataset([ex_input])
    pt_model.eval()
    exported_model = _capture_model(pt_model, ex_input)
    quantized_fx_model = nncf.quantize(exported_model, calibration_dataset)
    cache_dir = os.path.join(os.getcwd(),"model_cache")
    compile_and_save(quantized_fx_model, ex_input, cache_dir)

core = Core()
ov_model = read_ov_model(cache_dir)

ov_fp_weight, ov_int8 = get_ov_constants(ov_model)
fx_fp_weight, fx_int8 = get_fx_constants(quantized_fx_model)

unequal_mask = torch.ne(ov_int8, fx_int8)
different_values_tensor1 = ov_int8[unequal_mask]
different_values_tensor2 = fx_int8[unequal_mask]

# get number of unequal values
print(len(different_values_tensor1))

print("Values in tensor1 that are different:", different_values_tensor1)
print("Values in tensor2 that are different:", different_values_tensor2)


