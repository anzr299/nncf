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
from typing import Tuple

import numpy as np
import openvino.torch  # noqa
import pytest
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch._export import capture_pre_autograd_graph

import nncf
import nncf.common
import nncf.common.factory
from nncf.common.logging.track_progress import track
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.torch.fx.helpers import TinyImagenetDatasetManager
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node

IMAGE_SIZE = 64
BATCH_SIZE = 128


@pytest.fixture(name="tiny_imagenet_dataset", scope="module")
def tiny_imagenet_dataset_fixture():
    return TinyImagenetDatasetManager(IMAGE_SIZE, BATCH_SIZE).create_data_loaders()


@dataclass
class SanitySampleCase:
    model_id: str
    checkpoint_url: str
    top1_int8_ref: float
    ref_num_q: int
    ref_num_dq: int


MODELS = (
    SanitySampleCase(
        "resnet18",
        "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth",
        55.35,
        30,
        37,
    ),
)


def get_model(model_id: str, checkpoint_url: str, device: torch.device) -> torch.nn.Module:
    num_classes = 200  # 200 is for Tiny ImageNet, default is 1000 for ImageNet
    model = getattr(models, model_id)(weights=None)
    # Update the last FC layer for Tiny ImageNet number of classes.
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.to(device)
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def validate(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    top1_sum = 0.0
    with torch.no_grad():
        for images, target in track(val_loader, total=len(val_loader), description="Validation:"):
            images = images.to(device)
            target = target.to(device)

            # Compute output.
            output = model(images)

            # Measure accuracy and record loss.
            [acc1] = accuracy(output, target, topk=(1,))
            top1_sum += acc1.item()

        num_samples = len(val_loader)
        top1_avg = top1_sum / num_samples
    return top1_avg


def accuracy(output: torch.Tensor, target: torch.tensor, topk: Tuple[int, ...] = (1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def count_q_dq(model: torch.fx.GraphModule):
    q, dq = 0, 0
    for node in model.graph.nodes:
        if node.op == "call_function" and hasattr(node.target, "overloadpacket"):
            node_type = str(node.target.overloadpacket).split(".")[1]
            if node_type in ["quantize_per_tensor", "quantize_per_channel"]:
                q += 1
            elif node_type in ["dequantize_per_tensor", "dequantize_per_channel"]:
                dq += 1
    return q, dq

def npy_loader(path):
    return torch.from_numpy(np.load(path))

def fx_quantize(weight, scale, zp, quant_min, quant_max):
    res = torch.clamp(
        torch.round(weight * (1.0 / scale)) + zp, quant_min, quant_max
    )
    return res

@pytest.mark.parametrize("test_case", MODELS)
def test_sanity(test_case: SanitySampleCase, tiny_imagenet_dataset):
    with disable_patching():
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = get_model(test_case.model_id, test_case.checkpoint_url, device)
        _, val_dataloader, calibration_dataset = tiny_imagenet_dataset

        def transform_fn(data_item):
            return data_item[0].to(device)

        calibration_dataset = nncf.Dataset(calibration_dataset, transform_fn)
        # loaded_value = npy_loader("C:/Users/anazir/Downloads/tensor (1).npy")
        # print(loaded_value)
        # loaded_value = torch.load('saved_tensor')
        with torch.no_grad():
            ex_input = torch.ones((1, 3, 224, 224))
            model.eval()
            exported_model = capture_pre_autograd_graph(model, args=(ex_input,))
            quantized_model = nncf.quantize(exported_model, calibration_dataset)
            nncf.common.factory.NNCFGraphFactory.create(quantized_model).visualize_graph('graph1.dot')
            for node in quantized_model.graph.nodes:
                if(node.name == '_param_constant0'):
                    weight = get_tensor_constant_from_node(node, quantized_model)
                elif(node.name == 'conv2d_zero_point_0'):
                    zp = get_tensor_constant_from_node(node, quantized_model)
                elif(node.name == 'conv2d_scale_0'):
                    scale = get_tensor_constant_from_node(node, quantized_model)
                elif(node.name == 'quantize_per_channel_default'):
                    quantized_value = node.target(weight, scale, zp, 0, -128, 127, torch.uint8)
                elif(node.name == 'dequantize_per_channel_default'):
                    dequantized_value = node.target(quantized_value, scale, zp, 0, -128, 127, torch.uint8)
            # torch.save(scale, 'FX_int_8_scale')
            scale = scale.reshape([64,1,1,1])
            zp = zp.reshape([64, 1, 1, 1])
            print(scale.shape, weight.shape)
            int8_weight = fx_quantize(weight, scale, zp, -128, 127)
            print(torch.all(quantized_value == int8_weight))
            # for node in quantized_model.graph.nodes:
            #     if(node.name == 'conv2d_scale_0_updated_constant0'):
            #         scale = get_tensor_constant_from_node(node, quantized_model)
            #     elif(node.name == 'compressed_weight_updated_constant0'):
            #         int8_weight = get_tensor_constant_from_node(node, quantized_model)
            # dequantized_value = torch.mul(int8_weight, scale)
            # print(dequantized_value)
            torch.save(int8_weight, 'FX_int8')
            # print(torch.all(dequantized_value == loaded_value))
            # quantized_model = torch.compile(quantized_model, backend="openvino", options = {"device" : "CPU", "model_caching" : True, "cache_dir": "./model_cache"})
        
        # _ = quantized_model(ex_input)
        # top1_int8 = validate(val_dataloader, quantized_model, device)
        # print(top1_int8)
        # assert np.isclose(top1_int8, test_case.top1_int8_ref, atol=0.1)

        # num_q, num_dq = count_q_dq(quantized_model)
        # assert num_q == test_case.ref_num_q
        # assert num_dq == test_case.ref_num_dq
