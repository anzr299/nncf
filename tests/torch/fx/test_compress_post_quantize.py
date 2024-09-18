import torch
import torch.fx
import nncf
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from torch._export import capture_pre_autograd_graph
from nncf.common.factory import NNCFGraphFactory
from nncf.torch.quantization.layers import SymmetricWeightsDecompressor
from nncf.experimental.torch.fx.transformations import constant_update_transformation_builder, constant_update_fn
from torch.ao.quantization.fx.utils import create_getattr_from_value

class ShortTransformer(torch.nn.Module):
    def __init__(self, in_features, num_embeddings, share_weights=False):
        super().__init__()
        self.wte = torch.nn.Embedding(num_embeddings, in_features)
        self.linear = torch.nn.Linear(in_features, in_features)
        self.lm_head = torch.nn.Linear(in_features, num_embeddings)

        if share_weights:
            self.lm_head.weight = self.wte.weight

    def forward(self, input_ids):
        x = self.wte(input_ids)
        x = self.linear(x)
        res = self.lm_head(x)
        return res
    
model = ShortTransformer(5,10).eval()

def pattern(weight, scale, zero_point, mid, low, high, dtype):
    quantized = torch.ops.quantized_decomposed.quantize_per_channel.default(
        weight, scale, zero_point, mid, low, high, dtype
    )
    dequantized = torch.ops.quantized_decomposed.dequantize_per_channel.default(
        quantized, scale, zero_point, mid, low, high, dtype
    )
    return dequantized

def replacement(x, scale, zero_point, mid, low, high, dtype):
    # dequantized = torch.ops.quantized_decomposed.dequantize_per_channel.default(
    #     x, scale, zero_point, mid, low, high, dtype
    # )
    return torch.mul(x, torch.unsqueeze(scale, 1))

def constant_compression_transformation(model: torch.fx.GraphModule, pattern, replacement):
     torch.fx.subgraph_rewriter.replace_pattern(model, pattern, replacement)
     return model

with disable_patching():
    torch.manual_seed(42)
    model = ShortTransformer(5, 10).to('cpu')
    input_ids = [torch.randint(0, 10, (5,)) for i in range(10)]
    exported_model = capture_pre_autograd_graph(model, args=(input_ids[0],))
    exported_model(input_ids[0])
    count = 0
    for buffer in exported_model.buffers():
        count +=1
    print("Quantized Model Buffer:",count)
    quantized_model = nncf.quantize(exported_model, 
                                    calibration_dataset=nncf.Dataset(input_ids), 
                                    fast_bias_correction=False,
                                    subset_size=1)
    
    # compressed_model.recompile()
    # for node in quantized_model.graph.nodes:
    #     if node.target == torch.ops.quantized_decomposed.quantize_per_channel.default:
    #         input_tup = []
    #         for i in node.args:
    #             if isinstance(i, torch.fx.Node):
    #                 input_tup.append(getattr(quantized_model, i.target))
    #             else:
    #                 input_tup.append(i)
    #         result = node.target(*tuple(input_tup))
    #         constant_update_fn(quantized_model, node, result, 0)

#transformation
# compressed_model = constant_compression_transformation(quantized_model, pattern, replacement)
# compressed_model.graph.eliminate_dead_code()

count = 0
# compressed_model.recompile()
# quantized_model = torch.fx.GraphModule(quantized_model, quantized_model.graph)
for buffer in quantized_model.buffers():
    count +=1
print("Compressed Model Buffers: ", count)
for node in quantized_model.graph.nodes:
     if(node.name == 'mul_1'):
        # node_pre_value = getattr(quantized_model, node.name)
        input_tup = []
        for i in node.args:
            if isinstance(i, torch.fx.Node):
                if(i.op == 'get_attr'):
                    print(i, i.prev, i.next)
                    input_tup.append(getattr(quantized_model, i.target))
                else:
                    const_value = getattr(quantized_model, i.args[0].target)
                    input_tup.append(i.target(const_value, i.args[1]))
        print(input_tup)
        result = node.target(*tuple(input_tup))
        print(result, result.dtype, result.shape)
# print(compressed_model)
# compiled = torch.compile(compressed_model, backend='openvino')
# print(compiled(input_ids[0]))
# compressed_model = nncf.compress_weights(exported_model)
NNCFGraphFactory.create(quantized_model).visualize_graph("graph1.dot")