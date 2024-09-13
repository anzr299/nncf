import torch
import nncf
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from torch._export import capture_pre_autograd_graph
from nncf.common.factory import NNCFGraphFactory
from nncf.torch.quantization.layers import SymmetricWeightsDecompressor

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

def pattern(x, scale, zero_point, mid, low, high, dtype):
    quantized = torch.ops.quantized_decomposed.quantize_per_channel.default(
        x, scale, zero_point, mid, low, high, dtype
    )
    dequantized = torch.ops.quantized_decomposed.dequantize_per_channel.default(
        quantized, scale, zero_point, mid, low, high, dtype
    )
    return dequantized

def replacement(x, scale, zero_point, mid, low, high, dtype):
    matmul_result = x
    return matmul_result

def constant_compression_transformation(model: torch.fx.GraphModule, pattern, replacement):
     compressed_constant_model = None
     print(torch.fx.subgraph_rewriter.replace_pattern(model, pattern, replacement))
     return model

with disable_patching():
        model = ShortTransformer(5, 10)
        input_ids = [torch.randint(0, 10, (5,)) for i in range(10)]
        exported_model = capture_pre_autograd_graph(model, args=(input_ids[0],))
        # compressed_model = nncf.compress_weights(exported_model, mode=mode)
        quantized_model = nncf.quantize(exported_model, 
                                        calibration_dataset=nncf.Dataset(input_ids), 
                                        fast_bias_correction=False,
                                        subset_size=1)
        # print(quantized_model)
        #transformation
        compressed_model = constant_compression_transformation(quantized_model, pattern, replacement)
compressed_model.graph.eliminate_dead_code()
compressed_model.recompile()
# compressed_model = nncf.compress_weights(compressed_model)
print(compressed_model(input_ids[0]))
NNCFGraphFactory.create(quantized_model).visualize_graph("graph.dot")