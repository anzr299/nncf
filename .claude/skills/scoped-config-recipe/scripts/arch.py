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
List the weights NNCF will compress and summarize them by family.

    python arch.py <model> --out nodes.json

nodes.json maps each NNCF graph node name (what ScopedConfig matches; it has no layer index) to its HF
weight name, layer, op type, shape and size. The printout shows each family's share of the weights.
"""

import argparse
import collections

from common import default_device
from common import family_of
from common import layer_of
from common import load_model
from common import write_json

import nncf
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.algorithms.weight_compression.algorithm import get_weight_compression_configuration
from nncf.torch.model_creation import wrap_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default="nodes.json")
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()

    model, _ = load_model(args.model, args.device)
    num_params = {name: p.numel() for name, p in model.named_parameters()}
    example = {k: v.to(args.device) for k, v in model.dummy_inputs.items()}
    wrapped = wrap_model(model, example_input=example, trace_parameters=True)
    algo = WeightCompression(
        **get_weight_compression_configuration(mode=nncf.CompressWeightsMode.INT4_ASYM, ratio=1.0), subset_size=1
    )
    algo.set_backend_entity(wrapped)
    params, _, _ = algo.get_weight_compression_parameters(wrapped, wrapped.build_graph())

    nodes = [
        {
            "node": wp.node_with_weight.node_name,
            "weight": wp.weight_name,
            "node_type": wp.node_with_weight.node_type,
            "layer": layer_of(wp.weight_name),
            "shape": list(wp.weight_shape),
            "in_features": wp.weight_shape[wp.reduction_axes[0]],
            "num_weights": int(wp.num_weights),
        }
        for wp in params
    ]
    compressed = {n["weight"] for n in nodes}
    uncompressed = collections.Counter()
    for name, count in num_params.items():
        if name not in compressed:
            uncompressed[family_of(name)] += count
    arch = {"model": args.model, "config": model.config.to_diff_dict(), "nodes": nodes, "uncompressed": uncompressed}
    write_json(arch, args.out)

    total = sum(n["num_weights"] for n in nodes)
    fams = collections.defaultdict(list)
    for n in nodes:
        fams[family_of(n["weight"])].append(n)
    print(f"{'family':60s} {'count':>5s} {'shape':>20s} {'share':>7s}  node name")
    for fam, ns in sorted(fams.items(), key=lambda kv: -sum(n["num_weights"] for n in kv[1])):
        share = sum(n["num_weights"] for n in ns) / total
        print(f"{fam:60s} {len(ns):5d} {str(tuple(ns[0]['shape'])):>20s} {100 * share:6.2f}%  {ns[0]['node']}")
    print(f"{len(nodes)} compressible weights, {total / 1e9:.2f}B params -> {args.out}")


if __name__ == "__main__":
    main()
