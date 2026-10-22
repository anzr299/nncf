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

"""Helpers shared by the scoped-config-recipe scripts."""

import contextlib
import json
import re

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def layer_of(weight_name):
    m = re.search(r"layers\.(\d+)\.", weight_name)
    return int(m.group(1)) if m else None


def family_of(weight_name):
    return re.sub(r"layers\.\d+\.", "layers.N.", weight_name)


def default_device(index=0):
    """cuda:<index> if that GPU exists, else the first GPU, else CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{index if index < torch.cuda.device_count() else 0}"


def load_model(model, device):
    """
    Load on ONE device. device_map sharding silently breaks hybrid Qwen3.5/3.6 forwards. Packed MoE experts
    are switched to grouped_mm so NNCF compresses them as 3D weights.
    """
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
    if hasattr(m, "set_experts_implementation"):
        with contextlib.suppress(Exception):  # dense models have no experts
            m.set_experts_implementation("grouped_mm")
    return m, tok


def select(selector, nodes):
    """Node names an nncf.Scope / ScopedConfig selects from nodes.json (NNCF semantics; no subgraphs)."""
    patterns = [re.compile(p) for p in selector.get("patterns", [])]
    return {
        n["node"]
        for n in nodes
        if n["node"] in selector.get("names", [])
        or any(p.match(n["node"]) for p in patterns)
        or n["node_type"] in selector.get("types", [])
    }


def load_recipe(path):
    """recipe.json -> nncf.compress_weights() kwargs with nncf.ScopedConfig / nncf.IgnoredScope objects."""
    import nncf

    def scope_fields(d):
        d = dict(d)
        if "subgraphs" in d:
            d["subgraphs"] = [nncf.Subgraph(**sg) for sg in d["subgraphs"]]
        return d

    r = read_json(path)
    kwargs = dict(r)
    kwargs["mode"] = nncf.CompressWeightsMode(r["mode"])
    kwargs["scoped_configs"] = [
        nncf.ScopedConfig(**{**scope_fields(sc), "mode": nncf.CompressWeightsMode(sc["mode"])})
        for sc in r.get("scoped_configs", [])
    ]
    kwargs["ignored_scope"] = nncf.IgnoredScope(**scope_fields(r.get("ignored_scope", {})))
    if "advanced_parameters" in r:  # e.g. {"group_size_fallback_mode": "adjust"}
        adv = dict(r["advanced_parameters"])
        if "group_size_fallback_mode" in adv:
            adv["group_size_fallback_mode"] = nncf.GroupSizeFallbackMode(adv["group_size_fallback_mode"])
        kwargs["advanced_parameters"] = nncf.AdvancedCompressionParameters(**adv)
    return kwargs
