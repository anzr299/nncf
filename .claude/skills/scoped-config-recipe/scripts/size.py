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
Predicted size of recipes and the precision each weight family gets.

    python size.py recipe.json [more.json ...] --nodes nodes.json --base int4_asym:128

Ends with a markdown table, one column per recipe, of the precision and bits each layer type gets.

Stored bits per weight, as NNCF's torch decompressors store them (group g; -1 = per channel):
    INT4_ASYM 4 + 20/g    INT4_SYM 4 + 16/g    INT8_ASYM 8 + 24/in    INT8_SYM 8 + 16/in    ignored 16
    INT2_SYM 2 + 16/g    INT3_SYM 3 + 16/g (OpenVINO backend only; the torch backend has no decompressor for them)
The base is the base mode on every weight with embeddings and lm_head at INT8_ASYM (NNCF's default).
"""

import argparse
import collections
import os
from pathlib import Path

from common import family_of
from common import read_json
from common import select


def bits(mode, group_size, node):
    g = node["in_features"] if group_size in (None, -1) else group_size
    return {"int2_sym": 2 + 16 / g, "int3_sym": 3 + 16 / g, "int4_asym": 4 + 20 / g, "int4_sym": 4 + 16 / g, "int8_asym": 8 + 24 / g, "int8_sym": 8 + 16 / g}[mode]


def decisions(recipe, nodes):
    """{node: (mode, group_size)}. NNCF precedence: a scoped config beats ignored_scope; the last one wins."""
    d = {n["node"]: (recipe["mode"], recipe.get("group_size", -1)) for n in nodes}
    if not recipe.get("all_layers") and not recipe["mode"].startswith("int8"):
        # Embeddings and lm_head get the backup mode unless all_layers=True.
        for n in nodes:
            if n["node_type"] == "embedding" or n["weight"].startswith("lm_head"):
                d[n["node"]] = (recipe.get("backup_mode", "int8_asym"), -1)
    for name in select(recipe.get("ignored_scope", {}), nodes):
        d[name] = ("ignored", None)
    for sc in recipe.get("scoped_configs", []):
        for name in select(sc, nodes):
            d[name] = (sc["mode"], sc.get("group_size", -1))
    return d


def fmt_layers(layers):
    """[0, 1, 2, 5] -> '0-2, 5'"""
    runs = []
    for x in sorted(layers):
        if runs and x == runs[-1][1] + 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def fmt_share(x):
    return f"{x:.0f}%" if x >= 10 else f"{x:.2f}%" if x >= 0.005 else "~0%"


def short_name(family):
    """model.layers.N.self_attn.q_proj.weight -> (self_attn, q_proj); model.embed_tokens.weight -> ('', embed_tokens)"""
    name = family.removesuffix(".weight")
    if "layers.N." not in name:
        return "", name.split(".")[-1]
    parent, _, leaf = name.split("layers.N.", 1)[1].rpartition(".")
    return parent, leaf


def joined(parent, leaves):
    """Shared prefix/suffix written once: (self_attn, [q_proj, k_proj]) -> self_attn.q/k_proj."""
    if not parent:
        return ", ".join(leaves)
    if len(leaves) > 1:
        pre = os.path.commonprefix(leaves)
        pre = pre[: pre.rfind("_") + 1]
        suf = os.path.commonprefix([x[::-1] for x in leaves])[::-1]
        suf = suf[suf.find("_") :] if "_" in suf else ""
        mids = [x[len(pre) : len(x) - len(suf)] for x in leaves]
        mids = sorted(mids) if pre else mids  # in_proj_a/b; keep model order for q/k/v/o_proj
        leaves = [pre + "/".join(mids) + suf] if all(mids) else [" / ".join(leaves)]
    return f"{parent}.{leaves[0]}" + (" (router)" if leaves == ["gate"] else "")


def cell(choices):
    """{(mode, group_size): [nodes]} of one row in one recipe -> 'INT4_SYM g128 (4.125)'."""
    parts = []
    for (m, g), ns in sorted(choices.items(), key=lambda kv: -sum(n["num_weights"] for n in kv[1])):
        w = sum(n["num_weights"] for n in ns)
        if m == "ignored":
            text = "BF16 (ignored)"
        elif g in (None, -1):
            text = f"{m.upper()} ({sum(bits(m, g, n) * n['num_weights'] for n in ns) / w:.2f})"
        else:
            text = f"{m.upper()} g{g} ({sum(bits(m, g, n) * n['num_weights'] for n in ns) / w:.3f})"
        layers = {n["layer"] for n in ns if n["layer"] is not None}
        if len(choices) > 1 and layers:
            text += f" layer{'s' if len(layers) > 1 else ''} {fmt_layers(layers)}"
        parts.append(text)
    return "; ".join(parts)


def layer_table(labels, ds, sizes, nodes, uncompressed):
    """
    Markdown table: one column per recipe, one row per layer type, largest first. Layer types that share a
    parent module and get the same precision in every recipe share a row if each is under 0.5% of the weights
    (always for the weights outside the decoder layers, e.g. embed_tokens and lm_head).
    """
    total = sum(n["num_weights"] for n in nodes) + sum(uncompressed.values())
    fams = collections.defaultdict(list)
    for n in nodes:
        fams[family_of(n["weight"])].append(n)
    rows = {}
    for fam, ns in fams.items():
        parent, leaf = short_name(fam)
        layout = tuple(tuple(sorted((d[n["node"]], -1 if n["layer"] is None else n["layer"]) for n in ns)) for d in ds)
        mergeable = not parent or sum(n["num_weights"] for n in ns) / total < 0.005
        key = (parent, layout) if mergeable else (parent, leaf)
        rows.setdefault(key, (parent, [], []))
        rows[key][1].append(leaf)
        rows[key][2].extend(ns)

    print("| Layer type | Share | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for parent, leaves, ns in sorted(rows.values(), key=lambda r: -sum(n["num_weights"] for n in r[2])):
        cells = []
        for d in ds:
            choices = collections.defaultdict(list)
            for n in ns:
                choices[d[n["node"]]].append(n)
            cells.append(cell(choices))
        share = 100 * sum(n["num_weights"] for n in ns) / total
        print(f"| {joined(parent, leaves)} | {fmt_share(share)} | " + " | ".join(cells) + " |")
    if uncompressed:
        names = list(dict.fromkeys(uncompressed_kind(k) for k in uncompressed))
        share = 100 * sum(uncompressed.values()) / total
        print(f"| {', '.join(names)} | {fmt_share(share)} | " + " | ".join(["BF16 (not compressed)"] * len(ds)) + " |")
    print("| Total | | " + " | ".join(f"{g:.2f} GiB, {8 * g * 2**30 / total:.3f} bpw" for g in sizes) + " |")


def uncompressed_kind(family):
    if family.endswith(".bias"):
        return "biases"
    if "norm" in family:
        return "norms"
    return short_name(family)[1]


def gib(recipe, nodes, uncompressed):
    d = decisions(recipe, nodes)
    total = sum((16 if d[n["node"]][0] == "ignored" else bits(*d[n["node"]], n)) * n["num_weights"] for n in nodes)
    return (total + 16 * sum(uncompressed.values())) / 8 / 2**30, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="+")
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--base", default="int4_asym:128", help="MODE:GROUP_SIZE")
    args = ap.parse_args()

    arch = read_json(args.nodes)
    nodes, uncompressed = arch["nodes"], arch.get("uncompressed", {})
    mode, gs = args.base.split(":")
    base = {"mode": mode, "group_size": int(gs)}
    base_gib, _ = gib(base, nodes, uncompressed)
    print(f"base {args.base}: {base_gib:.2f} GiB")

    sizes, ds = [], []
    for path in args.recipes:
        size, d = gib(read_json(path), nodes, uncompressed)
        print(f"{path}: {size:.2f} GiB ({100 * (size / base_gib - 1):+.1f}% vs base)")
        sizes.append(size)
        ds.append(d)
    print()
    layer_table([Path(p).stem for p in args.recipes], ds, sizes, nodes, uncompressed)


if __name__ == "__main__":
    main()
