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
Per-weight quantization types chosen by a reference GGUF on local disk (e.g. an Unsloth UD quant).

    python reference.py <file.gguf> --nodes nodes.json [--out reference.json]

Needs the `gguf` package (llama.cpp's gguf-py); its HF -> GGUF tensor name map does the matching. Only the
header is read. Prints, per family, the most common type and the layers that differ from it.
"""

import argparse
import collections
import re

import gguf
from common import family_of
from common import read_json
from common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out", default="reference.json")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.gguf)
    arch_name = reader.fields["general.architecture"].contents()
    arch = next(a for a, name in gguf.MODEL_ARCH_NAMES.items() if name == arch_name)
    name_map = gguf.get_tensor_name_map(arch, reader.fields[f"{arch_name}.block_count"].contents())
    types = {t.name.removesuffix(".weight"): t.tensor_type.name for t in reader.tensors}

    result, unmapped = {}, []
    for w in (n["weight"] for n in read_json(args.nodes)["nodes"]):
        name = name_map.get_name(w.removesuffix(".weight").replace("language_model.", "")) or ""
        # Packed experts: one HF gate_up_proj, two GGUF tensors.
        found = {types[n] for n in (name, name.replace("gate_up", "gate"), name.replace("gate_up", "up")) if n in types}
        if found:
            result[w] = "/".join(sorted(found))
        else:
            unmapped.append(w)
    write_json(result, args.out)

    fams = collections.defaultdict(dict)
    for w, t in result.items():
        fams[family_of(w)][w] = t
    for fam, per in sorted(fams.items()):
        common = collections.Counter(per.values()).most_common(1)[0][0]
        other = {re.search(r"layers\.(\d+)", w).group(1): t for w, t in per.items() if t != common}
        print(f"{fam:60s} {common:8s} {other or ''}")
    print(f"mapped {len(result)} of {len(result) + len(unmapped)} NNCF weights -> {args.out}")
    if unmapped:
        print("not in the GGUF:", sorted({family_of(w) for w in unmapped}))


if __name__ == "__main__":
    main()
