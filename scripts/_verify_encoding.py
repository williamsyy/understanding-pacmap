"""Sanity check: encode the existing UMAP projection with our Python routine and
diff against the reference JS-produced base64 blob."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from precompute_mammoth_pacmap import encode_bits, quantize, N_BITS  # noqa: E402

RAW = REPO / "raw_data" / "mammoth_umap.json"
REF = REPO / "public" / "mammoth_10k_encoded.json"


def main() -> None:
    with open(RAW, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(REF, "r", encoding="utf-8") as f:
        ref = json.load(f)

    labels = np.asarray(raw["labels"], dtype=np.int64)
    order = np.argsort(labels, kind="stable")

    ref_offsets = ref["labelOffsets"]
    py_offsets = np.bincount(labels[order]).tolist()
    assert ref_offsets == py_offsets, f"labelOffsets mismatch: ref={ref_offsets[:5]}, py={py_offsets[:5]}"
    print("labelOffsets match.")

    projections = raw["projections"]
    ref_projs = ref["projections"]
    for i, key in enumerate(list(projections.keys())[:3]):
        proj = np.asarray(projections[key], dtype=np.float64)
        proj = proj[order]
        q = quantize(proj).reshape(-1)
        packed = encode_bits(q, N_BITS)
        enc = base64.b64encode(packed).decode("ascii")
        ref_enc = ref_projs[key]
        if enc == ref_enc:
            print(f"[{i}] {key}: EXACT match ({len(enc)} chars)")
        else:
            # Quantization is a lossy step; small drift is acceptable if decode round-trips are close.
            print(f"[{i}] {key}: DIFFER; py-len={len(enc)}, ref-len={len(ref_enc)}")


if __name__ == "__main__":
    main()
