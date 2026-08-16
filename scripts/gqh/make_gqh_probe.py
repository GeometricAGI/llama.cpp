#!/usr/bin/env python3
"""Build a GQH probe artifact: re-emit a GGUF with eligible weights at a GQH rung.

This exists because a new qtype must round-trip through a REAL artifact before
anything is built at scale -- encode, load, registration counts, bit-exact decode.
It is not the production exporter (geo-quant Track 1 owns that); it is the
smallest thing that produces a loadable model carrying GQH tensors.

Only 2-D weight tensors whose row length is a multiple of 256 are converted, which
is the format's alignment rule (spec C4). Everything else is copied verbatim, so
the artifact is a realistic mix of GQH and stock tensors.

The 5-byte per-tensor header (float32 tensor_scale + uint8 grid code) is stripped
from each encoded payload and collected into the "geoquant.gqh.headers" KV, whose
wire is mirrored in src/llama-gqh.cpp.

usage: <geo-quant .venv python> make_gqh_probe.py <in.gguf> <out.gguf>
           [--rung gqh3|gqh2_h] [--layers N] [--repo PATH]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

GQH_MAGIC = b"GQHh1\0\0\0"
SUPERBLOCK = 256


def build_header_kv(entries):
    """entries: list of (name, qtype, tensor_scale float, grid_code int)."""
    blob = bytearray(GQH_MAGIC)
    blob += struct.pack("<II", len(entries), 0)
    for name, qtype, scale, code in entries:
        raw = name.encode("utf-8")
        blob += struct.pack("<I", len(raw)) + raw
        blob += struct.pack("<IfBxxx", qtype, scale, code)
    return bytes(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rung", default="gqh3", choices=["gqh3", "gqh2_h"])
    ap.add_argument("--layers", type=int, default=None,
                    help="only convert tensors in the first N blocks (keeps the probe cheap)")
    ap.add_argument("--repo", default=None, help="path to the geo-quant repo")
    ap.add_argument("--phantom-headers", type=int, default=0,
                    help="TEST HOOK: append N header entries naming tensors that are not "
                         "resident, standing in for an MTP block. llama.cpp creates MTP "
                         "tensors with TENSOR_SKIP unless the context is an MTP one, so a "
                         "real artifact carries entries the loader will never match.")
    ap.add_argument("--drop-headers", type=int, default=0,
                    help="TEST HOOK: omit N header entries for tensors that ARE converted. "
                         "The loader must refuse this -- such a tensor would abort at decode.")
    ap.add_argument("--fakequant", action="store_true",
                    help="store decode(encode(w)) as plain F16 instead of GQH. Same numerics, "
                         "stock qtype -- the control that separates real quantization damage "
                         "from a decode bug when the two artifacts are scored side by side.")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[2] / "gguf-py"))
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve()))
    import gguf
    from geoquant.formats import gqh

    if args.rung == "gqh3":
        qtype, encode, decode = gguf.GGMLQuantizationType.GQH3, gqh.encode3, gqh.decode3
    else:
        qtype, encode, decode = gguf.GGMLQuantizationType.GQH2_H, gqh.encode, gqh.decode

    reader = gguf.GGUFReader(args.src, "r")
    arch = str(bytes(reader.fields["general.architecture"].parts[-1]), "utf-8")
    writer = gguf.GGUFWriter(args.dst, arch)

    for field in reader.fields.values():
        if field.name == "general.architecture":
            continue
        writer.add_key_value(field.name, field.contents(), field.types[0],
                             sub_type=field.types[-1] if len(field.types) > 1 else None)

    headers = []
    n_gqh = n_skip = 0
    for t in reader.tensors:
        name = str(t.name)
        # ReaderTensor.shape is in ggml order, so ne[0] -- the row length the
        # superblocks run along -- is shape[0]. numpy arrays are the reverse.
        ne0 = int(t.shape[0])
        ne1 = int(t.shape[1]) if len(t.shape) > 1 else 1
        eligible = (
            name.endswith(".weight")
            and len(t.shape) == 2
            and ne0 % SUPERBLOCK == 0
            and t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
            and "token_embd" not in name
            and "output.weight" != name
        )
        if eligible and args.layers is not None:
            parts = name.split(".")
            if parts[0] == "blk" and int(parts[1]) >= args.layers:
                eligible = False

        if not eligible:
            writer.add_tensor(name, t.data, raw_dtype=t.tensor_type)
            n_skip += 1
            continue

        w = np.ascontiguousarray(t.data.reshape(ne1, ne0), dtype=np.float32)
        wire = encode(w)
        scale, code = struct.unpack_from("<fB", wire, 0)
        body = np.frombuffer(wire[5:], dtype=np.uint8)

        # Verify the round trip before it goes on disk -- a probe that ships a
        # payload nobody checked is not a probe.
        back = decode(wire, w.shape)
        assert back.shape == w.shape and np.isfinite(back).all(), name

        if args.fakequant:
            writer.add_tensor(name, back.astype(np.float16),
                              raw_dtype=gguf.GGMLQuantizationType.F16)
            n_gqh += 1
            continue

        # add_tensor_info wants a numpy-order BYTE shape and derives ne from it.
        body = body.reshape(ne1, body.size // ne1)
        writer.add_tensor(name, body, raw_shape=body.shape, raw_dtype=qtype)
        headers.append((name, int(qtype), float(scale), int(code)))
        n_gqh += 1

    for _ in range(args.drop_headers):
        headers.pop()

    for i in range(args.phantom_headers):
        headers.append((f"blk.{90 + i}.nextn.ffn_down.weight", int(qtype), 1.0, 0))

    if headers:
        writer.add_key_value("geoquant.gqh.headers", build_header_kv(headers),
                             gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT8)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    how = f"{args.rung} fake-quant as F16" if args.fakequant else args.rung
    print(f"\n{args.dst}: {n_gqh} tensor(s) at {how}, {n_skip} copied verbatim")


if __name__ == "__main__":
    main()
