# GQH qtypes (108/109/110/111)

GQH (Geo-Quant Hierarchical) is a low-bit weight family from geo-quant, branch
`feat/custom-format-family`. The authoritative wire spec is
`geoquant/formats/gqh.py` (`decode3` / `decode` / `decode_c` / `decode4`) and
`docs/design-briefs/custom_format_family_SPEC.md` in that repo. This file records
only what the llama.cpp side owns: the qtype numbers, how the per-tensor header
travels, and what the exporter must emit.

| qtype | name | bpw | superblock | per-tensor header |
|---|---|---|---|---|
| 108 | `gqh3` | 3.28125 | 105 B / 256 weights | yes |
| 109 | `gqh2_h` | 2.28125 | 73 B / 256 weights | yes |
| 110 | `gqh2_c` | 2.0625 | 66 B / 256 weights | no (fp16 `d` in-block) |
| 111 | `gqh4` | 4.28125 | 137 B / 256 weights | yes |

111 is the widest rung, added for the 32 GB / full-context band. It is `gqh3` one
bit up and the simplest decode in the family: E4M3 `d` (1 B), 16 uint4 sub-block
ratios (8 B), then 256 uint4 codes packed two per byte (128 B, even weight in the
low nibble) into a 16-level grid `±(j/8)^gamma`, `j = 1..8`. No bit-planes.

107 is `GGML_TYPE_Q2_0_ROCMFP2` in the lucebox tree. It is left free here so both
trees keep one wire numbering, the same reason 105/106 carry lucebox's numbers.

## The per-tensor header

`gqh4`, `gqh3` and `gqh2_h` scale every weight by a 5-byte header -- `float32
tensor_scale` then `uint8 grid_code` -- that the standalone wire puts in front of
the superblock stream. A ggml block is fixed-size, so a 5-byte prefix cannot live
in the tensor data. It travels in GGUF KV instead.

**The GGUF tensor data is the standalone wire with the first 5 bytes removed.**
Nothing else changes, so `ggml_nbytes` is exactly `rows * (cols/256) * type_size`.

### `geoquant.gqh.headers` KV (schema v1, frozen)

A `u8` array. Parsed by `src/llama-gqh.cpp`, written by
`scripts/gqh/make_gqh_probe.py`:

```
header : magic "GQHh1\0\0\0" (8) | entry_count u32 | reserved u32 (=0)
entry  : name_len u32 | name utf-8 | qtype u32 (108|109|111)
         | tensor_scale f32 LE | grid_code u8 | pad[3] (=0)
```

`grid_code` indexes `GAMMA_GRID` for 108 and 111 and `A_GRID` for 109, all length 12.

The cover is checked in ONE direction only, and the asymmetry is deliberate:

- Every resident 108/109/111 tensor MUST have an entry. Without one it aborts at
  decode, so the load fails and names the tensor.
- An entry with no resident tensor is EXPECTED and only logged. MTP-block tensors
  are created with `TENSOR_SKIP` unless the context is an MTP one (`load_mtp` is
  false in `llama_model_load`, true only in `llama-quantize`), so a correct
  artifact carries entries a normal load will never match. Refusing them is the
  C2/C6 trap in the GQH handoff -- a prior campaign lost a build to a sidecar
  entry naming a non-materialized block. This code made the same mistake and was
  fixed after reproducing it with `--phantom-headers`.

Nothing is lost by allowing extra entries: a MIS-NAMED entry still fails, because
the tensor it should have named is then uncovered and the first rule fires.

## What the exporter must satisfy

The loader refuses, at load, anything it cannot serve correctly:

- `ne[0] % 256 == 0`. Short rows (Qwen3.8 `ssm_*`) ride a stock qtype.
- 2-D only. `tensor_scale` is fitted per matrix and this KV is keyed by tensor
  name, so a fused 3-D expert stack has no way to carry one scale per expert.
- Contiguous, and not in a host buffer. GQH decodes only on CUDA/HIP; a
  host-resident tensor is copied to the GPU under a different pointer and would
  miss the registry.
- `tensor_scale` finite and positive.

## Decode paths

`ggml/src/ggml-cuda/gqh.cu` holds the CUDA kernels. The header registry is NOT
there -- it is in `ggml/src/gqh.cpp` (ggml-base), so `src/llama-gqh.cpp` calls
`ggml_gqh_register` directly instead of the `dlsym` dance `llama-rocmfpx-mix.cpp`
needs for its backend-local registry.

All four rungs dequantize to f16/f32, so prefill takes the dequant -> cuBLAS
path. All four additionally have a FUSED batch-1 matvec
(`ggml_cuda_gqh_mul_mat_vec`), hooked in `ggml_cuda_mul_mat` for
`src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`, which decodes inline instead of
materialising the whole weight matrix per token.

It is a dedicated kernel, not an MMVQ vec-dot, following the ROCmFPX mix
precedent. `vec_dot_q_cuda_t` carries no per-tensor argument and GQH needs the
grid plus `tensor_scale`, so an MMVQ integration would mean changing a signature
every qtype shares. f32 activations also avoid q8_1's 32-weight block straddling
GQH's 16-weight sub-blocks, which carry different uint4 ratios.

GQH therefore still has no `mul_mat_vec_q` kernel, and
`ggml_cuda_qtype_has_no_mmvq()` in `mmvq.cu` must keep every path away from it.
Every route to that kernel must consult that one predicate; a fusion gate with
its own copy of the list is what made Qwen3.5/3.8 abort after a few tokens (fork
commit 619a576e).

### HIP / ROCm

Built and validated on ROCm 7.1 (Debian 13), gfx1100 (RX 7900 XTX) and gfx1201
(Radeon AI PRO R9700), both wave32:

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build-hip -G Ninja \
    -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1100;gfx1201" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
```

All 18 `test-gqh` cases pass on both. Perplexity agrees with CUDA to within 0.02%
(H100 12.8496, gfx1100 12.8595, gfx1201 12.8354 on the Qwen2.5-1.5B gqh3
artifact). The fused matvec matters MORE here than on NVIDIA, because the
dequant -> hipBLAS fallback is relatively worse:

| device | fused tg64 | dequant tg64 | speedup |
|---|---|---|---|
| gfx1100 RX 7900 XTX | 228.45 | 59.94 | 3.81x |
| gfx1201 R9700 | 184.61 | 56.61 | 3.26x |
| H100 (reference) | 318.90 | 118.19 | 2.70x |

The kernel reaches a much higher fraction of peak bandwidth on AMD (~23% on
gfx1100, ~28% on gfx1201) than on H100 (~9%), which is consistent with the H100
result being latency-bound rather than bandwidth-bound.

`GGML_GQH_FUSED=0` forces the dequant path, for A/B measurement.

What the static audit against `rocmfp3_mix.cu` turned up beforehand, all of which
held up in the real build:

- `__shfl_down_sync` does NOT exist in HIP's vendor shim. The reduction goes
  through `gqh_warp_shfl_down`, which branches on `__HIP_PLATFORM_AMD__` and
  passes an explicit 32-lane width so a logical group stays self-contained on
  wave64 (GFX8/9). gfx1151 and gfx1201 are wave32, where the width is a no-op.
- Warp identity is `threadIdx.x / 32`, i.e. logical 32-lane groups. That is only
  safe because every shuffle carries the width; do not drop it.
- A misaligned 128-bit load FAULTS on AMD instead of running slow, so the
  launcher checks the activation base and column stride are 16-byte aligned and
  returns false (keeping the dequant fallback) rather than trusting the layout.
- Sub-word reads use `memcpy` rather than casting a `uint16_t *` at a 2-byte
  offset, which keeps them legal on both toolchains.
- `__constant__` on AMD is read-only global rather than a separate bank, so the
  divergent-index cost that drove the register-select grid rewrite may profile
  differently there. The CUDA tuning did carry (the fused path wins on both), but
  the ratio differs, so re-measure per target rather than assuming.

### Fused matvec layout and tuning

One warp per output row, 8 weights per lane, so a lane reads 2 adjacent bytes of
the low-2-bit plane and 1 byte of the high-1-bit plane and the warp covers
[9,73) and [73,105) contiguously. Two things mattered, both found in the SASS:

- The grid must NOT be indexed as a table. As a kernel-arg array it compiled to
  divergent constant-bank loads (78 LDC against 23 FP ops), which serialise per
  unique address -- 8 replays per warp. Both grids are symmetric about zero, so
  `gqh_level()` selects a magnitude from registers and applies a sign instead.
  LDC 78 -> 15, tg 197 -> 296 t/s.
- `ratio/15` is indexed by the lane's sub-block, so it is divergent too. In LDS
  its 16 consecutive floats land in 16 distinct banks, conflict-free.

`gqh4` takes the LDS route rather than the register select. The register trick
scales with the magnitude count: gqh3's 4 magnitudes cost 3 selects, gqh4's 8 cost
7. Its 16 signed levels are staged into LDS alongside `s_ratio` instead -- same
argument as the ratio table, 16 consecutive floats over 16 distinct banks with one
address each, so the divergent read broadcasts conflict-free. gqh3 and gqh2_h keep
the register path untouched, so their codegen and their bit-exactness are
unchanged by 111 landing. A lane's 8 gqh4 codes are one unaligned 32-bit read at
`9 + lane*4` (`memcpy`, like gqh3's uint16, so the compiler may pick byte loads
rather than emit an access that faults on AMD), and code `t` is `(u >> 4*t) & 0xf`
in either nibble order.

Measured on Qwen2.5-1.5B (all 196 weight matrices at gqh3), H100, tg64:
dequant -> cuBLAS 118.2 t/s, fused 318.9 t/s (2.70x), F16 424.1 t/s. That is
307 GB/s against roughly 3.35 TB/s of HBM, so the kernel is latency-bound, not
bandwidth-bound -- warps-per-block barely moves it (2/4/8/16 -> 316/319/309/300),
so the limit is per-warp work: only `in/256` superblocks per row and a serial
accumulate chain. This is the starting point for the geo-evo pass, whose fitness
is throughput at fixed bit-exact correctness.

`ggml/src/gqh.cpp` is the CPU twin: the same registry plus scalar decoders behind
the type traits' `to_float`. It lives in ggml-base rather than a backend because
one registration at load has to serve both the CPU hooks and the CUDA converters.

The CPU backend's `supports_op` still refuses GQH, and that is NOT about
`to_float`. CPU `MUL_MAT` dispatches on `type_traits_cpu[type].vec_dot`, which
GQH does not have -- the entry is zero-initialised, so a node reaching the CPU
backend would call through a null pointer. Partial offload therefore needs a
`vec_dot`, not a `to_float`.

`from_float_ref` still aborts: minting a GQH tensor needs the encoder's
per-tensor grid search and the header KV it emits.

### What the header registry cannot reach

It is keyed by data pointer, so it only resolves callers that pass the real,
registered tensor data. Two callers stage a copy first and are out of reach:
`llama-model.cpp` dequant-for-introspection (`ggml_backend_tensor_get` into a
local buffer) and `llama-quantize`, which repoints `tensor->data` at a rotating
scratch buffer (`llama-quant.cpp:1146`) -- the same address serves every tensor
in turn, so pointer keying cannot work there even in principle. Reading a GQH
model as a requantization SOURCE needs the header keyed by tensor, plus a
`ggml_validate_row_data` case; today it stops at the validator with `invalid type
108`, which is loud and safe but not useful.

## Constants

`ggml/src/gqh-tables.h` is generated by
`scripts/gqh/gen_gqh_tables.py`, which imports the geo-quant reference and emits
raw float32 bit patterns, plus the layout constants read from the module itself.
Recomputing e.g. `powf(k/4, gamma)` on device drifts by an ULP and breaks
bit-exact parity, so nothing in either decoder derives a constant.

Regenerate after any change to `GAMMA_GRID`, `A_GRID`, the codebook or the sign
tables:

```
<geo-quant .venv python> scripts/gqh/gen_gqh_tables.py <path/to/geo-quant>
```

## Tests

```
ctest -R test-gqh                    # 18 cases: 3 rungs x 3 shapes x {CPU, CUDA}
```

`test-gqh-cpu-decode` drives `ggml_get_type_traits(type)->to_float` directly and
needs no GPU; `test-gqh-backend` stages the tensor on the GPU backend and runs
`MUL_MAT` twice -- against a full identity (dequant -> BLAS) and against 8 basis
vectors (fused matvec). Both compare BITWISE against the geo-quant reference
decode in `tests/gqh-vectors/`.

One documented exception, on the dequant path only: where the f32 reference sits
EXACTLY halfway between two fp16 values, which neighbour comes back is
backend-defined. NVIDIA and the host both pick ties-to-even; the ROCm path returns
the other one. `fp16_tie_equivalent()` accepts that single case and nothing else --
a 1-ULP difference anywhere but an exact tie still fails. gqh2_c is the only rung
that trips it, because its codebook is k/64 (dyadic) so products land on ties
constantly, while gqh3's (k/4)^gamma grid essentially never does. The f32 decode
itself is bit-exact on both platforms, which the fused comparison proves
independently.

Loader cover rules, which need a built artifact (no GPU-free unit test covers them):

```
# extra entries naming non-resident tensors -> must LOAD, logging the count
make_gqh_probe.py in.gguf mtp.gguf --repo <geo-quant> --layers 2 --phantom-headers 2
# a converted tensor whose entry is missing -> must REFUSE, naming the tensor
make_gqh_probe.py in.gguf bad.gguf --repo <geo-quant> --layers 2 --drop-headers 1
```

Probe artifact:

```
<geo-quant .venv python> scripts/gqh/make_gqh_probe.py in.gguf out.gguf --repo <geo-quant> --layers 4
<geo-quant .venv python> scripts/gqh/make_gqh_probe.py in.gguf ctl.gguf --repo <geo-quant> --layers 4 --fakequant
```

`--fakequant` stores `decode(encode(w))` as plain F16: same numerics, stock
qtype. Scoring the two side by side separates real quantization damage from a
decode bug. They must agree to every printed digit.

Serving checks on the Qwen2.5-1.5B gqh3 artifact: `llama-server` with `-np 4`
answers a single completion and 4 parallel slots coherently; CUDA graphs capture
and are reused with the fused kernel (32 reused over 32 decode steps); and
perplexity is flat across the fused/dequant routing boundary at
`-ub 1/8/9/64/512` -> 13.1980 / 13.1984 / 13.1983 / 13.2054 / 13.2094, with no
discontinuity at the `MMVQ_MAX_BATCH_SIZE` cap of 8.

Measured on Qwen2.5-1.5B, wikitext, 20 x 512: F16 10.3468, gqh3 12.8496,
fakequant twin 12.8496 (identical, so the decode is right in situ), and gqh3
forced through the fused matvec with `-b 1 -ub 1` 12.8351 -- marginally better
because the fused path accumulates f32-decoded weights in f32 rather than going
through an fp16 dequant. Note this artifact is a plain fit with no GPTQ and every
matrix at one rung, so its quality delta is not the rung's measured potential.
