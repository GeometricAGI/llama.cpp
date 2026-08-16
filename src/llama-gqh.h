#pragma once

// Loader-side support for the GQH qtypes (GGML_TYPE_GQH3 = 108,
// GGML_TYPE_GQH2_H = 109). Both rungs scale every weight by a 5-byte per-tensor
// header (float32 tensor_scale + uint8 grid code) that cannot live in a
// fixed-size ggml block, so it rides in the "geoquant.gqh.headers" GGUF KV.
// Decoding an UNREGISTERED GQH tensor aborts, so the KV is read, validated and
// registered with the CUDA decode registry as part of the LOAD -- any violation
// fails the load loudly instead of surfacing as a decode-time abort.
//
// GGML_TYPE_GQH2_C = 110 needs no entry: its scale is fp16 in-block.

#include <cstdint>
#include <vector>

struct gguf_context;
struct ggml_tensor;

// The tensor data pointers a model registered, so it can unregister exactly
// those when it is freed.
struct llama_gqh_registrations {
    std::vector<const void *> bases;
};

// Read + validate the header KV against the resident GQH tensors (exact 1:1
// cover with matching qtype) and register each tensor's header with the CUDA
// decode registry. Throws std::runtime_error on any violation.
void llama_gqh_register_tensors(
        const gguf_context * meta,
        const std::vector<ggml_tensor *> & gqh_tensors,
        llama_gqh_registrations & regs);

void llama_gqh_unregister_all(llama_gqh_registrations & regs);
