#pragma once

// Loader-side support for the ROCmFPX mix qtypes (GGML_TYPE_Q3_1_ROCMFP3_MIX =
// 105, GGML_TYPE_Q2_1_ROCMFP2_MIX = 106). Their per-tensor codebook + mode live
// out-of-band in the model's "geoquant.dmix2.sidecar" GGUF KV; decoding an
// UNREGISTERED 105/106 tensor aborts, so the sidecar is read, validated and
// registered with the CUDA decode registries as part of the LOAD — any
// violation (absent KV, malformed blob, non-exact tensor cover) fails the load
// loudly instead of surfacing as a decode-time abort.

#include <cstdint>
#include <utility>
#include <vector>

struct gguf_context;
struct ggml_tensor;

// The (tensor data pointer, qtype) pairs a model registered, so it can
// unregister exactly those when it is freed.
struct llama_rocmfpx_mix_registrations {
    std::vector<std::pair<const void *, int32_t>> bases;
};

// Read + validate the dmix2 sidecar KV against the resident 105/106 tensors
// (exact 1:1 cover with matching qtype) and register each tensor's codebook
// and mode with the CUDA decode registry (dense: n_experts = 1).
// Throws std::runtime_error on any violation.
void llama_rocmfpx_mix_register_tensors(
        const gguf_context * meta,
        const std::vector<ggml_tensor *> & mix_tensors,
        llama_rocmfpx_mix_registrations & regs);

void llama_rocmfpx_mix_unregister_all(llama_rocmfpx_mix_registrations & regs);
