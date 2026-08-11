#include "llama-rocmfpx-mix.h"

#include "llama-impl.h"

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

// ---------------------------------------------------------------------------
// dmix2 sidecar wire (schema v1, FROZEN — mirror of geoquant
// formats/export_rocmfpx_gguf.py serialize_dmix2_sidecar):
//   KV key : "geoquant.dmix2.sidecar" (u8 array)
//   header : magic "DMX2s1\0\0" (8) | entry_count u32 | reserved u32 (=0)
//   entry  : name_len u32 | name utf-8 | qtype u32 (105|106) | C u32 (=2)
//            | K u32 (8 for 105, 4 for 106) | mode u8 (0 fixed, 1 adaptive
//            7s1c) | pad[3] | codebook (C*K) x bf16-as-u16 LE, row-major
// ---------------------------------------------------------------------------

static const char *  DMIX2_KV_KEY = "geoquant.dmix2.sidecar";
static const uint8_t DMIX2_MAGIC[8] = { 'D', 'M', 'X', '2', 's', '1', 0, 0 };

// The CUDA mix kernels' wide block load steps 128 weights per row; rows off
// that grid read past the tensor on the final block. Same constant as the
// geoquant plan-time filter (DMIX_ROW_ALIGN) and the DS4 loader.
static const int64_t DMIX2_ROW_ALIGN = 128;

struct dmix2_entry {
    int32_t qtype    = 0;
    uint8_t mode     = 0;
    std::vector<uint8_t> codebook; // C*K bf16, raw little-endian bytes
};

// register_host(base, nb02, n_experts, out, in, codebooks_bf16_host,
//               modes_u8_host, rotations_u8_host_or_null)
typedef void (*mix_register_host_fn)(const void *, size_t, int, int, int,
                                     const void *, const uint8_t *, const uint8_t *);
typedef void (*mix_unregister_fn)(const void *);

static void * mix_dlsym(const char * name) {
#if defined(_WIN32)
    return (void *) GetProcAddress(GetModuleHandleA(nullptr), name);
#else
    return dlsym(RTLD_DEFAULT, name);
#endif
}

static int dmix2_levels_for_qtype(int32_t qtype) {
    switch (qtype) {
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 8;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 4;
        default:                         return 0;
    }
}

static std::map<std::string, dmix2_entry> dmix2_parse(const uint8_t * blob, size_t n) {
    auto fail = [](const std::string & msg) -> std::map<std::string, dmix2_entry> {
        throw std::runtime_error("rocmfpx-mix: invalid " + std::string(DMIX2_KV_KEY) + ": " + msg);
    };
    if (n < 16) {
        return fail("blob truncated (" + std::to_string(n) + " B < 16 B header)");
    }
    if (memcmp(blob, DMIX2_MAGIC, 8) != 0) {
        return fail("bad magic (expected DMX2s1)");
    }
    uint32_t count = 0, reserved = 0;
    memcpy(&count,    blob +  8, 4);
    memcpy(&reserved, blob + 12, 4);
    if (reserved != 0) {
        return fail("reserved field " + std::to_string(reserved) + " != 0 (format drift?)");
    }
    std::map<std::string, dmix2_entry> entries;
    size_t off = 16;
    for (uint32_t i = 0; i < count; ++i) {
        const std::string at = "entry " + std::to_string(i);
        if (off + 4 > n) return fail(at + ": truncated before name_len");
        uint32_t name_len = 0;
        memcpy(&name_len, blob + off, 4);
        off += 4;
        if (name_len == 0 || name_len > 1024 || off + name_len > n) {
            return fail(at + ": bad name_len " + std::to_string(name_len));
        }
        std::string name((const char *) blob + off, name_len);
        off += name_len;
        if (entries.count(name)) return fail("duplicate entry for '" + name + "'");
        if (off + 16 > n) return fail("'" + name + "': truncated metadata");
        uint32_t qtype = 0, c = 0, k = 0;
        memcpy(&qtype, blob + off + 0, 4);
        memcpy(&c,     blob + off + 4, 4);
        memcpy(&k,     blob + off + 8, 4);
        const uint8_t mode = blob[off + 12];
        off += 16; // 12 + mode byte + 3 pad
        const int want_k = dmix2_levels_for_qtype((int32_t) qtype);
        if (want_k == 0) return fail("'" + name + "': qtype " + std::to_string(qtype) + " is not 105/106");
        if (c != 2)      return fail("'" + name + "': C " + std::to_string(c) + " != 2");
        if ((int) k != want_k) {
            return fail("'" + name + "': K " + std::to_string(k) + " != " + std::to_string(want_k)
                        + " for qtype " + std::to_string(qtype));
        }
        if (mode > 1) return fail("'" + name + "': mode " + std::to_string(mode) + " not in {0, 1}");
        const size_t cb_bytes = (size_t) c * k * 2;
        if (off + cb_bytes > n) return fail("'" + name + "': truncated codebook");
        dmix2_entry e;
        e.qtype = (int32_t) qtype;
        e.mode  = mode;
        e.codebook.assign(blob + off, blob + off + cb_bytes);
        off += cb_bytes;
        entries.emplace(std::move(name), std::move(e));
    }
    if (off != n) {
        return fail(std::to_string(n - off) + " trailing bytes after "
                    + std::to_string(count) + " entries");
    }
    return entries;
}

void llama_rocmfpx_mix_register_tensors(
        const gguf_context * meta,
        const std::vector<ggml_tensor *> & mix_tensors,
        llama_rocmfpx_mix_registrations & regs) {
    if (mix_tensors.empty()) {
        return;
    }

    const int64_t kid = gguf_find_key(meta, DMIX2_KV_KEY);
    if (kid < 0) {
        throw std::runtime_error(std::string("rocmfpx-mix: model carries qtype-105/106 tensors but no '")
                                 + DMIX2_KV_KEY + "' KV — their codebooks are out-of-band and the "
                                 "tensors cannot be decoded without it");
    }
    if (gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(meta, kid) != GGUF_TYPE_UINT8) {
        throw std::runtime_error(std::string("rocmfpx-mix: '") + DMIX2_KV_KEY + "' KV is not a u8 array");
    }
    auto entries = dmix2_parse((const uint8_t *) gguf_get_arr_data(meta, kid),
                               gguf_get_arr_n(meta, kid));

    // Validate the EXACT cover first — register nothing on a bad sidecar.
    std::set<std::string> matched;
    for (const ggml_tensor * t : mix_tensors) {
        const std::string name = t->name;
        const auto it = entries.find(name);
        if (it == entries.end()) {
            throw std::runtime_error("rocmfpx-mix: resident tensor '" + name
                                     + "' (type " + ggml_type_name(t->type) + ") has no '"
                                     + DMIX2_KV_KEY + "' entry — refusing to load a tensor that "
                                     "would abort at decode time");
        }
        if (it->second.qtype != (int32_t) t->type) {
            throw std::runtime_error("rocmfpx-mix: sidecar entry for '" + name + "' says qtype "
                                     + std::to_string(it->second.qtype) + " but the tensor is "
                                     + std::to_string((int32_t) t->type));
        }
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::runtime_error("rocmfpx-mix: '" + name + "' is not 2-D — the dense "
                                     "(n_experts = 1) registration cannot describe it");
        }
        if (t->ne[0] % DMIX2_ROW_ALIGN != 0) {
            throw std::runtime_error("rocmfpx-mix: '" + name + "' row length "
                                     + std::to_string((long long) t->ne[0]) + " is off the mix kernels' "
                                     + std::to_string((long long) DMIX2_ROW_ALIGN) + "-weight grid");
        }
        if (!ggml_is_contiguous(t)) {
            throw std::runtime_error("rocmfpx-mix: '" + name + "' is not contiguous");
        }
        matched.insert(name);
    }
    if (matched.size() != entries.size()) {
        for (const auto & kv : entries) {
            if (!matched.count(kv.first)) {
                throw std::runtime_error("rocmfpx-mix: sidecar names '" + kv.first
                                         + "', which is not a resident qtype-105/106 tensor — the "
                                         "sidecar and the tensor data disagree about this model");
            }
        }
    }

    // Resolve the CUDA registry entry points. They live in the CUDA backend
    // (ggml/src/ggml-cuda/rocmfp{3,2}_mix.cu); a build without it cannot
    // decode these tensors anywhere, so the load must fail here too.
    auto reg105   = (mix_register_host_fn) mix_dlsym("ggml_cuda_rocmfp3_mix_register_host");
    auto reg106   = (mix_register_host_fn) mix_dlsym("ggml_cuda_rocmfp2_mix_register_host");
    auto unreg105 = (mix_unregister_fn)    mix_dlsym("ggml_cuda_rocmfp3_mix_unregister");
    auto unreg106 = (mix_unregister_fn)    mix_dlsym("ggml_cuda_rocmfp2_mix_unregister");
    if (!reg105 || !reg106 || !unreg105 || !unreg106) {
        throw std::runtime_error("rocmfpx-mix: this build has no CUDA mix decode registry "
                                 "(ggml_cuda_rocmfp*_mix_register_host not found) — qtype-105/106 "
                                 "tensors cannot be served");
    }

    for (const ggml_tensor * t : mix_tensors) {
        const dmix2_entry & e = entries.at(t->name);
        const int in  = (int) t->ne[0];
        const int out = (int) t->ne[1];
        const uint8_t mode = e.mode;
        if (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            reg105(t->data, ggml_nbytes(t), /*n_experts =*/ 1, out, in,
                   e.codebook.data(), &mode, /*rotations =*/ nullptr);
        } else {
            reg106(t->data, ggml_nbytes(t), /*n_experts =*/ 1, out, in,
                   e.codebook.data(), &mode, /*rotations =*/ nullptr);
        }
        regs.bases.emplace_back(t->data, (int32_t) t->type);
        LLAMA_LOG_INFO("%s: registered %s (qtype %d, mode %d, %d x %d)\n",
                       __func__, t->name, (int) t->type, (int) mode, out, in);
    }
    LLAMA_LOG_INFO("%s: %zu qtype-105/106 tensor(s) registered from '%s'\n",
                   __func__, regs.bases.size(), DMIX2_KV_KEY);
}

void llama_rocmfpx_mix_unregister_all(llama_rocmfpx_mix_registrations & regs) {
    if (regs.bases.empty()) {
        return;
    }
    auto unreg105 = (mix_unregister_fn) mix_dlsym("ggml_cuda_rocmfp3_mix_unregister");
    auto unreg106 = (mix_unregister_fn) mix_dlsym("ggml_cuda_rocmfp2_mix_unregister");
    for (const auto & [base, qtype] : regs.bases) {
        if (qtype == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            if (unreg105) unreg105(base);
        } else {
            if (unreg106) unreg106(base);
        }
    }
    regs.bases.clear();
}
