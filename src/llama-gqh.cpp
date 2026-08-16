#include "llama-gqh.h"

#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// GQH header KV wire (schema v1, FROZEN -- the exporter side is geo-quant
// branch feat/custom-format-family):
//   KV key : "geoquant.gqh.headers" (u8 array)
//   header : magic "GQHh1\0\0\0" (8) | entry_count u32 | reserved u32 (=0)
//   entry  : name_len u32 | name utf-8 | qtype u32 (108|109)
//            | tensor_scale f32 LE | grid_code u8 | pad[3] (=0)
// These are the same 5 bytes that prefix the tensor payload in the standalone
// wire (geoquant.formats.gqh encode3/encode); the GGUF tensor data is that wire
// with the 5-byte prefix stripped, so it stays a whole number of ggml blocks.
// ---------------------------------------------------------------------------

static const char *  GQH_KV_KEY   = "geoquant.gqh.headers";
static const uint8_t GQH_MAGIC[8] = { 'G', 'Q', 'H', 'h', '1', 0, 0, 0 };

// GAMMA_GRID / A_GRID length in geoquant.formats.gqh.
static const uint32_t GQH_GRID_CODES = 12;

// Every rung needs whole 256-weight superblocks along the row axis (spec C4).
static const int64_t GQH_SUPERBLOCK = 256;

struct gqh_entry {
    int32_t qtype        = 0;
    float   tensor_scale = 0.0f;
    uint8_t grid_code    = 0;
};

static bool gqh_qtype_has_header(int32_t qtype) {
    return qtype == GGML_TYPE_GQH3 || qtype == GGML_TYPE_GQH2_H;
}

static std::map<std::string, gqh_entry> gqh_parse(const uint8_t * blob, size_t n) {
    auto fail = [](const std::string & msg) -> std::map<std::string, gqh_entry> {
        throw std::runtime_error("gqh: invalid " + std::string(GQH_KV_KEY) + ": " + msg);
    };
    if (n < 16) {
        return fail("blob truncated (" + std::to_string(n) + " B < 16 B header)");
    }
    if (memcmp(blob, GQH_MAGIC, 8) != 0) {
        return fail("bad magic (expected GQHh1)");
    }
    uint32_t count = 0, reserved = 0;
    memcpy(&count,    blob +  8, 4);
    memcpy(&reserved, blob + 12, 4);
    if (reserved != 0) {
        return fail("reserved field " + std::to_string(reserved) + " != 0 (format drift?)");
    }
    std::map<std::string, gqh_entry> entries;
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
        if (off + 12 > n) return fail("'" + name + "': truncated metadata");
        uint32_t qtype = 0;
        float    scale = 0.0f;
        memcpy(&qtype, blob + off + 0, 4);
        memcpy(&scale, blob + off + 4, 4);
        const uint8_t grid_code = blob[off + 8];
        if (blob[off + 9] || blob[off + 10] || blob[off + 11]) {
            return fail("'" + name + "': padding is not zero (format drift?)");
        }
        off += 12;
        if (!gqh_qtype_has_header((int32_t) qtype)) {
            return fail("'" + name + "': qtype " + std::to_string(qtype) + " is not 108/109");
        }
        if (grid_code >= GQH_GRID_CODES) {
            return fail("'" + name + "': grid code " + std::to_string(grid_code)
                        + " >= " + std::to_string(GQH_GRID_CODES));
        }
        // encode3/encode set tensor_scale to max|w| and fall back to 1.0 for an
        // all-zero tensor, so a non-finite or non-positive scale means the KV is
        // wrong, not that the tensor is unusual.
        if (!std::isfinite(scale) || scale <= 0.0f) {
            return fail("'" + name + "': tensor_scale " + std::to_string(scale)
                        + " is not finite and positive");
        }
        gqh_entry e;
        e.qtype        = (int32_t) qtype;
        e.tensor_scale = scale;
        e.grid_code    = grid_code;
        entries.emplace(std::move(name), e);
    }
    if (off != n) {
        return fail(std::to_string(n - off) + " trailing bytes after "
                    + std::to_string(count) + " entries");
    }
    return entries;
}

void llama_gqh_register_tensors(
        const gguf_context * meta,
        const std::vector<ggml_tensor *> & gqh_tensors,
        llama_gqh_registrations & regs) {
    if (gqh_tensors.empty()) {
        return;
    }

    const int64_t kid = gguf_find_key(meta, GQH_KV_KEY);
    if (kid < 0) {
        throw std::runtime_error(std::string("gqh: model carries qtype-108/109 tensors but no '")
                                 + GQH_KV_KEY + "' KV -- their per-tensor scale and grid code are "
                                 "out-of-band and the tensors cannot be decoded without it");
    }
    if (gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(meta, kid) != GGUF_TYPE_UINT8) {
        throw std::runtime_error(std::string("gqh: '") + GQH_KV_KEY + "' KV is not a u8 array");
    }
    auto entries = gqh_parse((const uint8_t *) gguf_get_arr_data(meta, kid),
                             gguf_get_arr_n(meta, kid));

    // Validate the EXACT cover first -- register nothing on a bad KV.
    std::set<std::string> matched;
    for (const ggml_tensor * t : gqh_tensors) {
        const std::string name = t->name;
        const auto it = entries.find(name);
        if (it == entries.end()) {
            throw std::runtime_error("gqh: resident tensor '" + name
                                     + "' (type " + ggml_type_name(t->type) + ") has no '"
                                     + GQH_KV_KEY + "' entry -- refusing to load a tensor that "
                                     "would abort at decode time");
        }
        if (it->second.qtype != (int32_t) t->type) {
            throw std::runtime_error("gqh: header entry for '" + name + "' says qtype "
                                     + std::to_string(it->second.qtype) + " but the tensor is "
                                     + std::to_string((int32_t) t->type));
        }
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            // tensor_scale is fitted per 2-D matrix by the encoder, and this KV is
            // keyed by tensor name, so a fused 3-D expert stack has no way to carry
            // one scale per expert. Refuse rather than apply expert 0's scale to all.
            throw std::runtime_error("gqh: '" + name + "' is not 2-D -- the header KV carries one "
                                     "scale per tensor, so fused expert stacks are not supported yet");
        }
        if (t->ne[0] % GQH_SUPERBLOCK != 0) {
            throw std::runtime_error("gqh: '" + name + "' row length "
                                     + std::to_string((long long) t->ne[0]) + " is not a multiple of "
                                     + std::to_string((long long) GQH_SUPERBLOCK)
                                     + " -- the exporter must leave short rows on a stock qtype");
        }
        if (!ggml_is_contiguous(t)) {
            throw std::runtime_error("gqh: '" + name + "' is not contiguous");
        }
        if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
            // A host-resident GQH tensor would be copied to the GPU under a
            // DIFFERENT pointer at eval time, missing the registry and hitting the
            // decode-time abort. Refuse at load, where the fix is clear.
            throw std::runtime_error("gqh: '" + name + "' resides in host buffer '"
                                     + ggml_backend_buffer_name(t->buffer) + "' -- GQH decodes only "
                                     "on the CUDA backend; offload every layer that carries a "
                                     "qtype-108/109 tensor (increase -ngl)");
        }
        matched.insert(name);
    }
    // Entries with no resident tensor are EXPECTED, not an error. MTP-block tensors
    // are created with TENSOR_SKIP unless the context is an MTP one (see load_mtp in
    // llama-model-loader), so a correct artifact carries entries this load will never
    // match. Refusing them is the C2/C6 trap the GQH handoff warns about -- a prior
    // campaign lost a build to a sidecar entry naming a non-materialized block.
    //
    // Nothing is lost by allowing them: an unmatched entry is never looked up, and a
    // MIS-NAMED entry still fails, because the tensor it should have named is then
    // uncovered and the loop above throws. Only the count is worth reporting.
    if (matched.size() != entries.size()) {
        LLAMA_LOG_INFO("%s: %zu header entr%s name no resident tensor (expected for "
                       "MTP-block weights outside an MTP context)\n",
                       __func__, entries.size() - matched.size(),
                       entries.size() - matched.size() == 1 ? "y" : "ies");
    }

    for (const ggml_tensor * t : gqh_tensors) {
        const gqh_entry & e = entries.at(t->name);
        ggml_gqh_register(t->data, ggml_nbytes(t), e.tensor_scale, (int) e.grid_code);
        regs.bases.push_back(t->data);
        LLAMA_LOG_INFO("%s: registered %s (%s, scale %.9g, grid %d, %lld x %lld)\n",
                       __func__, t->name, ggml_type_name(t->type), e.tensor_scale,
                       (int) e.grid_code, (long long) t->ne[1], (long long) t->ne[0]);
    }
    LLAMA_LOG_INFO("%s: %zu GQH tensor(s) registered from '%s'\n",
                   __func__, regs.bases.size(), GQH_KV_KEY);
}

void llama_gqh_unregister_all(llama_gqh_registrations & regs) {
    if (regs.bases.empty()) {
        return;
    }
    for (const void * base : regs.bases) {
        ggml_gqh_unregister(base);
    }
    regs.bases.clear();
}
