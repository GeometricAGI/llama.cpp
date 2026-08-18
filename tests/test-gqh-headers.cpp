// The "geoquant.gqh.headers" KV parser, tested without a model.
//
// gqh4/gqh3/gqh2_h carry a 5-byte per-tensor header out of band, and
// llama_gqh_register_tensors is the ONLY place that decides which qtypes require
// one. Until this test existed that decision was reachable only through a real
// model load, so adding a rung meant either trusting the one-line change or
// building a multi-GB artifact to exercise it.
//
// Runs on the CPU: the registry lives in ggml-base (ggml/src/gqh.cpp), not in a
// backend, so nothing here needs a GPU.
//
// Covers the parse, the header values that come back out of the registry, and the
// three rules that are easy to get backwards:
//   - gqh2_c (110) needs NO entry -- its scale is fp16 in-block
//   - a resident header-bearing tensor with no entry MUST fail the load
//   - an entry with no resident tensor MUST be accepted (MTP TENSOR_SKIP tensors
//     legitimately produce these; refusing them cost a prior campaign a build)

#include "llama-gqh.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Declared in ggml/src/gqh.h, which is internal to ggml and not on the public
// include path. It is GGML_API, so declaring it here is enough to link.
extern "C" bool ggml_gqh_lookup(const void * p, float * tensor_scale, int * grid_code);

static const char * GQH_KV_KEY = "geoquant.gqh.headers";

static int failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

// A negative case must be refused for ITS OWN reason, not incidentally by some
// earlier validation, so print the message every refusal came back with.
static void check_refused(const std::string & err, const char * what) {
    check(!err.empty(), what);
    if (!err.empty()) {
        printf("  refused (%s): %s\n", what, err.c_str());
    }
}

struct entry {
    std::string name;
    uint32_t    qtype;
    float       scale;
    uint8_t     code;
};

// Mirrors scripts/gqh/make_gqh_probe.py: build_header_kv.
static std::vector<uint8_t> build_kv(const std::vector<entry> & entries, uint32_t count_override,
                                     const char * magic = "GQHh1\0\0\0") {
    std::vector<uint8_t> b;
    auto put = [&](const void * p, size_t n) {
        const uint8_t * q = (const uint8_t *) p;
        b.insert(b.end(), q, q + n);
    };
    put(magic, 8);
    const uint32_t count = count_override ? count_override : (uint32_t) entries.size();
    const uint32_t reserved = 0;
    put(&count, 4);
    put(&reserved, 4);
    for (const entry & e : entries) {
        const uint32_t len = (uint32_t) e.name.size();
        put(&len, 4);
        put(e.name.data(), len);
        put(&e.qtype, 4);
        put(&e.scale, 4);
        put(&e.code, 1);
        const uint8_t pad[3] = { 0, 0, 0 };
        put(pad, 3);
    }
    return b;
}

static gguf_context * make_meta(const std::vector<uint8_t> & blob) {
    gguf_context * g = gguf_init_empty();
    gguf_set_arr_data(g, GQH_KV_KEY, GGUF_TYPE_UINT8, blob.data(), (int64_t) blob.size());
    return g;
}

// A context of 2-D GQH tensors with real (allocated) data, so the registry has
// pointer ranges to resolve -- the row-slice lookup the decoders use.
struct model {
    ggml_context *              ctx = nullptr;
    std::vector<ggml_tensor *>  tensors;

    model(const std::vector<std::pair<const char *, ggml_type>> & spec) {
        ggml_init_params ip = {};
        ip.mem_size = 64u * 1024 * 1024;
        ctx = ggml_init(ip);
        for (const auto & [name, type] : spec) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx, type, 256, 4);
            ggml_set_name(t, name);
            tensors.push_back(t);
        }
    }
    ~model() { if (ctx) { ggml_free(ctx); } }
};

// What llama-model.cpp collects: the header-bearing rungs only. The callee then
// requires an entry for every tensor it is GIVEN, so a gqh2_c tensor reaching it
// is a caller bug, not a malformed artifact -- keep the two roles separate here.
static std::vector<ggml_tensor *> header_bearing(const model & m) {
    std::vector<ggml_tensor *> v;
    for (ggml_tensor * t : m.tensors) {
        if (t->type == GGML_TYPE_GQH3 || t->type == GGML_TYPE_GQH2_H ||
            t->type == GGML_TYPE_GQH4) {
            v.push_back(t);
        }
    }
    return v;
}

// Returns the error message, or "" on success.
static std::string try_register(gguf_context * meta, model & m, llama_gqh_registrations & regs) {
    try {
        llama_gqh_register_tensors(meta, header_bearing(m), regs);
    } catch (const std::exception & e) {
        return e.what();
    }
    return "";
}

int main() {
    // ---- 1. the happy path: one tensor per header-bearing rung ------------
    {
        model m({ { "blk.0.ffn_down.weight", GGML_TYPE_GQH3   },
                  { "blk.0.ffn_up.weight",   GGML_TYPE_GQH2_H },
                  { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4   },
                  { "blk.0.attn_q.weight",   GGML_TYPE_GQH2_C } });
        auto blob = build_kv({ { "blk.0.ffn_down.weight", 108, 0.125f,  3 },
                               { "blk.0.ffn_up.weight",   109, 2.5f,    7 },
                               { "blk.0.ffn_gate.weight", 111, 8.0f,   11 } }, 0);
        gguf_context * meta = make_meta(blob);
        llama_gqh_registrations regs;
        const std::string err = try_register(meta, m, regs);
        check(err.empty(), ("happy path registers: " + err).c_str());

        // gqh2_c must NOT be registered -- it needs nothing out of band.
        struct expect { int idx; float scale; int code; bool registered; };
        const expect want[] = { { 0, 0.125f,  3, true  },
                                { 1, 2.5f,    7, true  },
                                { 2, 8.0f,   11, true  },
                                { 3, 0.0f,    0, false } };
        for (const expect & w : want) {
            float scale = -1.0f;
            int   code  = -1;
            const bool got = ggml_gqh_lookup(m.tensors[w.idx]->data, &scale, &code);
            check(got == w.registered,
                  (std::string("registration of ") + ggml_get_name(m.tensors[w.idx])).c_str());
            if (w.registered) {
                check(scale == w.scale && code == w.code,
                      (std::string("header values of ") + ggml_get_name(m.tensors[w.idx])).c_str());
                // The decoders look up by ROW SLICE, not just the base pointer.
                const size_t row = ggml_row_size(m.tensors[w.idx]->type, 256);
                float s2 = -1.0f; int c2 = -1;
                check(ggml_gqh_lookup((const char *) m.tensors[w.idx]->data + 2 * row, &s2, &c2)
                          && s2 == w.scale && c2 == w.code,
                      (std::string("row-slice lookup of ") + ggml_get_name(m.tensors[w.idx])).c_str());
            }
        }
        llama_gqh_unregister_all(regs);
        float s_gone = -1.0f; int c_gone = -1;
        check(!ggml_gqh_lookup(m.tensors[0]->data, &s_gone, &c_gone), "unregister_all clears");
        gguf_free(meta);
    }

    // ---- 2. a resident gqh4 tensor with no entry must FAIL ---------------
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        gguf_context * meta = make_meta(build_kv({}, 0));
        llama_gqh_registrations regs;
        check_refused(try_register(meta, m, regs), "uncovered gqh4 tensor is refused");
        llama_gqh_unregister_all(regs);
        gguf_free(meta);
    }

    // ---- 3. an entry naming no resident tensor must be ACCEPTED ----------
    // MTP-block tensors are created with TENSOR_SKIP on a normal load, so a
    // correct artifact carries entries that will never match. The cover check is
    // deliberately one-directional.
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        auto blob = build_kv({ { "blk.0.ffn_gate.weight",   111, 8.0f, 11 },
                              { "blk.64.nextn.ffn_down.weight", 111, 1.0f, 0 } }, 0);
        gguf_context * meta = make_meta(blob);
        llama_gqh_registrations regs;
        const std::string err = try_register(meta, m, regs);
        check(err.empty(), ("phantom MTP entry is accepted: " + err).c_str());
        llama_gqh_unregister_all(regs);
        gguf_free(meta);
    }

    // ---- 4. an entry whose qtype does not carry a header must FAIL -------
    // 110 is gqh2_c: fp16 d in-block, so an entry for it is a malformed export.
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4   },
                  { "blk.0.attn_q.weight",   GGML_TYPE_GQH2_C } });
        gguf_context * meta = make_meta(build_kv({ { "blk.0.ffn_gate.weight", 111, 8.0f, 11 },
                                                   { "blk.0.attn_q.weight",   110, 1.0f,  0 } }, 0));
        llama_gqh_registrations regs;
        check_refused(try_register(meta, m, regs), "entry for a headerless qtype is refused");
        llama_gqh_unregister_all(regs);
        gguf_free(meta);
    }

    // ---- 5. qtype mismatch between entry and resident tensor must FAIL ---
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        gguf_context * meta = make_meta(build_kv({ { "blk.0.ffn_gate.weight", 108, 8.0f, 11 } }, 0));
        llama_gqh_registrations regs;
        check_refused(try_register(meta, m, regs), "qtype mismatch is refused");
        llama_gqh_unregister_all(regs);
        gguf_free(meta);
    }

    // ---- 6. structural corruption must FAIL, not read past the blob ------
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        const std::vector<entry> one = { { "blk.0.ffn_gate.weight", 111, 8.0f, 11 } };

        struct bad { std::vector<uint8_t> blob; const char * what; };
        std::vector<bad> cases;
        cases.push_back({ build_kv(one, 0, "GQHh2\0\0\0"), "wrong magic" });
        cases.push_back({ build_kv(one, 9),                "entry_count over-declared" });
        {   // truncated mid-entry
            auto b = build_kv(one, 0);
            b.resize(b.size() - 5);
            cases.push_back({ b, "truncated entry" });
        }
        {   // empty blob
            cases.push_back({ std::vector<uint8_t>{}, "empty blob" });
        }
        for (bad & c : cases) {
            gguf_context * meta = make_meta(c.blob);
            llama_gqh_registrations regs;
            check_refused(try_register(meta, m, regs), c.what);
            llama_gqh_unregister_all(regs);
            gguf_free(meta);
        }
    }

    // ---- 7. a grid code outside GAMMA_GRID must FAIL ---------------------
    // 12 codes, so 12 is out of range; the decoders index the table unchecked.
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        gguf_context * meta = make_meta(build_kv({ { "blk.0.ffn_gate.weight", 111, 1.0f, 12 } }, 0));
        llama_gqh_registrations regs;
        check_refused(try_register(meta, m, regs), "out-of-range grid code is refused");
        llama_gqh_unregister_all(regs);
        gguf_free(meta);
    }

    // ---- 8. the KV missing entirely, with GQH tensors resident, must FAIL -
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        gguf_context * meta = gguf_init_empty();
        llama_gqh_registrations regs;
        check_refused(try_register(meta, m, regs), "missing KV with resident GQH is refused");
        llama_gqh_unregister_all(regs);
        gguf_free(meta);
    }

    if (failures) {
        fprintf(stderr, "\ntest-gqh-headers: %d check(s) failed\n", failures);
        return 1;
    }
    printf("OK   test-gqh-headers: KV parse, header values, cover rules and 6 malformed inputs\n");
    return 0;
}
