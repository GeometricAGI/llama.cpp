// Bit-identity harness for the ROCmFPX mix qtype (105/106) CUDA decode paths.
//
// Loads a geoquant-packed block stream + its dmix2 codebook from files, stages
// the tensor on the GPU backend, registers it with the CUDA mix registry
// (ggml_cuda_rocmfp*_mix_register_host, resolved via dlsym like the llama
// loader does), then checks TWO decode paths against the geoquant reference
// dequant, bitwise:
//
//   1. dequant->cuBLAS: W (qtype 105/106, rows x cols) x I_cols (f32 identity,
//      ncols = cols > MMVQ batch cap, so ggml_cuda_mul_mat takes the generic
//      chain, whose to_fp16 shim is the registry-aware mix dequant kernel).
//      dst[i, j] = fp16(decode(W[i, j])) * 1.0 summed over exact zeros
//      == fp16 of the reference f32 decode, compared BITWISE against the
//      reference dequant rounded to fp16 (ref_fp16.bin).
//
//   2. fused matvec: W x onehot8 (first 8 basis columns; ncols = 8 <= MMVQ
//      batch cap routes into ggml_cuda_rocmfp*_mix_mul_mat_vec). The fused
//      kernel accumulates decode(W) * x in f32, so with one-hot inputs the
//      output IS the raw f32 decode, compared BITWISE against the reference
//      f32 dequant (ref_f32.bin).
//
// usage: test-rocmfpx-mix-decode <qtype 105|106> <rows> <cols> <mode 0|1>
//            <packed.bin> <codebook.bin> <ref_fp16.bin> <ref_f32.bin>
// exit: 0 = bit-identical, 1 = mismatch/error, 77 = skipped (no GPU / registry)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
int main() { printf("SKIP: dlsym-based test, POSIX only\n"); return 77; }
#else
#include <dlfcn.h>

typedef void (*mix_register_host_fn)(const void *, size_t, int, int, int,
                                     const void *, const uint8_t *, const uint8_t *);
typedef void (*mix_unregister_fn)(const void *);

static std::vector<uint8_t> read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t) n) {
        fprintf(stderr, "short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    return buf;
}

int main(int argc, char ** argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s <qtype 105|106> <rows> <cols> <mode 0|1> "
                        "<packed.bin> <codebook.bin> <ref_fp16.bin> <ref_f32.bin>\n", argv[0]);
        return 1;
    }
    const int qtype = atoi(argv[1]);
    const int rows  = atoi(argv[2]);
    const int cols  = atoi(argv[3]);
    const uint8_t mode = (uint8_t) atoi(argv[4]);
    if ((qtype != 105 && qtype != 106) || rows <= 0 || cols <= 0 || cols % 128) {
        fprintf(stderr, "bad qtype/dims (cols must be a multiple of 128)\n");
        return 1;
    }
    const ggml_type wtype = (ggml_type) qtype;

    const std::vector<uint8_t> packed   = read_file(argv[5]);
    const std::vector<uint8_t> codebook = read_file(argv[6]);
    const std::vector<uint8_t> ref16    = read_file(argv[7]);
    const std::vector<uint8_t> ref32    = read_file(argv[8]);
    const size_t n_elem = (size_t) rows * cols;
    const size_t expect_packed = (size_t) rows * (cols / 32) * (qtype == 105 ? 14 : 10);
    const size_t expect_cb     = (size_t) 2 * (qtype == 105 ? 8 : 4) * 2;
    if (packed.size() != expect_packed || codebook.size() != expect_cb ||
        ref16.size() != n_elem * 2 || ref32.size() != n_elem * 4) {
        fprintf(stderr, "input size mismatch: packed %zu (want %zu), codebook %zu (want %zu), "
                        "ref16 %zu, ref32 %zu\n", packed.size(), expect_packed,
                        codebook.size(), expect_cb, ref16.size(), ref32.size());
        return 1;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("SKIP: no GPU backend device\n");
        return 77;
    }
    auto reg_host = (mix_register_host_fn) dlsym(RTLD_DEFAULT,
        qtype == 105 ? "ggml_cuda_rocmfp3_mix_register_host" : "ggml_cuda_rocmfp2_mix_register_host");
    auto unreg = (mix_unregister_fn) dlsym(RTLD_DEFAULT,
        qtype == 105 ? "ggml_cuda_rocmfp3_mix_unregister" : "ggml_cuda_rocmfp2_mix_unregister");
    if (!reg_host || !unreg) {
        printf("SKIP: mix registry symbols not present in this build\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        printf("SKIP: GPU backend failed to initialize\n");
        return 77;
    }

    const int nvec = 8; // <= MMVQ_MAX_BATCH_SIZE: routes into the fused matvec hook

    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W  = ggml_new_tensor_2d(ctx, wtype, cols, rows);
    ggml_tensor * X1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, cols);   // identity
    ggml_tensor * X2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, nvec);   // first 8 basis vecs
    ggml_set_name(W, "W_mix");
    ggml_tensor * D1 = ggml_mul_mat(ctx, W, X1); // (rows, cols): decode via dequant->cuBLAS
    ggml_tensor * D2 = ggml_mul_mat(ctx, W, X2); // (rows, 8):    decode via fused matvec
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, D1);
    ggml_build_forward_expand(gf, D2);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "graph allocation failed\n");
        return 1;
    }

    ggml_backend_tensor_set(W, packed.data(), 0, packed.size());
    {
        std::vector<float> ident((size_t) cols * cols, 0.0f);
        for (int j = 0; j < cols; ++j) {
            ident[(size_t) j * cols + j] = 1.0f;
        }
        ggml_backend_tensor_set(X1, ident.data(), 0, ident.size() * sizeof(float));
        std::vector<float> onehot((size_t) cols * nvec, 0.0f);
        for (int j = 0; j < nvec; ++j) {
            onehot[(size_t) j * cols + j] = 1.0f;
        }
        ggml_backend_tensor_set(X2, onehot.data(), 0, onehot.size() * sizeof(float));
    }

    // Register AFTER allocation: W->data is the device pointer the decode
    // kernels will look up. Dense registration: n_experts = 1, nb02 = nbytes.
    reg_host(W->data, ggml_nbytes(W), 1, rows, cols, codebook.data(), &mode, nullptr);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> d1(n_elem);
    std::vector<float> d2((size_t) rows * nvec);
    ggml_backend_tensor_get(D1, d1.data(), 0, d1.size() * sizeof(float));
    ggml_backend_tensor_get(D2, d2.data(), 0, d2.size() * sizeof(float));

    const uint16_t * r16 = (const uint16_t *) ref16.data();
    const float    * r32 = (const float    *) ref32.data();

    // 1. to_fp16 path: dst[i, j] (stored at j*rows + i) == fp16 reference, bitwise.
    size_t bad1 = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const float got = d1[(size_t) j * rows + i];
            const float want = ggml_fp16_to_fp32(r16[(size_t) i * cols + j]);
            if (memcmp(&got, &want, 4) != 0) {
                if (bad1 < 5) {
                    fprintf(stderr, "to_fp16 mismatch at [%d,%d]: got %a want %a\n", i, j, got, want);
                }
                ++bad1;
            }
        }
    }

    // 2. fused matvec path: dst[i, j] == f32 reference decode of W[i, j], bitwise.
    size_t bad2 = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < nvec; ++j) {
            const float got = d2[(size_t) j * rows + i];
            const float want = r32[(size_t) i * cols + j];
            if (memcmp(&got, &want, 4) != 0) {
                if (bad2 < 5) {
                    fprintf(stderr, "fused-matvec mismatch at [%d,%d]: got %a want %a\n", i, j, got, want);
                }
                ++bad2;
            }
        }
    }

    unreg(W->data);
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    printf("qtype %d mode %d %dx%d: to_fp16 path %zu/%zu mismatches, "
           "fused matvec path %zu/%zu mismatches\n",
           qtype, mode, rows, cols, bad1, n_elem, bad2, (size_t) rows * nvec);
    if (bad1 || bad2) {
        return 1;
    }
    printf("OK: both CUDA decode paths are bit-identical to the reference dequant\n");
    return 0;
}
#endif
