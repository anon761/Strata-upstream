// Real-artifact Q5_0 / Q5_1 PLE rows against ggml's reference dequantizer.
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ple_q5_parity <gguf-containing-ple>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const auto* tensor = gguf.find("per_layer_token_embd.weight");
    // Q5_0 (#296: OrcaRouter's Q4_K_S) and Q5_1 (the Uncensored finetune's Q5_K_M): 5 blocks of 32 per row
    if (!tensor || (tensor->type != 6 && tensor->type != 7) || tensor->shape.size() != 2 || tensor->shape[0] != 160) {
        std::fprintf(stderr, "expected a Q5_0 or Q5_1 PLE [160, N]\n");
        return 2;
    }
    const bool q5_1 = tensor->type == 7;
    const size_t row_bytes = q5_1 ? 120 : 110;
    strata::kernels::PleIoOptions options;
    options.mode = strata::kernels::PleIo::Mmap;
    strata::kernels::PleTable table;
    std::string err;
    if (!table.open(argv[1], err, options)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const auto* traits = ggml_get_type_traits(q5_1 ? GGML_TYPE_Q5_1 : GGML_TYPE_Q5_0);
    const uint8_t* bytes = gguf.tensor_data(*tensor);
    const uint32_t probes[] = {0, 1, 12345, 20000003, (uint32_t) (table.rows() - 1)};
    double max_abs = 0.0;
    for (uint32_t row : probes) {
        float got[160], want[160];
        table.read_row(row, got);
        traits->to_float(bytes + (size_t) row * row_bytes, want, 160);
        for (int i = 0; i < 160; ++i)
            max_abs = std::max(max_abs, (double) std::fabs(got[i] - want[i]));
    }
    uint32_t rows[16];
    for (int i = 0; i < 16; ++i) rows[i] = probes[i % 5];
    float batch[16 * 160];
    if (!table.issue(rows) || !table.collect(batch, err)) {
        std::fprintf(stderr, "collect: %s\n", err.c_str());
        return 1;
    }
    for (int h = 0; h < 16; ++h) {
        float want[160];
        traits->to_float(bytes + (size_t) rows[h] * row_bytes, want, 160);
        for (int i = 0; i < 160; ++i)
            max_abs = std::max(max_abs, (double) std::fabs(batch[h * 160 + i] - want[i]));
    }
    std::printf("%s PLE (%s): %zu rows, max_abs %.3e %s\n", q5_1 ? "Q5_1" : "Q5_0", table.format(),
                (size_t) table.rows(), max_abs,
                max_abs <= 1e-6 ? "PASS" : "FAIL");
    return max_abs <= 1e-6 ? 0 : 1;
}
