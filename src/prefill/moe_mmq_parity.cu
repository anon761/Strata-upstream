// src/prefill/moe_mmq_parity.cu - fork diagnostic: one native expert's gate/up through the prefill MMQ against
// ggml's own dequantizer, on a real GGUF shard.  Validates the MMQ path for the ordinary quant types (Q4_K,
// Q5_K, Q8_0, Q5_1, ...) the same way native_expert_parity validates the grouped decode kernels.
//
//     build/moe_mmq_parity <shard.gguf> [layer]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moe_mmq_parity <shard.gguf> [layer]\n"); return 2; }
    strata::GgufFile gguf(argv[1]);
    const int layer = argc > 2 ? std::atoi(argv[2]) : 0;
    const int64_t H = 2560, FF = 640, T = 4;
    const strata::TensorInfo *tg = nullptr, *tu = nullptr, *td = nullptr;
    for (const auto& ti : gguf.tensors()) {
        if (ti.name == "blk." + std::to_string(layer) + ".ffn_gate_exps.weight") tg = &ti;
        if (ti.name == "blk." + std::to_string(layer) + ".ffn_up_exps.weight") tu = &ti;
        if (ti.name == "blk." + std::to_string(layer) + ".ffn_down_exps.weight") td = &ti;
    }
    if (!tg || !tu || !td) { std::printf("layer %d: no expert tensors in this shard\n", layer); return 1; }
    const int gu_type = (int) tg->type;
    const size_t row = (size_t) ggml_row_size((ggml_type) gu_type, H);
    const size_t gu_half = row * (size_t) FF;   // one expert's gate (or up) region
    std::vector<uint8_t> gub(2 * gu_half);
    std::memcpy(gub.data(), gguf.tensor_data(*tg), gu_half);
    std::memcpy(gub.data() + gu_half, gguf.tensor_data(*tu), gu_half);

    const auto* tt = ggml_get_type_traits((ggml_type) gu_type);
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H);
    for (int64_t r = 0; r < FF; ++r) {
        tt->to_float(gub.data() + (size_t) r * row, G.data() + r * H, H);
        tt->to_float(gub.data() + gu_half + (size_t) r * row, U.data() + r * H, H);
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) T * H);
    for (auto& v : x) v = nd(rng);

    // the MMQ gate/up product, exactly as prefill.cpp sets it up (gate rows then up rows, one expert, ids identity)
    cudaStream_t s;
    cudaStreamCreate(&s);
    void *dgu, *dxq, *dbounds, *dids;
    float *ddst, *dx;
    cudaMalloc(&dgu, gub.size());
    cudaMalloc((void**) &dx, x.size() * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dxq, strata::prefill::mmq::q8_bytes(T, H));
    cudaMalloc(&dbounds, 2 * 4);
    cudaMalloc(&dids, T * 4);
    cudaMalloc((void**) &ddst, (size_t) 1280 * T * 4);
    cudaMemcpy(dgu, gub.data(), gub.size(), cudaMemcpyHostToDevice);
    const int32_t hb[2] = {0, (int32_t) T}, ids[T] = {0, 1, 2, 3};
    cudaMemcpy(dbounds, hb, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dids, ids, T * 4, cudaMemcpyHostToDevice);
    strata::prefill::mmq::quantize(dx, nullptr, dxq, gu_type, H, H, T, s);
    strata::prefill::mmq::Context ctx;
    strata::prefill::mmq::Product p;
    p.w = dgu;
    p.type = gu_type;
    p.w_rows = 1280;
    p.w_cols = H;
    p.expert_bytes = strata::prefill::mmq::matrix_bytes(gu_type, 1280, H);
    p.n = 1;
    p.xq = dxq;
    p.bounds = (const int32_t*) dbounds;
    p.ids = (const int32_t*) dids;
    p.total_rows = T;
    p.max_rows = T;
    p.dst = ddst;
    p.ld_dst = 1280;
    ctx.run(p, s);
    cudaStreamSynchronize(s);
    std::vector<float> got((size_t) 1280 * T);
    cudaMemcpy(got.data(), ddst, got.size() * 4, cudaMemcpyDeviceToHost);

    std::vector<float> ref((size_t) 1280 * T);
    for (int t = 0; t < T; ++t)
        for (int64_t r = 0; r < 1280; ++r) {
            const float* w = (r < FF ? G.data() + r * H : U.data() + (r - FF) * H);
            double o = 0;
            for (int64_t i = 0; i < H; ++i) o += (double) w[i] * x[t * H + i];
            ref[t * 1280 + r] = (float) o;
        }
    const double e = rel(got, ref);
    std::printf("layer %d gate/up %s: MMQ rel %.2e %s\n", layer, ggml_type_name((ggml_type) gu_type), e,
                e < 3e-2 ? "ok" : "FAIL");
    cudaFree(dgu); cudaFree(dxq); cudaFree(ddst); cudaFree(dx);

    // the down product (Q5_1/Q8_0 in the ordinary files), exactly as prefill.cpp sets it up
    const int d_type = (int) td->type;
    const size_t d_row = (size_t) ggml_row_size((ggml_type) d_type, FF);
    const size_t d_bytes = d_row * (size_t) H;
    const auto* ttd = ggml_get_type_traits((ggml_type) d_type);
    std::vector<float> D((size_t) H * FF);
    for (int64_t r = 0; r < H; ++r) ttd->to_float(gguf.tensor_data(*td) + (size_t) r * d_row, D.data() + r * FF, FF);
    std::vector<float> h((size_t) T * FF);
    for (auto& v : h) v = nd(rng);
    void *dd, *dhq;
    float *ddout, *dh;
    cudaMalloc(&dd, d_bytes);
    cudaMalloc((void**) &dh, h.size() * 4);
    cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dhq, strata::prefill::mmq::q8_bytes(T, FF));
    cudaMalloc((void**) &ddout, (size_t) H * T * 4);
    cudaMemcpy(dd, gguf.tensor_data(*td), d_bytes, cudaMemcpyHostToDevice);
    strata::prefill::mmq::quantize(dh, nullptr, dhq, d_type, FF, FF, T, s);
    strata::prefill::mmq::Product dp;
    dp.w = dd;
    dp.type = d_type;
    dp.w_rows = H;
    dp.w_cols = FF;
    dp.expert_bytes = strata::prefill::mmq::matrix_bytes(d_type, H, FF);
    dp.n = 1;
    dp.xq = dhq;
    dp.bounds = (const int32_t*) dbounds;
    dp.ids = (const int32_t*) dids;
    dp.total_rows = T;
    dp.max_rows = T;
    dp.dst = ddout;
    dp.ld_dst = H;
    ctx.run(dp, s);
    cudaStreamSynchronize(s);
    std::vector<float> gotd((size_t) H * T);
    cudaMemcpy(gotd.data(), ddout, gotd.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> refd((size_t) H * T);
    for (int t = 0; t < T; ++t)
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[t * FF + i];
            refd[t * H + r] = (float) o;
        }
    const double ed = rel(gotd, refd);
    std::printf("layer %d down %s: MMQ rel %.2e %s  got[0]=%.4f ref[0]=%.4f got[1]=%.4f ref[1]=%.4f\n", layer,
                ggml_type_name((ggml_type) d_type), ed, ed < 3e-2 ? "ok" : "FAIL", gotd[0], refd[0], gotd[1], refd[1]);
    cudaFree(dd); cudaFree(dhq); cudaFree(ddout); cudaFree(dh); cudaFree(dbounds); cudaFree(dids);

    // the dense projection path (native_mmvq), e.g. an ordinary file's Q8_0 attention/shexp weight and the head
    {
        const strata::TensorInfo* dw = nullptr;
        for (const auto& ti : gguf.tensors())
            if (ti.name == "blk." + std::to_string(layer) + ".ffn_gate_shexp.weight") dw = &ti;
        if (dw != nullptr && strata::kernels::native_mmvq_supported((int) dw->type)) {
            const int n_in = (int) H, n_out = (int) dw->shape[1];
            const size_t wrow = (size_t) ggml_row_size((ggml_type) dw->type, n_in);
            const size_t wb = strata::kernels::native_mmvq_weight_bytes((int) dw->type, n_in, n_out);
            std::vector<uint8_t> W(wb);
            std::memcpy(W.data(), gguf.tensor_data(*dw), wb);
            const auto* tw = ggml_get_type_traits((ggml_type) dw->type);
            std::vector<float> Wf((size_t) n_out * n_in);
            for (int64_t r = 0; r < n_out; ++r) tw->to_float(W.data() + (size_t) r * wrow, Wf.data() + r * n_in, n_in);
            std::vector<float> xv(n_in), refm(n_out);
            for (auto& v : xv) v = nd(rng);
            for (int64_t r = 0; r < n_out; ++r) {
                double o = 0;
                for (int64_t i = 0; i < n_in; ++i) o += (double) Wf[r * n_in + i] * xv[i];
                refm[r] = (float) o;
            }
            void *dW, *dx, *dsc;
            float* dym;
            cudaMalloc(&dW, wb);
            cudaMalloc(&dx, n_in * 4);
            cudaMalloc(&dsc, strata::kernels::native_q8_1_bytes(n_in, 1));
            cudaMalloc((void**) &dym, n_out * 4);
            cudaMemcpy(dW, W.data(), wb, cudaMemcpyHostToDevice);
            cudaMemcpy(dx, xv.data(), n_in * 4, cudaMemcpyHostToDevice);
            std::fprintf(stderr, "dense %s n_in=%d n_out=%d wb=%zu ptrs %p %p %p %p\n",
                         ggml_type_name((ggml_type) dw->type), n_in, n_out, wb, dW, dx, dsc, dym);
            strata::kernels::native_quantize_q8_1((const float*) dx, dsc, n_in, 1, s);
            strata::kernels::native_mmvq((int) dw->type, dW, dsc, dym, n_in, n_out, 1, s);
            cudaStreamSynchronize(s);
            std::vector<float> gotm(n_out);
            cudaMemcpy(gotm.data(), dym, n_out * 4, cudaMemcpyDeviceToHost);
            const double em = rel(gotm, refm);
            std::printf("layer %d dense %s: native_mmvq rel %.2e %s\n", layer, ggml_type_name((ggml_type) dw->type),
                        em, em < 3e-2 ? "ok" : "FAIL");
            cudaFree(dW); cudaFree(dx); cudaFree(dsc); cudaFree(dym);
        }
    }
    return (e < 3e-2 && ed < 3e-2) ? 0 : 1;
}
