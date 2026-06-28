#pragma once
// causal-trans-conv.h: Causal ConvTranspose1d primitive for the
// Qwen3-TTS 12Hz tokenizer decoder.
//
// PyTorch reference (Qwen3TTSTokenizerV2CausalTransConvNet):
//   y = ConvTranspose1d(x, k, stride)         # raw length (T-1)*stride + K
//   y = y[...,: y.shape[-1] - (K - stride)]  # right-trim K-stride frames
//   final length: T * stride
//
// GGML implementation: the weight is pre-permuted at load time from the
// PyTorch (IC, OC, K) layout to a [IC, K*OC] layout with k varying
// faster than oc inside K*OC. The forward graph multiplies this weight
// against a channels-first input via ggml_mul_mat to produce a column
// matrix [K*OC, T_in], scatters it into [T_raw, OC] via ggml_col2im_1d
// with padding=0, right-trims to [T_in*stride, OC], transposes to
// channels-first [OC, T_in*stride], and adds the bias.

#include "ggml.h"
#include "gguf-weights.h"
#include "qt-error.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

// Load a ConvTranspose1d weight stored on disk in PyTorch layout
// (IC, OC, K) and pre-permute it to ggml [IC, K*OC] with k fastest in
// K*OC. Source dtype must be F32.
//
//   src flat[ic*OC*K + oc*K + k] = w[ic][oc][k]   PyTorch row-major
//   dst flat[(oc*K + k)*IC + ic] = w[ic][oc][k]   ggml row-major, ne=(IC, K*OC)
static struct ggml_tensor * qwen_load_ctw_f32(WeightCtx * wctx, const GGUFModel & gf, const std::string & name) {
    struct ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        qt_throw("[CausalTransConv] tensor '%s' not found", name.c_str());
    }
    // Source dtype follows the GGUF norm (pure llama.cpp policy). The F32
    // master keeps tensors in F32, the BF16 variant keeps them in their
    // source BF16, and the K-quant variants land them in F16 through the
    // aligned fallback (kernel rows of width K=2 do not divide a K-quant
    // block size). All three are widened to F32 here; the K*OC*IC
    // permutation always lands in a freshly allocated F32 buffer anyway.
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16) {
        qt_throw("[CausalTransConv] '%s' expected F32, F16 or BF16, got type %d", name.c_str(), (int) src->type);
    }
    int K  = (int) src->ne[0];
    int OC = (int) src->ne[1];
    int IC = (int) src->ne[2];

    struct ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, IC, K * OC);
    ggml_set_name(dst, name.c_str());

    const void * raw  = gf_get_data(gf, name.c_str());
    auto         buf  = std::make_unique<float[]>((size_t) IC * (size_t) K * (size_t) OC);
    float *      dstp = buf.get();

    auto load_src = [&](size_t idx) -> float {
        if (src->type == GGML_TYPE_F32) {
            return ((const float *) raw)[idx];
        }
        if (src->type == GGML_TYPE_F16) {
            return ggml_fp16_to_fp32(((const ggml_fp16_t *) raw)[idx]);
        }
        return ggml_bf16_to_fp32(((const ggml_bf16_t *) raw)[idx]);
    };

    for (int ic = 0; ic < IC; ic++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                dstp[(size_t) (oc * K + k) * IC + ic] = load_src((size_t) ic * OC * K + oc * K + k);
            }
        }
    }

    wctx->pending.push_back({ dst, dstp, (size_t) IC * (size_t) K * (size_t) OC * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return dst;
}

// Causal ConvTranspose1d forward graph.
//   w_perm: [IC, K*OC] f32, pre-permuted by qwen_load_ctw_f32
//   b: [OC] f32 or NULL
//   x: [T_in, IC] f32, T-first
//   stride: upsample factor
//   kernel: kernel size
//   oc: output channels (must match the K*OC factorization of w_perm)
// Returns [T_in*stride, OC] f32, T-first.
static struct ggml_tensor * qwen_causal_trans_conv1d(struct ggml_context * ctx,
                                                     struct ggml_tensor *  w_perm,
                                                     struct ggml_tensor *  b,
                                                     struct ggml_tensor *  x,
                                                     int                   stride,
                                                     int                   kernel,
                                                     int                   oc) {
    int trim = kernel - stride;

    // Transpose x to channels-first [IC, T_in] for the mul_mat contraction
    struct ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));

    // mul_mat contracts over IC: col [K*OC, T_in]
    struct ggml_tensor * col = ggml_mul_mat(ctx, w_perm, xt);

    // col2im_1d with padding=0: [T_raw, OC] T-first, T_raw = (T_in-1)*stride + K
    struct ggml_tensor * y = ggml_col2im_1d(ctx, col, stride, oc, 0);

    // Right-trim K-stride frames -> [T_in*stride, OC] T-first
    if (trim > 0) {
        int64_t T_keep = y->ne[0] - trim;
        y              = ggml_view_2d(ctx, y, T_keep, y->ne[1], y->nb[1], 0);
    }

    if (b) {
        // bias [OC] broadcasts as (1, OC) onto (T, OC) via ne[0]=1
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Causal Conv1d with optional stride. Left pad with (kernel_eff - stride),
// add an extra right pad to align with stride boundaries, then run a
// standard ggml_conv_1d. Matches MimiConv1d.causal forward exactly:
//   kernel_eff   = (k - 1) * d + 1
//   padding_total = kernel_eff - stride
//   extra_pad    = ceil((T + padding_total - kernel_eff) / stride) * stride
//                  + kernel_eff - padding_total - T
//                = (T - 1) % stride for the common case
// The output length is (T + padding_total + extra_pad - kernel_eff) / stride + 1
// = ceil(T / stride). Stride defaults to 1 to preserve the Qwen3 causal
// path used by pre_conv and the DAC decoder.
//   w: [k, IC, OC] f32, source layout (K, IC, OC) maps to ggml ne directly
//   b: [OC] f32 or NULL
//   x: [T, IC] f32 T-first
//   pad_mode: CTC_PAD_CONSTANT (zero pad, default for SEANet and the DAC
//             decoder) or CTC_PAD_REPLICATE (edge pad, replicates the
//             first / last frame to match Mimi's downsample which is the
//             only conv passing pad_mode="replicate" upstream).
// Returns [ceil(T / stride), OC] f32 T-first.
enum QwenPadMode {
    CTC_PAD_CONSTANT  = 0,
    CTC_PAD_REPLICATE = 1,
};

static struct ggml_tensor * qwen_causal_conv1d(struct ggml_context * ctx,
                                               struct ggml_tensor *  w,
                                               struct ggml_tensor *  b,
                                               struct ggml_tensor *  x,
                                               int                   k,
                                               int                   d,
                                               int                   s        = 1,
                                               int                   pad_mode = CTC_PAD_CONSTANT) {
    int OC          = (int) w->ne[2];
    int kernel_eff  = (k - 1) * d + 1;
    int padding_tot = kernel_eff - s;

    // Mimi extra padding: ensures the causal conv lands on a stride boundary
    // by extending the input on the right with zeros or replicated edges
    // depending on pad_mode.
    int T         = (int) x->ne[0];
    int n_frames  = (T + padding_tot - kernel_eff + s - 1) / s + 1;
    int ideal_len = (n_frames - 1) * s + kernel_eff - padding_tot;
    int extra_pad = ideal_len - T;
    if (extra_pad < 0) {
        extra_pad = 0;
    }

    struct ggml_tensor * y = x;
    if (pad_mode == CTC_PAD_REPLICATE) {
        // Edge pad: repeat x[t=0] padding_tot times on the left and x[t=T-1]
        // extra_pad times on the right via a single ggml_repeat per side.
        int IC = (int) x->ne[1];
        if (padding_tot > 0) {
            struct ggml_tensor * first = ggml_view_2d(ctx, x, 1, IC, x->nb[1], 0);
            struct ggml_tensor * tmpl  = ggml_new_tensor_2d(ctx, x->type, padding_tot, IC);
            struct ggml_tensor * lp    = ggml_repeat(ctx, first, tmpl);
            y                          = ggml_concat(ctx, lp, y, 0);
        }
        if (extra_pad > 0) {
            size_t               last_off = (size_t) (T - 1) * x->nb[1];
            struct ggml_tensor * last     = ggml_view_2d(ctx, x, 1, IC, x->nb[1], last_off);
            struct ggml_tensor * tmpl     = ggml_new_tensor_2d(ctx, x->type, extra_pad, IC);
            struct ggml_tensor * rp       = ggml_repeat(ctx, last, tmpl);
            y                             = ggml_concat(ctx, y, rp, 0);
        }
    } else if (padding_tot > 0 || extra_pad > 0) {
        y = ggml_pad_ext(ctx, y, padding_tot, extra_pad, 0, 0, 0, 0, 0, 0);
    }

    // ggml_conv_1d expects 3D input [T, IC, N], add the batch dim
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], 1);
    y = ggml_conv_1d(ctx, w, y, s, 0, d);
    // squeeze batch back to 2D
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);

    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, OC);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}
