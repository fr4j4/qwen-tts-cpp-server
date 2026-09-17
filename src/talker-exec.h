#pragma once
// talker-exec.h: pre-built replayable graph for the Talker decode step
// (one token per audio frame, appended to the KV cache).
//
// The legacy path (talker_forward_core) rebuilds a ~1,400-node graph for
// every decode step: ggml_init + node builds + sched reset/alloc + mask
// construction + 3 uploads + compute + readbacks, 12.5 times per second
// of audio. The compute itself is ~1.5 ms on an RTX 3060; the rest is
// launch overhead.
//
// This executor builds the decode graph once (T=1, fixed KV read window
// W) on its own isolated scheduler and replays it. Per step it refreshes
// four tiny leaf inputs:
//   x_in      [hidden]        the embedding of the token just sampled
//   pos_in    [1] i32         absolute position n_past
//   mask_in   [W] f16         causal row for absolute position n_past
//   setidx_in [1] i32         cache write row (== n_past)
// K/V are written into the persistent cache with ggml_set_rows (dynamic
// row index), so no graph rebuild is ever needed. The attention reads a
// FIXED [hd, W, n_kv] extent of the cache; rows beyond n_past are hidden
// by the causal mask row (0 = attend, -inf = ignore).
//
// Isolated scheduler (same pattern as code-predictor-exec.h): alloc-once
// at init, so pool growth from variable-size prefills on the shared
// scheduler can never move or free these buffers.
//
// Window guard: if kv->cur_len >= W the frame falls back to the legacy
// rebuild path (rare: W=2048 covers prompt ~250 + ~1800 generated).
// GPU-only: requires use_flash_attn (fused FA kernel), like cp_exec.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
#include "talker-forward.h"
#include "talker-weights.h"

#include <cstdint>
#include <cstring>
#include <vector>

struct TalkerExec {
    const TalkerWeights * tw    = nullptr;
    KVCache *             kv    = nullptr;
    BackendPair           bp    = {};
    int  hidden  = 0;
    int  vocab   = 0;
    int  W       = 0;              // fixed KV read window
    bool clamp_fp16 = false;

    struct ggml_context * gctx    = nullptr;
    struct ggml_cgraph *  gf      = nullptr;
    struct ggml_tensor *  x_in    = nullptr;   // [hidden]        f32
    struct ggml_tensor *  pos_in  = nullptr;   // [1]             i32
    struct ggml_tensor *  mask_in = nullptr;   // [W]             f16
    struct ggml_tensor *  setidx  = nullptr;   // [1]             i32
    struct ggml_tensor *  logits  = nullptr;   // [vocab]         out
    struct ggml_tensor *  hfin    = nullptr;   // [hidden]        out

    ggml_backend_sched_t sched = nullptr;

    // Host-side static causal table [W, W] f16: row p is the attention
    // mask for absolute position p (0 for k <= p, -inf beyond).
    std::vector<ggml_fp16_t> causal;

    std::vector<float> logits_out;
    std::vector<float> hidden_out;
};

// One decoder layer for the persistent graph. Identical math to
// talker_layer_forward, with the KV handling replaced: write via
// ggml_set_rows at a dynamic row (setidx leaf), read a FIXED [hd, W,
// n_kv] extent, mask from the mask_in leaf.
static struct ggml_tensor * talker_exec_layer(struct ggml_context *        ctx,
                                              const TalkerWeights *        tw,
                                              const TalkerLayer &          layer,
                                              struct ggml_tensor *         x,
                                              struct ggml_tensor *         pos_in,
                                              struct ggml_tensor *         mask_in,
                                              struct ggml_tensor *         setidx,
                                              struct ggml_tensor *         k_cache,
                                              struct ggml_tensor *         v_cache,
                                              int                          n_kv_heads,
                                              int                          W,
                                              bool                         clamp_fp16,
                                              struct ggml_cgraph *         gf) {
    const int n_q_heads = tw->num_attention_heads;
    const int n_kv      = n_kv_heads;
    const int hd        = tw->head_dim;
    const float eps     = tw->rms_norm_eps;

    struct ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    h                      = ggml_mul(ctx, h, layer.input_norm_w);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, h);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, h);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, h);

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, 1);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, 1);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, 1);

    q = ggml_rms_norm(ctx, q, eps);
    q = ggml_mul(ctx, q, layer.attn.q_norm_w);
    k = ggml_rms_norm(ctx, k, eps);
    k = ggml_mul(ctx, k, layer.attn.k_norm_w);

    q = ggml_rope_ext(ctx, q, pos_in, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, pos_in, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, tw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // K/V row for this step: [hd, 1, n_kv], written into the persistent
    // cache at the dynamic row held in setidx.
    struct ggml_tensor * k_row = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    struct ggml_tensor * v_row = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    struct ggml_tensor * k_set = ggml_set_rows(ctx, k_cache, k_row, setidx);
    struct ggml_tensor * v_set = ggml_set_rows(ctx, v_cache, v_row, setidx);
    ggml_build_forward_expand(gf, k_set);
    ggml_build_forward_expand(gf, v_set);

    // Fixed-extent reads: [hd, W, n_kv]. Rows beyond the current position
    // carry stale/garbage values; the causal mask row hides them.
    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache, hd, W, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache, hd, W, n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    if (clamp_fp16) {
        v_full = ggml_clamp(ctx, v_full, -65504.0f, 65504.0f);
    }

    // Q: [hd, n_q_heads, 1] -> [hd, 1, n_q_heads] for flash_attn_ext.
    struct ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);

    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn  = ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask_in, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, 1);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);
    x                      = ggml_add(ctx, x, o);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }

    struct ggml_tensor * h2 = ggml_rms_norm(ctx, x, eps);
    h2                      = ggml_mul(ctx, h2, layer.post_attn_norm_w);

    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
    gate                      = ggml_silu(ctx, gate);
    struct ggml_tensor * gu   = ggml_mul(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    x = ggml_add(ctx, x, mlp);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }
    return x;
}

static bool talker_exec_init(TalkerExec *          te,
                             const TalkerWeights * tw,
                             KVCache *             kv,
                             BackendPair           bp,
                             int                   W,
                             bool                  clamp_fp16) {
    te->tw        = tw;
    te->kv        = kv;
    te->bp        = bp;
    te->hidden    = tw->hidden_size;
    te->vocab     = tw->vocab_size;
    te->W         = W;
    te->clamp_fp16 = clamp_fp16;

    // Static causal table: row p allows k <= p.
    te->causal.assign((size_t) W * (size_t) W, ggml_fp32_to_fp16(-INFINITY));
    for (int p = 0; p < W; p++) {
        for (int k = 0; k <= p; k++) {
            te->causal[(size_t) p * (size_t) W + (size_t) k] = ggml_fp32_to_fp16(0.0f);
        }
    }

    const int    n_layers    = tw->num_hidden_layers;
    const int    max_nodes   = 48 * n_layers + 128;
    const size_t arena_bytes = ggml_tensor_overhead() * max_nodes +
                               ggml_graph_overhead_custom(max_nodes, false);

    struct ggml_init_params gp = { arena_bytes, NULL, true };
    te->gctx                   = ggml_init(gp);
    if (!te->gctx) {
        return false;
    }

    te->x_in    = ggml_new_tensor_1d(te->gctx, GGML_TYPE_F32, te->hidden);
    te->pos_in  = ggml_new_tensor_1d(te->gctx, GGML_TYPE_I32, 1);
    te->mask_in = ggml_new_tensor_1d(te->gctx, GGML_TYPE_F16, W);
    te->setidx  = ggml_new_tensor_1d(te->gctx, GGML_TYPE_I32, 1);
    ggml_set_name(te->x_in, "talker_exec_x");
    ggml_set_name(te->pos_in, "talker_exec_pos");
    ggml_set_name(te->mask_in, "talker_exec_mask");
    ggml_set_name(te->setidx, "talker_exec_setidx");

    te->gf = ggml_new_graph_custom(te->gctx, max_nodes, false);

    struct ggml_tensor * h = te->x_in;
    for (int l = 0; l < n_layers; l++) {
        h = talker_exec_layer(te->gctx, tw, tw->layers[(size_t) l], h, te->pos_in, te->mask_in, te->setidx,
                              kv->k[(size_t) l], kv->v[(size_t) l], tw->num_key_value_heads, W, clamp_fp16,
                              te->gf);
    }

    struct ggml_tensor * h_final = ggml_rms_norm(te->gctx, h, tw->rms_norm_eps);
    h_final                      = ggml_mul(te->gctx, h_final, tw->norm_w);
    ggml_set_name(h_final, "hidden_final");
    ggml_set_output(h_final);

    struct ggml_tensor * logits = ggml_mul_mat(te->gctx, tw->codec_head_w, h_final);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(te->gf, h_final);
    ggml_build_forward_expand(te->gf, logits);

    // Isolated scheduler, alloc-once: talker prefills on the shared
    // scheduler can grow their pool without touching these buffers.
    te->sched = backend_sched_new(te->bp, 4096);
    if (!te->sched) {
        ggml_free(te->gctx);
        te->gctx = nullptr;
        return false;
    }
    ggml_backend_sched_reset(te->sched);
    if (!ggml_backend_sched_alloc_graph(te->sched, te->gf)) {
        ggml_backend_sched_free(te->sched);
        te->sched = nullptr;
        ggml_free(te->gctx);
        te->gctx = nullptr;
        return false;
    }
    // [tx-graph-capture] Instrumentación temporal: nodos reales del grafo.
    fprintf(stderr, "[TalkerExec] graph size=%d\n", ggml_graph_size(te->gf));

    // Upload the process-lifetime constants once. mask_in starts as the
    // row for position 0; it is refreshed per step with the row for
    // n_past.
    ggml_backend_tensor_set(te->mask_in, te->causal.data(), 0, (size_t) W * sizeof(ggml_fp16_t));
    const int32_t zero_idx = 0;
    ggml_backend_tensor_set(te->setidx, &zero_idx, 0, sizeof(int32_t));
    const int32_t zero_pos = 0;
    ggml_backend_tensor_set(te->pos_in, &zero_pos, 0, sizeof(int32_t));

    te->logits_out.assign((size_t) te->vocab, 0.0f);
    te->hidden_out.assign((size_t) te->hidden, 0.0f);
    return true;
}

static void talker_exec_free(TalkerExec * te) {
    if (te->sched) {
        ggml_backend_sched_free(te->sched);
        te->sched = nullptr;
    }
    if (te->gctx) {
        ggml_free(te->gctx);
        te->gctx = nullptr;
    }
    te->causal.clear();
}

// Replay the decode graph for the token embedding in input_embed_1 at
// absolute position n_past (= kv->cur_len). Outputs match
// talker_forward_decode exactly: hidden_last (post final norm) and the
// full logits row. Callers must ensure n_past < W (guard in the pipeline).
static bool talker_exec_decode(TalkerExec *  te,
                               const float * input_embed_1,
                               int           n_past,
                               TalkerForwardOutput * out) {
    const int W = te->W;

    ggml_backend_tensor_set(te->x_in, input_embed_1, 0, (size_t) te->hidden * sizeof(float));

    const int32_t pos = n_past;
    ggml_backend_tensor_set(te->pos_in, &pos, 0, sizeof(int32_t));

    // Mask row for the absolute query position (static table, one row).
    ggml_backend_tensor_set(te->mask_in, te->causal.data() + (size_t) n_past * (size_t) W, 0,
                            (size_t) W * sizeof(ggml_fp16_t));

    const int32_t row = n_past;
    ggml_backend_tensor_set(te->setidx, &row, 0, sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(te->sched, te->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[TalkerExec] FATAL: decode graph compute failed (n_past=%d)\n", n_past);
        return false;
    }

    out->hidden = te->hidden;
    out->vocab  = te->vocab;
    out->hidden_last.assign((size_t) te->hidden, 0.0f);
    out->logits_last.assign((size_t) te->vocab, 0.0f);
    ggml_backend_tensor_get(ggml_graph_get_tensor(te->gf, "logits"), out->logits_last.data(), 0,
                            (size_t) te->vocab * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(te->gf, "hidden_final"), out->hidden_last.data(), 0,
                            (size_t) te->hidden * sizeof(float));

    te->kv->cur_len = n_past + 1;
    return true;
}
