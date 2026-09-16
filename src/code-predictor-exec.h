#pragma once
// code-predictor-exec.h: pre-built replayable graphs for the 15 predictor
// sub-steps of one audio frame.
//
// The legacy path (code-predictor-forward.h) rebuilds the ggml graph for
// every sub-step of every frame: ggml_init + ~250 node builds + sched
// alloc + position/mask upload + compute, 15 times per 83 ms of audio.
// The graphs are identical every time: shape depends only on the
// sub-step index (T=2 prefill for g=0, T=1 decode for g=1..14, cache
// position n_past = g+1, lm_head g). Positions and the causal mask are
// constants of the sub-step.
//
// This executor builds each sub-step graph once at load and replays it:
// per frame the only per-step work is sched alloc, x_in upload (the
// embedding of the previously sampled code), compute and the logits
// readback. Sampling stays host side with the exact same Philox
// subsequences as the legacy path, so codes are bit identical.
//
// Fallback: if init fails the pipeline keeps the legacy rebuild path
// (cp_exec == NULL). KALI_QWEN_CP_LEGACY=1 forces the legacy path for
// A/B benchmarking.

#include "code-predictor-forward.h"  // reuses the layer builder + embed_row_from_backend
#include "code-predictor-weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
#include "sampling.h"
#include "talker-weights.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

struct CPExecStep {
    struct ggml_context * gctx    = nullptr;
    struct ggml_cgraph *  gf      = nullptr;
    struct ggml_tensor *  x_in    = nullptr;
    struct ggml_tensor *  pos_in  = nullptr;
    struct ggml_tensor *  mask_in = nullptr;
    int T      = 0;
    int n_past = 0;
    int g_head = 0;
};

struct CodePredictorExec {
    const CodePredictorWeights * cw    = nullptr;
    KVCache *                    kv    = nullptr;
    ggml_backend_sched_t         sched = nullptr;
    int  talker_hidden = 0;
    int  vocab         = 0;
    int  n_acoustic    = 0;
    bool use_flash_attn = false;
    bool clamp_fp16     = false;

    std::vector<CPExecStep> steps;
    std::vector<float>      prefill_input;  // 2 * talker_hidden
    std::vector<float>      step_input;     // talker_hidden
    std::vector<float>      logits;         // vocab

    // Per-step constants uploaded once at first use: positions [0..T) at
    // the step's cache offset and the causal F16 mask. Indexed by
    // sub-step (g).
    std::vector<std::vector<int32_t>>     positions;
    std::vector<std::vector<ggml_fp16_t>> masks;
};

// Build one sub-step graph: same node order and params as the legacy
// code_predictor_run, with the position / mask tensors kept as handleable
// members so the replay can refresh them after every sched alloc.
static bool cp_exec_build_step(CodePredictorExec * ex, CPExecStep * st, int T, int n_past, int g_head) {
    const CodePredictorWeights * cw      = ex->cw;
    const int                    n_layers = cw->num_hidden_layers;
    const int                    max_nodes = 48 * n_layers + 64;
    const size_t                 arena_bytes = ggml_tensor_overhead() * max_nodes +
                                               ggml_graph_overhead_custom(max_nodes, false);

    struct ggml_init_params gp  = { arena_bytes, NULL, true };
    st->gctx                    = ggml_init(gp);
    if (!st->gctx) {
        return false;
    }
    st->T      = T;
    st->n_past = n_past;
    st->g_head = g_head;

    st->x_in    = ggml_new_tensor_2d(st->gctx, GGML_TYPE_F32, ex->talker_hidden, T);
    st->pos_in  = ggml_new_tensor_1d(st->gctx, GGML_TYPE_I32, T);
    st->mask_in = ggml_new_tensor_2d(st->gctx, GGML_TYPE_F16, n_past + T, T);
    ggml_set_name(st->x_in, "sub_input");
    ggml_set_name(st->pos_in, "positions");
    ggml_set_name(st->mask_in, "causal_mask");

    st->gf = ggml_new_graph_custom(st->gctx, max_nodes, false);

    struct ggml_tensor * h = st->x_in;
    if (cw->mtp_proj_w) {
        h = ggml_mul_mat(st->gctx, cw->mtp_proj_w, h);
        if (cw->mtp_proj_b) {
            h = ggml_add(st->gctx, h, cw->mtp_proj_b);
        }
        ggml_set_name(h, "mtp_proj_out");
    }

    for (int l = 0; l < n_layers; l++) {
        h = code_predictor_layer_forward(st->gctx, cw, cw->layers[(size_t) l], h, st->pos_in, st->mask_in,
                                         ex->kv->k[(size_t) l], ex->kv->v[(size_t) l], n_past, T,
                                         ex->use_flash_attn, ex->clamp_fp16, st->gf);
    }

    struct ggml_tensor * h_final = ggml_rms_norm(st->gctx, h, cw->rms_norm_eps);
    h_final                      = ggml_mul(st->gctx, h_final, cw->norm_w);

    struct ggml_tensor * logits = ggml_mul_mat(st->gctx, cw->lm_head[(size_t) g_head], h_final);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(st->gf, logits);
    return true;
}

static bool code_predictor_exec_init(CodePredictorExec *          ex,
                                     const CodePredictorWeights * cw,
                                     KVCache *                    kv,
                                     ggml_backend_sched_t         sched,
                                     int                          talker_hidden,
                                     bool                         use_flash_attn,
                                     bool                         clamp_fp16) {
    ex->cw             = cw;
    ex->kv             = kv;
    ex->sched          = sched;
    ex->talker_hidden  = talker_hidden;
    ex->vocab          = cw->vocab_size;
    ex->n_acoustic     = cw->num_acoustic_codebooks;
    ex->use_flash_attn = use_flash_attn;
    ex->clamp_fp16     = clamp_fp16;

    ex->steps.assign((size_t) ex->n_acoustic, CPExecStep{});
    // Step 0 is the two-position prefill (talker hidden + embed(c0)) that
    // reads lm_head 0; step g feeds one embedding at cache position g+1
    // and reads lm_head g, mirroring the legacy loop exactly.
    if (!cp_exec_build_step(ex, &ex->steps[0], 2, 0, 0)) {
        goto fail;
    }
    for (int g = 1; g < ex->n_acoustic; g++) {
        if (!cp_exec_build_step(ex, &ex->steps[(size_t) g], 1, g + 1, g)) {
            goto fail;
        }
    }
    ex->prefill_input.assign((size_t) 2 * (size_t) talker_hidden, 0.0f);
    ex->step_input.assign((size_t) talker_hidden, 0.0f);
    ex->logits.assign((size_t) ex->vocab, 0.0f);

    // Precompute every step's position vector and causal mask. These are
    // pure functions of (T, n_past), which are frozen at build time.
    ex->positions.resize(ex->steps.size());
    ex->masks.resize(ex->steps.size());
    for (size_t i = 0; i < ex->steps.size(); i++) {
        const CPExecStep & s = ex->steps[i];
        ex->positions[i].resize((size_t) s.T);
        for (int t = 0; t < s.T; t++) {
            ex->positions[i][(size_t) t] = s.n_past + t;
        }
        ex->masks[i].assign((size_t) s.T * (size_t)(s.n_past + s.T), ggml_fp32_to_fp16(-INFINITY));
        for (int q = 0; q < s.T; q++) {
            const int q_pos = s.n_past + q;
            for (int k = 0; k <= q_pos; k++) {
                ex->masks[i][(size_t) q * (size_t)(s.n_past + s.T) + (size_t) k] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
    return true;

fail:
    for (auto & s : ex->steps) {
        if (s.gctx) {
            ggml_free(s.gctx);
        }
    }
    ex->steps.clear();
    return false;
}

static void code_predictor_exec_free(CodePredictorExec * ex) {
    for (auto & s : ex->steps) {
        if (s.gctx) {
            ggml_free(s.gctx);
        }
    }
    ex->steps.clear();
}

// Replay one sub-step graph on the SHARED pipeline scheduler. Each call
// resets + allocates + uploads positions / mask / input, computes and
// reads the logits row back.
//
// NOTE (perf lesson): a per-step persistent scheduler (alloc once,
// replay without reset) was tested and CORRUPTS output deterministically
// on this ggml pin: its CUDA-graph reuse machinery keys captures on the
// graph's first-node pointer + uid, which is safe under llama.cpp's
// rebuild-alloc-compute pattern but produces stale/crossed replays with
// persistent alloc-once graphs that coexist with the talker's rebuilt
// graphs. Do not reintroduce without reworking the capture keying.

static bool cp_exec_run_step(CodePredictorExec * ex, CPExecStep * st, const float * input) {
    ggml_backend_sched_reset(ex->sched);
    if (!ggml_backend_sched_alloc_graph(ex->sched, st->gf)) {
        fprintf(stderr, "[CodePredictor] FATAL: graph alloc failed (g=%d)\n", st->g_head);
        ggml_backend_sched_reset(ex->sched);
        return false;
    }

    ggml_backend_tensor_set(st->x_in, input, 0,
                            (size_t) st->T * (size_t) ex->talker_hidden * sizeof(float));
    ggml_backend_tensor_set(st->pos_in, ex->positions[(size_t) st->g_head].data(), 0,
                            (size_t) st->T * sizeof(int32_t));
    ggml_backend_tensor_set(st->mask_in, ex->masks[(size_t) st->g_head].data(), 0,
                            ex->masks[(size_t) st->g_head].size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ex->sched, st->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[CodePredictor] FATAL: graph compute failed (g=%d)\n", st->g_head);
        ggml_backend_sched_reset(ex->sched);
        return false;
    }

    const size_t row_bytes = (size_t) ex->vocab * sizeof(float);
    ggml_backend_tensor_get(ggml_graph_get_tensor(st->gf, "logits"), ex->logits.data(),
                            (size_t)(st->T - 1) * row_bytes, row_bytes);

    ex->kv->cur_len = st->n_past + st->T;
    ggml_backend_sched_reset(ex->sched);
    return true;
}

// Run the predictor for one audio frame over the pre-built graphs.
// Inputs and outputs match code_predictor_step exactly; sampling consumes
// the same Philox subsequences (subseq_base + 1 + g), so the produced
// codes are bit identical to the legacy path.
static bool code_predictor_exec_frame(CodePredictorExec *          ex,
                                      const TalkerWeights *        tw,
                                      const float *                talker_hidden_last,
                                      int                          c0,
                                      float                        temperature,
                                      int                          top_k,
                                      float                        top_p,
                                      int64_t                      seed,
                                      int64_t                      subseq_base,
                                      const char *                 dump_dir,
                                      CodePredictorOutput *        out) {
    const int talker_hidden = ex->talker_hidden;
    const int n_acoustic    = ex->n_acoustic;

    if (n_acoustic + 1 > ex->kv->max_seq_len) {
        fprintf(stderr, "[CodePredictor] FATAL: frame width %d exceeds cache max_seq_len %d\n", n_acoustic + 1,
                ex->kv->max_seq_len);
        return false;
    }

    out->codes.assign((size_t)(n_acoustic + 1), 0);
    out->codes[0] = c0;

    // Prefill payload: talker hidden at slot 0, embed(c0) at slot 1.
    kv_cache_reset(ex->kv);
    std::memcpy(ex->prefill_input.data(), talker_hidden_last, (size_t) talker_hidden * sizeof(float));
    embed_row_from_backend(tw->codec_embedding, c0, talker_hidden,
                           ex->prefill_input.data() + (size_t) talker_hidden);

    for (int g = 0; g < n_acoustic; g++) {
        const float * input_ptr;
        if (g == 0) {
            input_ptr = ex->prefill_input.data();
        } else {
            embed_row_from_backend(ex->cw->codec_embedding[(size_t)(g - 1)], out->codes[(size_t) g],
                                   talker_hidden, ex->step_input.data());
            input_ptr = ex->step_input.data();
        }
        if (!cp_exec_run_step(ex, &ex->steps[(size_t) g], input_ptr)) {
            return false;
        }

        float u_g = 0.0f;
        int   cg  = sample_top_k_p(ex->logits.data(), ex->vocab, temperature, top_k, top_p, 1.0f, nullptr, 0,
                                   seed, subseq_base + 1 + g, &u_g);
        if (subseq_base + 1 + g < 32) {
            fprintf(stderr, "[Sample-CP] g=%d c=%d u=%.10f subseq=%lld\n", g, cg, (double) u_g,
                    (long long)(subseq_base + 1 + g));
        }
        if (cg < 0) {
            fprintf(stderr, "[CodePredictor] FATAL: sample returned no candidate at g=%d\n", g);
            return false;
        }
        out->codes[(size_t)(g + 1)] = cg;
    }

    if (dump_dir) {
        DebugDumper d;
        debug_init(&d, dump_dir);
        std::vector<int32_t> codes32(out->codes.begin(), out->codes.end());
        int                  n = (int) codes32.size();
        debug_dump_i32_as_f32(&d, "codes-step0", codes32.data(), &n, 1);
    }
    return true;
}
