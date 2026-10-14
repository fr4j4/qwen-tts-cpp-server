#pragma once
// codec-chunked-decode.h: bounded VRAM codec decode with rolling left
// context. Strict equivalent of the upstream Qwen3-TTS 12 Hz tokenizer
// chunked_decode entry
// (qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
// line 886).
//
// The codec decoder runs a causal Conv1d, a sliding window causal
// transformer, an upsample stage and a DAC decoder. Decoding a chunk
// of frames in isolation introduces edge artefacts at the chunk
// boundary because the causal conv kernel and the transformer attention
// have no left context to draw from. Prepending left_ctx_frames
// previously decoded frames and stripping the resulting samples after
// the decode restores continuity.
//
// Two entry points:
//
//   codec_chunked_decode : one shot decode of a full codes buffer.
//     Bit perfect equivalent of pipeline_codec_decode when the audio
//     fits in a single chunk_frames sized window. Bounds VRAM beyond
//     that, mirrors the upstream chunked_decode loop frame for frame.
//
//   codec_chunked_decoder_stream : rolling state for AR streaming.
//     The pipeline pushes one frame at a time as the talker produces
//     them ; push_frame decodes and emits a fresh chunk_frames sized
//     audio block through the on_chunk callback as soon as enough new
//     frames have accumulated. flush drains the tail at EOS.

#include "pipeline-codec.h"
#include "qwen.h"

#include <cstdint>
#include <cstring>
#include <vector>

// One shot chunked decode. codes is K major [K, T] row major (T fastest).
// Returns audio of length T * TOKENIZER_HOP_LENGTH on success, empty on
// failure. chunk_frames clamps to 1, left_ctx_frames clamps to 0.
static inline std::vector<float> codec_chunked_decode(PipelineCodec * pc,
                                                      const int32_t * codes,
                                                      int             K,
                                                      int             T,
                                                      int             chunk_frames,
                                                      int             left_ctx_frames) {
    std::vector<float> out;
    if (T <= 0) {
        return out;
    }
    if (chunk_frames < 1) {
        chunk_frames = 1;
    }
    if (left_ctx_frames < 0) {
        left_ctx_frames = 0;
    }
    out.reserve((size_t) T * (size_t) TOKENIZER_HOP_LENGTH);

    int start = 0;
    while (start < T) {
        int end = start + chunk_frames;
        if (end > T) {
            end = T;
        }
        // Upstream rule : context_size collapses to start when
        // left_ctx_frames would underflow before frame 0.
        int ctx         = (start - left_ctx_frames > 0) ? left_ctx_frames : start;
        int slice_start = start - ctx;
        int slice_T     = end - slice_start;

        std::vector<int32_t> slice((size_t) K * (size_t) slice_T);
        for (int k = 0; k < K; k++) {
            std::memcpy(slice.data() + (size_t) k * (size_t) slice_T,
                        codes + (size_t) k * (size_t) T + (size_t) slice_start, (size_t) slice_T * sizeof(int32_t));
        }
        std::vector<float> wav = pipeline_codec_decode(pc, slice.data(), K, slice_T);
        if (wav.empty()) {
            return std::vector<float>();
        }
        const size_t drop = (size_t) ctx * (size_t) TOKENIZER_HOP_LENGTH;
        if (wav.size() > drop) {
            out.insert(out.end(), wav.begin() + drop, wav.end());
        }
        start = end;
    }
    return out;
}

// Rolling streaming decoder. Stores codes K major as K parallel vectors
// (by_k[k][t]) so emit_one can memcpy a contiguous K major slice into
// pipeline_codec_decode without a transpose. push_frame triggers as
// many emits as possible after appending one frame ; flush emits the
// tail at EOS.
struct codec_chunked_decoder_stream {
    std::vector<std::vector<int32_t>> by_k;
    int                               K;
    int                               T_so_far;
    int                               chunk_frames;
    int                               left_ctx_frames;
    int                               emit_start_frame;
    // Size of the next emit. Starts at first_chunk_frames (progressive
    // start: low time to first audio) and grows x4 after each emit up to
    // chunk_frames ; equals chunk_frames when the progressive start is off.
    int                               next_chunk;
    // Set true when an emit returned false because the on_chunk callback
    // requested a cancel. Stays false on decode failures so the caller
    // can route to QT_STATUS_CANCELLED vs QT_STATUS_GENERATE_FAILED on
    // a push_frame / flush negative return.
    bool                              cancelled;

    // first_chunk_frames_ <= 0 disables the progressive start (every emit
    // is chunk_frames long, the upstream framing).
    void init(int K_, int chunk_frames_, int left_ctx_frames_, int first_chunk_frames_ = 0) {
        K                = K_;
        T_so_far         = 0;
        chunk_frames     = chunk_frames_ < 1 ? 1 : chunk_frames_;
        left_ctx_frames  = left_ctx_frames_ < 0 ? 0 : left_ctx_frames_;
        emit_start_frame = 0;
        next_chunk       = (first_chunk_frames_ > 0 && first_chunk_frames_ < chunk_frames) ? first_chunk_frames_
                                                                                            : chunk_frames;
        cancelled        = false;
        by_k.assign((size_t) K, {});
    }

    // Append one frame (K int32 codes, one per codebook). Drain any
    // chunks that became emittable. Returns false on decode failure or
    // when cb returns false (cancellation).
    bool push_frame(PipelineCodec * pc, const int32_t * frame_codes, qt_audio_chunk_cb cb, void * cb_ud) {
        for (int k = 0; k < K; k++) {
            by_k[(size_t) k].push_back(frame_codes[k]);
        }
        T_so_far++;

        while (T_so_far - emit_start_frame >= next_chunk) {
            if (!emit_one(pc, emit_start_frame + next_chunk, cb, cb_ud)) {
                return false;
            }
            // Growth x2 on the first step, x4 after (2, 4, 16, 64, 256, cap):
            // each emit re-decodes up to left_ctx_frames of context, so fewer
            // emits = less codec work. The first step stays x2 because the
            // first chunk is tiny (160 ms of audio) and a x4 second chunk
            // would arrive about when it ran out ; past that the audio
            // already sent covers the next chunk with margin while RTF is
            // well under 1/3 (bench_stream.py min_slack_ms).
            const int growth = (emit_start_frame == next_chunk) ? 2 : 4;
            next_chunk       = next_chunk * growth < chunk_frames ? next_chunk * growth : chunk_frames;
        }
        return true;
    }

    // Drain the tail. If frames remain past emit_start_frame, decode
    // them with left context and emit one final short chunk. Idempotent
    // on empty tail.
    bool flush(PipelineCodec * pc, qt_audio_chunk_cb cb, void * cb_ud) {
        if (T_so_far > emit_start_frame) {
            return emit_one(pc, T_so_far, cb, cb_ud);
        }
        return true;
    }

  private:
    // Decode [emit_start_frame - ctx .. end_frame] with left context
    // stripped from the emitted samples, then advance emit_start_frame.
    bool emit_one(PipelineCodec * pc, int end_frame, qt_audio_chunk_cb cb, void * cb_ud) {
        int ctx         = (emit_start_frame - left_ctx_frames > 0) ? left_ctx_frames : emit_start_frame;
        int slice_start = emit_start_frame - ctx;
        int slice_T     = end_frame - slice_start;

        std::vector<int32_t> slice((size_t) K * (size_t) slice_T);
        for (int k = 0; k < K; k++) {
            std::memcpy(slice.data() + (size_t) k * (size_t) slice_T, by_k[(size_t) k].data() + (size_t) slice_start,
                        (size_t) slice_T * sizeof(int32_t));
        }
        std::vector<float> wav = pipeline_codec_decode(pc, slice.data(), K, slice_T);
        if (wav.empty()) {
            return false;
        }
        const size_t  drop       = (size_t) ctx * (size_t) TOKENIZER_HOP_LENGTH;
        const float * emit_first = wav.data() + drop;
        int           emit_n     = (int) (wav.size() - drop);
        if (emit_n > 0 && !cb(emit_first, emit_n, cb_ud)) {
            cancelled = true;
            return false;
        }
        emit_start_frame = end_frame;
        return true;
    }
};
