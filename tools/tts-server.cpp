// tts-server.cpp: OpenAI-compatible HTTP server backed by the qwentts
// ABI. Loads a talker + codec once, GPU resident, and serves synthesis over
// POST /v1/audio/speech. The shared core lives in src/tts-server.h ; this
// file only wires the qt_* ABI into the generic adapter.

#include "tts-server.h"

#include "qwen.h"
#include "version.h"
#include "webui_index.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "qwentts.cpp %s\n\n", QWEN_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n\n"
            "Optional:\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n"
            "  --lang <name>           Language label (default: auto)\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --chunk-sec <f>         Max streamed audio chunk in seconds (default and cap: 24.0)\n"
            "  --first-chunk-frames <n> Progressive start: first chunk after n codec frames,\n"
            "                          growing x4 up to --chunk-sec (default: GPU 2, CPU 0 = off)\n"
            "  --left-context-sec <f>  Codec left context per chunk in seconds (default: GPU 6.0, CPU 2.0)\n"
            "  --no-warmup             Skip the warmup synthesis before listening\n"
            "  --threads <n>           CPU threads for ggml (default: physical cores)\n"
            "  --split-chars <n>       Texts longer than n bytes are synthesised as pieces of\n"
            "                          whole sentences (<= ~n/2 bytes each) in one stream\n"
            "                          (default: 800, 0 = never split)\n",
            prog);
}

// Split a long text into pieces of whole sentences of at most ~target
// bytes. Sentences end at . ! ? (followed by whitespace or the end, so
// "3.5" stays whole), at the UTF-8 marks … 。 ！ ？, or at a newline. A
// sentence longer than target is cut at the last ", " "; " ": " or space
// before target. Pieces are trimmed ; empty ones are dropped.
static std::vector<std::string> split_text(const std::string & s, size_t target) {
    auto trim = [](const std::string & x) {
        size_t a = x.find_first_not_of(" \t\r\n");
        size_t b = x.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? std::string() : x.substr(a, b - a + 1);
    };
    std::vector<std::string> sentences;
    std::string              cur;
    for (size_t i = 0; i < s.size(); i++) {
        cur += s[i];
        bool end = false;
        const char c = s[i];
        if (c == '\n') {
            end = true;
        } else if ((c == '.' || c == '!' || c == '?') &&
                   (i + 1 == s.size() || s[i + 1] == ' ' || s[i + 1] == '\n' || s[i + 1] == '\t')) {
            end = true;
        } else {
            static const char * marks[] = { "\xE2\x80\xA6", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F" };
            for (const char * m : marks) {
                const size_t n = std::strlen(m);
                if (cur.size() >= n && cur.compare(cur.size() - n, n, m) == 0) {
                    end = true;
                }
            }
        }
        if (end) {
            sentences.push_back(cur);
            cur.clear();
        }
    }
    sentences.push_back(cur);

    std::vector<std::string> pieces;
    std::string              acc;
    auto flush = [&] {
        std::string t = trim(acc);
        if (!t.empty()) {
            pieces.push_back(t);
        }
        acc.clear();
    };
    for (std::string sent : sentences) {
        while (sent.size() > target) {
            size_t cut = std::string::npos;
            for (const char * sep : { ", ", "; ", ": " }) {
                size_t k = sent.rfind(sep, target);
                if (k != std::string::npos && k > target / 3 && (cut == std::string::npos || k > cut)) {
                    cut = k + 1;
                }
            }
            if (cut == std::string::npos) {
                size_t k = sent.rfind(' ', target);
                cut      = (k != std::string::npos && k > 0) ? k : target;
            }
            flush();
            acc = sent.substr(0, cut);
            flush();
            sent = sent.substr(cut);
        }
        if (acc.size() + sent.size() > target) {
            flush();
        }
        acc += sent;
    }
    flush();
    return pieces;
}

// Joins between the pieces of a split text. Each piece is a whole
// synthesis : it ends with the silence the model leaves after EOS and the
// next one starts with its own lead-in silence, so a join measured 1.0-2.0 s
// of silence against 0.4-0.8 s between sentences inside a piece, which made
// the joins audible. This filter sits between the pipeline and the client
// sink and keeps at most trail_keep of silence at the end of a piece and
// lead_keep at the start of the next one. Silence = 10 ms blocks under
// -44 dBFS RMS ; cuts happen inside silence, with a 5 ms fade, so they
// cannot click. Silence inside a piece is held back, not dropped : it is
// released as soon as speech resumes.
struct join_trimmer {
    static constexpr int   kBlock = 240;     // 10 ms at 24 kHz
    static constexpr float kThr   = 0.006f;  // RMS, about -44 dBFS
    static constexpr int   kFade  = 120;     // 5 ms

    const tts_sink *   out        = nullptr;
    int                lead_keep  = 0;
    int                trail_keep = 0;
    bool               leading    = false;  // dropping lead-in silence
    bool               hold_trail = false;  // holding silence for a trailing cut
    bool               spoke      = false;  // this piece already produced speech
    std::vector<float> pend;                 // incomplete block
    std::vector<float> lead;                 // last lead_keep silent samples
    std::vector<float> held;                 // silence since the last speech

    bool emit(const float * s, int n) { return n <= 0 || (*out)(s, n); }

    void begin_piece(bool first, bool last) {
        leading    = !first;
        hold_trail = !last;
        spoke      = false;
        pend.clear();
        lead.clear();
        held.clear();
    }

    bool block(const float * b, int n) {
        double e = 0.0;
        for (int i = 0; i < n; i++) {
            e += (double) b[i] * b[i];
        }
        const bool silent = n > 0 && std::sqrt(e / n) < kThr;
        if (leading) {
            if (silent) {
                lead.insert(lead.end(), b, b + n);
                if ((int) lead.size() > lead_keep) {
                    lead.erase(lead.begin(), lead.end() - lead_keep);
                }
                return true;
            }
            leading = false;
            for (int i = 0; i < kFade && i < (int) lead.size(); i++) {
                lead[(size_t) i] *= (float) i / kFade;
            }
            if (!emit(lead.data(), (int) lead.size())) {
                return false;
            }
            lead.clear();
        }
        // Silence before the first speech of a piece passes straight through
        // (the first piece's lead-in is the time to first audio).
        if (hold_trail && spoke) {
            if (silent) {
                held.insert(held.end(), b, b + n);
                return true;
            }
            if (!emit(held.data(), (int) held.size())) {
                return false;
            }
            held.clear();
        }
        spoke = spoke || !silent;
        return emit(b, n);
    }

    bool push(const float * s, int n) {
        pend.insert(pend.end(), s, s + n);
        size_t off = 0;
        for (; off + kBlock <= pend.size(); off += kBlock) {
            if (!block(pend.data() + off, kBlock)) {
                return false;
            }
        }
        pend.erase(pend.begin(), pend.begin() + (std::ptrdiff_t) off);
        return true;
    }

    bool end_piece() {
        if (!pend.empty() && !block(pend.data(), (int) pend.size())) {
            return false;
        }
        pend.clear();
        if (hold_trail && !held.empty()) {
            const int keep = (int) held.size() < trail_keep ? (int) held.size() : trail_keep;
            for (int i = 0; i < kFade && i < keep; i++) {
                held[(size_t) (keep - 1 - i)] *= (float) i / kFade;
            }
            if (!emit(held.data(), keep)) {
                return false;
            }
        }
        held.clear();
        return true;
    }
};

// Trim a path down to its file name for the reported model id.
static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

int main(int argc, char ** argv) {
    const char *  talker_path = NULL;
    const char *  codec_path  = NULL;
    std::string   lang        = "auto";
    server_config cfg;
    bool          use_fa     = true;
    bool          clamp_fp16 = false;
    float         chunk_sec  = 24.0f;
    int           first_chunk_frames = -1;  // -1 : backend default (GPU 2, CPU 0)
    bool          warmup             = true;
    int           split_chars        = 800;
    float         left_ctx_sec       = -1.0f;  // < 0 : backend default (GPU 6.0, CPU 2.0)

    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            talker_path = argv[++i];
        } else if (!std::strcmp(arg, "--codec") && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (!std::strcmp(arg, "--host") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if (!std::strcmp(arg, "--port") && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--lang") && i + 1 < argc) {
            lang = argv[++i];
        } else if (!std::strcmp(arg, "--no-fa")) {
            use_fa = false;
        } else if (!std::strcmp(arg, "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!std::strcmp(arg, "--chunk-sec") && i + 1 < argc) {
            chunk_sec = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--first-chunk-frames") && i + 1 < argc) {
            first_chunk_frames = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--threads") && i + 1 < argc) {
#ifdef _WIN32
            _putenv_s("QWEN_N_THREADS", argv[++i]);
#else
            setenv("QWEN_N_THREADS", argv[++i], 1);
#endif
        } else if (!std::strcmp(arg, "--split-chars") && i + 1 < argc) {
            split_chars = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--no-warmup")) {
            warmup = false;
        } else if (!std::strcmp(arg, "--left-context-sec") && i + 1 < argc) {
            left_ctx_sec = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!talker_path || !codec_path) {
        print_usage(argv[0]);
        return 0;
    }

    struct qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path = talker_path;
    iparams.codec_path  = codec_path;
    iparams.use_fa      = use_fa;
    iparams.clamp_fp16  = clamp_fp16;

    struct qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[Server] FATAL: %s\n", qt_last_error());
        return 1;
    }

    // Backend dependent stream framing. GPU : progressive start (first
    // audio after 2 frames) with 6 s of codec left context, which covers
    // the codec transformer window (72 frames = 5.76 s) and keeps the
    // chunked audio close to a one block decode. CPU : the codec decode
    // dominates (~60 ms/frame) and RTF is already > 1, so the extra
    // decodes would only slow it down ; keep the upstream framing.
    const bool gpu = qt_has_gpu(q) != 0;
    if (first_chunk_frames < 0) {
        first_chunk_frames = gpu ? 2 : 0;
    }
    if (left_ctx_sec < 0.0f) {
        left_ctx_sec = gpu ? 6.0f : 2.0f;
    }
    fprintf(stderr, "[Server] stream: %s, first chunk %d frames, chunk cap %.1f s, left context %.1f s\n",
            gpu ? "GPU" : "CPU", first_chunk_frames, chunk_sec, left_ctx_sec);

    tts_backend be;
    be.model_id = basename_of(talker_path);
    int n       = qt_n_speakers(q);
    for (int i = 0; i < n; i++) {
        be.voices.push_back(qt_speaker_name(q, i));
    }
    for (int i = 0; i < qt_n_languages(q); i++) {
        be.languages.push_back(qt_language_name(q, i));
    }
    be.backend = gpu ? "GPU" : "CPU";
    be.version = QWEN_VERSION;
    be.ui_html.assign((const char *) webui_index_html, webui_index_html_len);

    // The adapter always drives the streaming pipeline : on_chunk routes to
    // the shared sink, which either streams to the socket (pcm) or fills a
    // one-shot buffer (wav). Either way the audio path is identical.
    be.synthesize = [q, &lang, chunk_sec, first_chunk_frames, left_ctx_sec, split_chars](
                        const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        // Long texts : the model drifts (no EOS, babble) past ~1000 chars in
        // one prompt, so synthesise whole-sentence pieces back to back on
        // the same stream, same seed. Short texts stay one piece.
        std::vector<std::string> pieces;
        if (split_chars > 0 && req.input.size() > (size_t) split_chars) {
            // Pieces of at most split_chars / 2 bytes. Longer pieces (3/4 of
            // split_chars : 3 instead of 4 for 1170 chars) skipped text on 2
            // of 6 seeds (ASR recall 0.73 / 0.82), so the pieces stay short
            // and join_trimmer smooths the joins instead.
            pieces = split_text(req.input, (size_t) split_chars / 2);
            fprintf(stderr, "[Server] long text (%zu bytes): %zu pieces\n", req.input.size(), pieces.size());
        } else {
            pieces.push_back(req.input);
        }

        struct qt_tts_params p;
        qt_tts_default_params(&p);
        p.lang = req.language.empty() ? lang.c_str() : req.language.c_str();
        // Capped at 24 s (the upstream default): one codec decode over a
        // longer block reserves GBs of VRAM and ggml aborts on OOM (a
        // 62 s block asked for 4 GB on the 0.6B and killed the server).
        const float req_chunk      = req.chunk_sec > 0.0f ? req.chunk_sec : chunk_sec;
        p.codec_chunk_sec          = req_chunk < 24.0f ? req_chunk : 24.0f;
        p.codec_left_context_sec   = left_ctx_sec;
        p.codec_first_chunk_frames = req.first_chunk_frames >= 0 ? req.first_chunk_frames : first_chunk_frames;
        if (!req.voice.empty() && qt_n_speakers(q) > 0) {
            p.speaker = req.voice.c_str();
        }
        if (!req.instructions.empty()) {
            p.instruct = req.instructions.c_str();
        }
        p.seed = req.seed;

        // Trampoline : the C ABI on_chunk forwards to the C++ sink.
        const tts_sink * sink_ptr = &sink;
        p.on_chunk                = [](const float * s, int ns, void * u) -> bool {
            return (*static_cast<const tts_sink *>(u))(s, ns);
        };
        p.on_chunk_user_data = (void *) sink_ptr;

        // Several pieces : route the audio through the join trimmer (about
        // 0.30 s kept at the end of a piece + 0.15 s at the start of the
        // next, like a pause between sentences). One piece : untouched.
        join_trimmer trim;
        trim.out        = &sink;
        trim.trail_keep = 24000 * 30 / 100;
        trim.lead_keep  = 24000 * 15 / 100;
        if (pieces.size() > 1) {
            p.on_chunk = [](const float * s, int ns, void * u) -> bool {
                return static_cast<join_trimmer *>(u)->push(s, ns);
            };
            p.on_chunk_user_data = (void *) &trim;
        }
        for (size_t k = 0; k < pieces.size(); k++) {
            p.text = pieces[k].c_str();
            trim.begin_piece(k == 0, k + 1 == pieces.size());
            struct qt_audio out = {};
            enum qt_status  rc  = qt_synthesize(q, &p, &out);
            qt_audio_free(&out);
            if (rc == QT_STATUS_OK && pieces.size() > 1 && !trim.end_piece()) {
                rc = QT_STATUS_CANCELLED;
            }
            if (rc != QT_STATUS_OK) {
                err = qt_last_error();
                return (int) rc;
            }
        }
        return 0;
    };

    // Warmup before listening : the first synthesis of the process pays
    // CUDA/cuBLAS init, graph allocation and CUDA graph capture. Running a
    // short throwaway synthesis here keeps that off the first client
    // request. Failure only warns (e.g. a Base model needs a reference).
    if (warmup) {
        tts_request w;
        w.input = "Hola.";
        w.seed  = 0;
        if (!be.voices.empty()) {
            w.voice = be.voices[0];
        } else {
            w.instructions = "A calm, clear voice.";
        }
        tts_sink    discard = [](const float *, int) { return true; };
        std::string werr;
        const auto  t0 = std::chrono::steady_clock::now();
        const int   wrc = be.synthesize(w, discard, werr);
        const long  ms  = (long) std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        if (wrc == 0) {
            fprintf(stderr, "[Server] warmup done in %ld ms\n", ms);
        } else {
            fprintf(stderr, "[Server] WARN: warmup failed (%d): %s\n", wrc, werr.c_str());
        }
    }

    int rc = tts_server_run(be, cfg);
    qt_free(q);
    return rc;
}
