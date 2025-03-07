#include "common.h"
#include "whisper.h"
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>

#define BUFFER_DURATION_SEC 10
#define BUFFER_SIZE (WHISPER_SAMPLE_RATE * BUFFER_DURATION_SEC)

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.2f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = false;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
};

struct StreamToken {
    whisper_token_data data;
    std::string text;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}

static char * escape_double_quotes_and_backslashes(const char * str) {
    if (str == NULL) {
        return NULL;
    }

    size_t escaped_length = strlen(str) + 1;

    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"' || str[i] == '\\') {
            escaped_length++;
        }
    }

    char * escaped = (char *)calloc(escaped_length, 1); // pre-zeroed
    if (escaped == NULL) {
        return NULL;
    }

    size_t pos = 0;
    for (size_t i = 0; str[i] != '\0'; i++) {
        if (str[i] == '"' || str[i] == '\\') {
            escaped[pos++] = '\\';
        }
        escaped[pos++] = str[i];
    }

    // no need to set zero due to calloc() being used prior

    return escaped;
}

static bool output_json(
            struct whisper_context * ctx,
            const whisper_params & params,
            const int speech_counter,
            std::vector<StreamToken> accumulated_tokens,
            bool full) {
    int indent = 0;

    auto doindent = [&]() {
        // No indent
        // for (int i = 0; i < indent; i++) std::cout << "\t";
    };

    auto donewline = [&]() {
        // No indent
        // std::cout << "\n";
    };

    auto start_arr = [&](const char *name) {
        doindent();
        std::cout << "\"" << name << "\": [";
        donewline();
        indent++;
    };

    auto end_arr = [&](bool end) {
        indent--;
        doindent();
        std::cout << (end ? "]" : "],");
        donewline();
    };

    auto start_obj = [&](const char *name) {
        doindent();
        if (name) {
            std::cout << "\"" << name << "\": {";
        } else {
            std::cout << "{";
        }
        donewline();
        indent++;
    };

    auto end_obj = [&](bool end) {
        indent--;
        doindent();
        std::cout << (end ? "}" : "},");
        donewline();
    };

    auto start_value = [&](const char *name) {
        doindent();
        std::cout << "\"" << name << "\": ";
    };

    auto value_s = [&](const char *name, const char *val, bool end) {
        start_value(name);
        char * val_escaped = escape_double_quotes_and_backslashes(val);
        std::cout << "\"" << val_escaped << (end ? "\"" : "\",");
        donewline();
        free(val_escaped);
    };

    auto end_value = [&](bool end) {
        std::cout << (end ? "" : ",");
        donewline();
    };

    auto value_i = [&](const char *name, const int64_t val, bool end) {
        start_value(name);
        std::cout << val;
        end_value(end);
    };

    auto value_f = [&](const char *name, const float val, bool end) {
        start_value(name);
        std::cout << val;
        end_value(end);
    };

    auto value_b = [&](const char *name, const bool val, bool end) {
        start_value(name);
        std::cout << (val ? "true" : "false");
        end_value(end);
    };

    auto times_o = [&](int64_t t0, int64_t t1, bool end) {
        start_obj("timestamps");
        value_s("from", to_timestamp(t0, false).c_str(), false);
        value_s("to", to_timestamp(t1, false).c_str(), true);
        end_obj(false);
        start_obj("offsets");
        value_i("from", t0 * 10, false);
        value_i("to", t1 * 10, true);
        end_obj(end);
    };

    start_obj(nullptr);
        start_arr("transcription");
            // Get all of the accumulated tokens and into a single text string
            std::string full_output;
            for (const auto& token : accumulated_tokens) {
                full_output += token.text;
            }
            start_obj(nullptr);
                // times_o(t0, t1, false);
                value_i("segment", speech_counter, false);
                value_s("text", full_output.c_str(), false);

                if (full) {
                    start_arr("tokens");
                    for (const auto& token : accumulated_tokens) {
                        start_obj(nullptr);
                            value_s("text", token.text.c_str(), false);
                            if(token.data.t0 > -1 && token.data.t1 > -1) {
                                // If we have per-token timestamps, write them out
                                times_o(token.data.t0, token.data.t1, false);
                            }
                            value_i("id", token.data.id, false);
                            value_i("tid", token.data.tid, false);
                            value_f("p", token.data.p, false);
                            value_f("t_dtw", token.data.t_dtw, false);
                            value_i("vlen", token.data.vlen, true);
                        bool is_end = &token == &accumulated_tokens.back();
                        end_obj(is_end);
                    }
                    end_arr(true);
                }

                // TODO diarization
                // if (params.diarize && pcmf32s.size() == 2) {
                //     value_s("speaker", estimate_diarization_speaker(pcmf32s, t0, t1, true).c_str(), true);
                // }

                // if (params.tinydiarize) {
                //     value_b("speaker_turn_next", whisper_full_get_segment_speaker_turn_next(ctx, i), true);
                // }
            end_obj(true);
        end_arr(true);
    end_obj(true);
    return true;
}

bool read_pcm_from_stdin(std::vector<float> &buffer, size_t num_mono_samples) {
    // For stereo input, we expect 2 * num_mono_samples int16 values.
    size_t num_samples_expected = num_mono_samples * 2;
    std::vector<int16_t> temp_buffer(num_samples_expected);
    size_t bytes_needed = num_samples_expected * sizeof(int16_t);
    size_t bytes_read = fread(temp_buffer.data(), 1, bytes_needed, stdin);
    if (bytes_read == 0) return false;

    size_t samples_read = bytes_read / sizeof(int16_t);
    if (samples_read % 2 != 0) {
        // If we got an odd number of samples, drop the last sample.
        samples_read -= 1;
    }

    // Process every pair as one mono sample.
    for (size_t i = 0; i < samples_read; i += 2) {
        int16_t left = temp_buffer[i];
        int16_t right = temp_buffer[i + 1];
        int16_t mono = (left + right) / 2;
        buffer.push_back(mono / 32768.0f);
    }
    return true;
}

int main(int argc, char ** argv) {
    whisper_params params;

    std::cout << "Whisper sample rate: " << WHISPER_SAMPLE_RATE << std::endl;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    std::vector<float> audio_buffer(BUFFER_SIZE, 0.0f);
    std::vector<StreamToken> accumulated_tokens;
    std::string accumulated_text; // Store all transcriptions for the current segment

    wav_writer wavWriter;
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }

    fflush(stdout);

    // Initialize global indexes:
    size_t total_index = 0;
    int speech_counter = 0;
    bool was_speaking = false;

    while (true) {
        std::vector<float> new_audio;
        if (!read_pcm_from_stdin(new_audio, WHISPER_SAMPLE_RATE / 2)) {
            break;
        }

        size_t new_samples = new_audio.size();

        if (params.save_audio) {
            wavWriter.write(new_audio.data(), new_audio.size());
        }

        // Build me up buttercup
        if (new_samples < WHISPER_SAMPLE_RATE / 2) {
            continue;
        }

        // Update the total_index (global count) by the number of new samples read.
        // printf("\n%i/%i %i: ", speech_counter, total_index, new_samples);
        audio_buffer.resize(BUFFER_SIZE + total_index + new_samples);
        std::copy(new_audio.begin(), new_audio.end(), audio_buffer.begin() + total_index);
        total_index += new_samples;

        bool is_speaking = !::vad_simple(new_audio, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false);
        if (!is_speaking || !was_speaking) {
            speech_counter++;
            printf("\n[New Speech Segment %d]", speech_counter);
            fflush(stdout);

            // Reset the buffer and indices for a new speech segment
            total_index = 0;
            audio_buffer.clear();
            audio_buffer.resize(BUFFER_SIZE + total_index + new_samples);
            std::copy(new_audio.begin(), new_audio.end(), audio_buffer.begin() + total_index);
            total_index += new_samples;

            accumulated_text.clear();
            accumulated_tokens.clear();
        }
        int total_ms = ((int)total_index * 1000) / WHISPER_SAMPLE_RATE;

        was_speaking = is_speaking;

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_progress   = false;
        wparams.print_special    = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.language         = "en";
        wparams.n_threads        = params.n_threads;
        wparams.audio_ctx        = 0;
        wparams.token_timestamps = true;
        wparams.suppress_nst     = true;

        if (audio_buffer.size() < WHISPER_SAMPLE_RATE / 2) {
            audio_buffer.resize(WHISPER_SAMPLE_RATE / 2, 0.0f);
        }

        if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) != 0) {
            fprintf(stderr, "Failed to process audio\n");
            return 2;
        }

        // printf("\33[2K\r");
        std::string speaker = "";
        std::vector<StreamToken> new_tokens;
        for (int i = 0; i < whisper_full_n_segments(ctx); ++i) {
            // printf("%s ", whisper_full_get_segment_text(ctx, i));

            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                if (id >= whisper_token_eot(ctx)) {
                    continue;
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p   (ctx, i, j);
                const whisper_token_data data = whisper_full_get_token_data(ctx, i, j);
                const StreamToken token = {data, text};

                const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size()))));

                // printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
                // printf("%s (%d/%d) ", text, data.id, id);
                new_tokens.push_back(token);
            }
        }

        accumulated_tokens = new_tokens;

        printf("\n");
        output_json(ctx, params, speech_counter, accumulated_tokens, true);
        // printf("\n%i: ", speech_counter);
        // for (const auto& token : accumulated_tokens) {
        //     printf("%s", token.text.c_str());
        // }
        fflush(stdout);
    }

    whisper_free(ctx);
    return 0;
}