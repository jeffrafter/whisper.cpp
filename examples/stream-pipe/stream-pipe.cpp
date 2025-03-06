#include "common.h"
#include "whisper.h"
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>

#define SAMPLE_RATE 16000
#define BUFFER_DURATION_SEC 3
#define BUFFER_SIZE (SAMPLE_RATE * BUFFER_DURATION_SEC)

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

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
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

bool read_pcm_from_stdin(std::vector<float> &buffer, size_t num_samples) {
    std::vector<int16_t> temp_buffer(num_samples);
    size_t bytes_needed = num_samples * sizeof(int16_t);
    size_t bytes_read = fread(temp_buffer.data(), 1, bytes_needed, stdin);

    if (bytes_read == 0) return false;

    for (size_t i = 0; i < bytes_read / sizeof(int16_t); i++) {
        buffer.push_back(temp_buffer[i] / 32768.0f);
    }

    return true;
}

int main(int argc, char ** argv) {
    whisper_params params;

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
    std::vector<std::pair<int, std::string>> accumulated_tokens;
    int last_truncated_position = 0; // Store the last truncation point
    std::string accumulated_text; // Store all transcriptions for the current segment

    wav_writer wavWriter;
    if (params.save_audio) {
        wavWriter.open("output.wav", SAMPLE_RATE, 16, 1);
    }

    printf("[Listening...]\n");
    fflush(stdout);

    // Initialize global indexes:
    size_t total_index = 0;
    size_t offset_start_index = 0;
    int speech_counter = 0;
    bool was_speaking = false;

    while (true) {
        std::vector<float> new_audio;
        if (!read_pcm_from_stdin(new_audio, SAMPLE_RATE / 2)) {
            break;
        }

        size_t new_samples = new_audio.size();

        if (params.save_audio) {
            wavWriter.write(new_audio.data(), new_audio.size());
        }

        if (new_audio.size() < SAMPLE_RATE / 2) {
            new_audio.resize(SAMPLE_RATE / 2, 0.0f);
        }

        // Check if appending new_audio will fit into audio_buffer (which has fixed capacity BUFFER_SIZE)
        printf("\nCurrent audio buffer size: %zu. New samples: %zu\n", audio_buffer.size(), new_samples);
        if (total_index + new_samples <= BUFFER_SIZE) {
            // There's enough room: copy new samples at the end.
            std::copy(new_audio.begin(), new_audio.end(), audio_buffer.begin() + total_index);
        } else {
            // Calculate current valid samples and how many need to be removed.
            size_t current_buffer_samples = total_index - offset_start_index;
            size_t samples_to_remove = (current_buffer_samples + new_samples) - BUFFER_SIZE;
            size_t remaining_samples = current_buffer_samples - samples_to_remove;

            // Shift the valid samples left by samples_to_remove positions.
            // From: starting at audio_buffer.begin() + samples_to_remove,
            // to:   audio_buffer.begin() + samples_to_remove + remaining_samples,
            // Copy them to the beginning of the buffer.
            std::copy(audio_buffer.begin() + samples_to_remove,
                    audio_buffer.begin() + samples_to_remove + remaining_samples,
                    audio_buffer.begin());

            // Now copy the new samples into the freed space at the end.
            std::copy(new_audio.begin(), new_audio.end(), audio_buffer.begin() + remaining_samples);

            offset_start_index += samples_to_remove;
        }

        // Update the total_index (global count) by the number of new samples read.
        total_index += new_samples;

        int total_ms = ((int)total_index * 1000) / SAMPLE_RATE;
        int offset_ms = ((int)offset_start_index * 1000) / SAMPLE_RATE;

        bool is_speaking = !::vad_simple(new_audio, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false);
        if (!is_speaking || !was_speaking) {
            speech_counter++;
            printf("\n[New Speech Segment %d]\n\n", speech_counter);
            fflush(stdout);

            // Reset the buffer and indices for a new speech segment
            audio_buffer.assign(BUFFER_SIZE, 0.0f);
            // Copy new_audio into the beginning of the buffer
            std::copy(new_audio.begin(), new_audio.end(), audio_buffer.begin());
            total_index = new_audio.size();
            offset_start_index = 0;

            accumulated_text.clear();
            accumulated_tokens.clear();
            last_truncated_position = 0;
        }
        was_speaking = is_speaking;

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_progress   = false;
        wparams.print_special    = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.language         = "en";
        wparams.n_threads        = params.n_threads;
        wparams.audio_ctx        = 0;
        wparams.offset_ms        = offset_ms;

        if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) != 0) {
            fprintf(stderr, "Failed to process audio\n");
            return 2;
        }

        printf("\n%d / %d: ", total_ms, offset_ms);
        // printf("\33[2K\r");
        std::string speaker = "";
        std::vector<std::pair<int, std::string>> new_tokens;
        for (int i = 0; i < whisper_full_n_segments(ctx); ++i) {
            // printf("%s ", whisper_full_get_segment_text(ctx, i));

            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                if (id >= whisper_token_eot(ctx)) {
                    continue;
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p   (ctx, i, j);

                const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size()))));

                // printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
                new_tokens.emplace_back(id, text);
            }
        }

        if (!new_tokens.empty()) {
            int first_new_token_id = new_tokens.front().first;
            // Start looking at or after the last truncated position
            auto it = std::find_if(accumulated_tokens.begin() + last_truncated_position, accumulated_tokens.end(),
                                [&](const std::pair<int, std::string>& token) {
                                    return token.first == first_new_token_id;
                                });

            // If we find a match, truncate the accumulated tokens list at that point
            if (it != accumulated_tokens.end()) {
                int index = std::distance(accumulated_tokens.begin(), it);
                last_truncated_position = index;
                accumulated_tokens.erase(it, accumulated_tokens.end());
            }

            // Append new tokens
            accumulated_tokens.insert(accumulated_tokens.end(), new_tokens.begin(), new_tokens.end());
        }

        for (const auto& token : accumulated_tokens) {
            printf("%s", token.second.c_str());
        }
        fflush(stdout);
    }

    whisper_free(ctx);
    return 0;
}