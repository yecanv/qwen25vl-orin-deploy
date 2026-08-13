// hybrid_driver: TRT ViT 特征 → llama.cpp LLM 的混合端到端驱动
// 用法: hybrid_driver <model.gguf> <features.bin> <n_img_tokens> <grid_nx> <question_utf8>
// 特征文件: fp32 raw, [n_img_tokens x n_embd] 行主序(TRT ViT merger 输出,行主序合并网格)
// 位置语义逐行照抄 mtmd(mtmd-helper-common.h decode_embd_batch + mtmd.cpp MROPE 分支):
//   文本批: 1 pos/token(llama 核心对 mrope 模型自动广播 4 通道)
//   图块批: pos[4*N] 分段块状 块0=t(=pos_0) 块1=y(行) 块2=x(列) 块3=0
//   图后推进: n_past += max(nx, ny)
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0,
                            /*add_special=*/false, /*parse_special=*/true);
    std::vector<llama_token> out(n);
    llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), out.data(), n, false, true);
    return out;
}

// 文本批解码: 标量 pos, 返回 0 成功
static int decode_text(llama_context * ctx, const std::vector<llama_token> & toks,
                       llama_pos pos0, bool logits_last) {
    llama_batch b = llama_batch_init((int32_t)toks.size(), 0, 1);
    b.n_tokens = (int32_t)toks.size();
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i]    = toks[i];
        b.pos[i]      = pos0 + (llama_pos)i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]   = false;
    }
    if (logits_last) b.logits[b.n_tokens - 1] = true;
    int ret = llama_decode(ctx, b);
    llama_batch_free(b);
    return ret;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <features.bin> <n_img_tokens> <grid_nx> <question>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * feat_path  = argv[2];
    const int    n_img      = atoi(argv[3]);
    const int    nx         = atoi(argv[4]);
    const int    ny         = n_img / nx;
    const std::string question = argv[5];

    llama_backend_init();

    double t0 = now_ms();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "FATAL: model load failed\n"); return 2; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 2048;
    cp.n_batch  = 2048;
    cp.n_ubatch = 2048;   // 图块 1024 token 一次进
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FATAL: context init failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd_inp(model);
    double t_load = now_ms() - t0;

    // 读特征
    FILE * f = fopen(feat_path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", feat_path); return 3; }
    std::vector<float> feats((size_t)n_img * n_embd);
    size_t rd = fread(feats.data(), sizeof(float), feats.size(), f);
    fclose(f);
    if (rd != feats.size()) { fprintf(stderr, "FATAL: feature size mismatch %zu != %zu\n", rd, feats.size()); return 3; }

    // 提示词三段
    const std::string part1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
    const std::string part2 = "<|vision_end|>" + question + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> tk1 = tokenize(vocab, part1);
    std::vector<llama_token> tk2 = tokenize(vocab, part2);
    const int A = (int)tk1.size();

    fprintf(stderr, "n_embd=%d n_img=%d grid=%dx%d part1=%d part2=%d\n",
            n_embd, n_img, nx, ny, A, (int)tk2.size());

    // ① 文本段一
    t0 = now_ms();
    if (decode_text(ctx, tk1, 0, false)) { fprintf(stderr, "FATAL: decode part1\n"); return 4; }
    double t_p1 = now_ms() - t0;

    // ② 图块: embd 注入 + mrope 4 块位置(照抄 set_position_mrope_2d)
    t0 = now_ms();
    std::vector<llama_pos>    pos((size_t)n_img * 4);
    std::vector<int32_t>      n_seq_id(n_img, 1);
    llama_seq_id              seq0 = 0;
    std::vector<llama_seq_id*> seq_ids(n_img + 1, nullptr);
    std::vector<int8_t>       logits(n_img, 0);
    for (int i = 0; i < n_img; i++) {
        pos[i            ] = A;                 // t: 常量 = pos_0
        pos[i + n_img    ] = A + i / nx;        // y: 行
        pos[i + n_img * 2] = A + i % nx;        // x: 列
        pos[i + n_img * 3] = 0;                 // z: 未用
        seq_ids[i] = &seq0;
    }
    llama_batch ib = {
        /*n_tokens=*/ n_img,
        /*token   =*/ nullptr,
        /*embd    =*/ feats.data(),
        /*pos     =*/ pos.data(),
        /*n_seq_id=*/ n_seq_id.data(),
        /*seq_id  =*/ seq_ids.data(),
        /*logits  =*/ logits.data(),
    };
    if (llama_decode(ctx, ib)) { fprintf(stderr, "FATAL: decode image embd\n"); return 5; }
    double t_img = now_ms() - t0;

    // ③ 文本段二: 位置从 A + max(nx,ny) 续(照抄 mtmd_image_tokens_get_n_pos MROPE)
    const llama_pos st2 = A + std::max(nx, ny);
    t0 = now_ms();
    if (decode_text(ctx, tk2, st2, true)) { fprintf(stderr, "FATAL: decode part2\n"); return 6; }
    double t_p2 = now_ms() - t0;

    // ④ 贪心生成
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    std::string answer;
    llama_pos cur = st2 + (llama_pos)tk2.size();
    int n_gen = 0;
    t0 = now_ms();
    for (int step = 0; step < 256; step++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char buf[256];
        int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (len > 0) { answer.append(buf, len); fwrite(buf, 1, len, stdout); fflush(stdout); }
        n_gen++;
        std::vector<llama_token> one{tok};
        if (decode_text(ctx, one, cur, true)) { fprintf(stderr, "FATAL: decode gen\n"); return 7; }
        cur++;
    }
    double t_gen = now_ms() - t0;
    printf("\n");

    fprintf(stderr, "TIMING_JSON {\"load_ms\":%.0f,\"prefill_part1_ms\":%.0f,\"prefill_img1024_ms\":%.0f,"
            "\"prefill_part2_ms\":%.0f,\"gen_tokens\":%d,\"gen_ms\":%.0f,\"decode_tok_s\":%.2f}\n",
            t_load, t_p1, t_img, t_p2, n_gen, t_gen, n_gen > 0 ? n_gen * 1000.0 / t_gen : 0.0);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
