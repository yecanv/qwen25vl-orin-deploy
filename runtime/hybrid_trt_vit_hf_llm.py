#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混合端到端:板端真 TRT ViT 引擎特征 + 真 LLM(HF 同权重)双图分辨
链路:Orin 上 vit_fp16_st.engine 分别前向两图(golden 0.999853 的那个引擎)
     → 特征 npz 回桌面 → 替换 HF 模型的视觉塔 → 正常 generate。
被验证的实物:TRT 视觉特征的语义可用性 + 多图注入顺序 + M-RoPE 段隔离。
唯一的替身是解码器运行时(HF 代替 TRT-LLM,原因:0.12 无 mrope 输入)。
附加度量:TRT 特征 vs HF 自算特征的逐图余弦(引擎保真度)。
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.mrope_multiimg_check import MODEL_DIR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(REPO, "results", "verified", "multiimg_vit_embeds.npz")


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    D = np.load(NPZ)
    trt_a = torch.from_numpy(D["embeds_a"].astype(np.float32))
    trt_b = torch.from_numpy(D["embeds_b"].astype(np.float32))

    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    visual = getattr(model, "visual", None) or model.model.visual

    img_a = Image.open(os.path.join(REPO, "assets", "demo896.jpg")).convert("RGB")
    img_b = Image.open(os.path.join(REPO, "assets", "shapes896_hybrid.png")).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": "这是两张不同的图片。请分别回答:第一张图里有什么?第二张图里有什么?"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img_a, img_b], return_tensors="pt").to("cuda")

    # ---- 度量①:TRT 特征 vs HF 自算特征,逐图余弦 ----
    with torch.no_grad():
        hf_all = visual(inputs["pixel_values"].to(torch.float16),
                        grid_thw=inputs["image_grid_thw"]).float().cpu()
    def cos(x, y):
        x, y = x.ravel(), y.ravel()
        return float(x @ y / (x.norm() * y.norm() + 1e-12))
    c_a = cos(trt_a, hf_all[:1024]); c_b = cos(trt_b, hf_all[1024:])
    print(f"TRT vs HF 视觉特征余弦: 图A {c_a:.6f}  图B {c_b:.6f}", flush=True)

    # ---- 视觉塔替换为 TRT 特征桩(拼接顺序 = 图序,与 run_vl 注入一致) ----
    trt_cat = torch.cat([trt_a, trt_b]).to("cuda", torch.float16)
    class TrtVisualStub(torch.nn.Module):
        def forward(self, pixel_values, grid_thw=None, **kw):
            n = int((grid_thw.prod(dim=-1) // 4).sum())
            assert n == trt_cat.shape[0], f"视觉 token 数 {n} != {trt_cat.shape[0]}"
            return trt_cat
        # generate 路径可能查询的属性
        @property
        def dtype(self):
            return torch.float16
    stub = TrtVisualStub()
    if hasattr(model, "visual"):
        model.visual = stub
    else:
        model.model.visual = stub

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]
    dt = time.time() - t0

    rec = {"vision_source": "Orin vit_fp16_st.engine(golden 0.999853,双图各一次前向)",
           "llm": "HF 同权重 fp16(解码器运行时替身,原因:TRT-LLM 0.12 无 mrope 输入)",
           "cos_trt_vs_hf_visual": {"img_a": round(c_a, 6), "img_b": round(c_b, 6)},
           "gen_s": round(dt, 1), "answer": ans}
    outp = os.path.join(REPO, "results", "verified", "hybrid_trt_vit_multiimg.json")
    json.dump(rec, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("ANSWER:", ans, flush=True)
    print("HYBRID_MULTIIMG_DONE", flush=True)


if __name__ == "__main__":
    main()
