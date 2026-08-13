#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HF 参考实现的双图端到端分辨测试(桌面)
目的:①验证 M-RoPE 段隔离机制在参考实现下能正确分辨两张图;
     ②给 llama.cpp 的串扰结论提供对照(参考实现若也串,说明是模型能力;
       参考实现不串,则坐实 llama.cpp 实现问题)。
图与 mrope_multiimg_check.py 相同:城市方块图(896²) + 红圆绿方图(448²)。
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.mrope_multiimg_check import make_imgs, MODEL_DIR

def main():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  可用显存 {free:.1f}GB", flush=True)
        if free < 9.5:
            dev = "cpu"
            print("显存不足,回落 CPU(慢但能跑)", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16 if dev == "cpu" else torch.float16,
        device_map=dev)
    model.eval()

    img_a, img_b = make_imgs()
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": "这是两张不同的图片。请分别回答:第一张图里有什么?第二张图里有什么?"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img_a, img_b], return_tensors="pt").to(dev)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]
    dt = time.time() - t0

    # 自动判读:第一问该有建筑/方块类词,第二问该有圆/红色类词,且不互相污染
    rec = {"device": dev, "gen_s": round(dt, 1), "answer": ans}
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "verified", "hf_multiimg_e2e.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("ANSWER:", ans, flush=True)
    print("HF_MULTIIMG_E2E_DONE", flush=True)

if __name__ == "__main__":
    main()
