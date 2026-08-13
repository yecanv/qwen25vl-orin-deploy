#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Qwen2.5-VL checkpoint 剥离语言主干 → 纯 Qwen2ForCausalLM 格式
（safetensors 外科手术，不实例化模型，绕开 transformers 版本差异）"""
import json, os, shutil, sys

SRC = os.path.expanduser("~/models/qwen25vl-3b")
DST = os.path.expanduser("~/models/qwen25vl-3b-text")

from safetensors import safe_open
from safetensors.torch import save_file

os.makedirs(DST, exist_ok=True)

# ---- 1. 读权重索引，分类键 ----
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wmap = idx["weight_map"]
keep, drop = {}, []
for k, shard in wmap.items():
    if ".visual." in k or k.startswith("visual."):
        drop.append(k)
    elif k.startswith("model.") or k == "lm_head.weight":
        keep.setdefault(shard, []).append(k)
    else:
        print(f"[warn] 未分类键（默认保留）: {k}")
        keep.setdefault(shard, []).append(k)
print(f"保留 {sum(len(v) for v in keep.values())} 张量 / 丢弃 {len(drop)} 张量(visual)")

# ---- 2. 逐 shard 搬运（内存友好）----
new_map, total_bytes = {}, 0
for i, (shard, keys) in enumerate(sorted(keep.items()), 1):
    out_name = f"model-{i:05d}-of-{len(keep):05d}.safetensors"
    tensors = {}
    with safe_open(f"{SRC}/{shard}", framework="pt") as f:
        for k in keys:
            t = f.get_tensor(k)
            tensors[k] = t
            total_bytes += t.numel() * t.element_size()
    save_file(tensors, f"{DST}/{out_name}", metadata={"format": "pt"})
    for k in keys:
        new_map[k] = out_name
    print(f"  {shard} -> {out_name}: {len(keys)} 张量")
json.dump({"metadata": {"total_size": total_bytes}, "weight_map": new_map},
          open(f"{DST}/model.safetensors.index.json", "w"), indent=2)
print(f"文本主干总大小: {total_bytes/1e9:.2f} GB")

# ---- 3. 重写 config：qwen2_5_vl → qwen2 ----
cfg = json.load(open(f"{SRC}/config.json"))
KEEP_KEYS = ["hidden_size", "intermediate_size", "num_hidden_layers",
             "num_attention_heads", "num_key_value_heads", "hidden_act",
             "max_position_embeddings", "rms_norm_eps", "rope_theta",
             "tie_word_embeddings", "torch_dtype", "vocab_size", "use_cache",
             "attention_dropout", "initializer_range", "sliding_window",
             "use_sliding_window", "max_window_layers",
             "bos_token_id", "eos_token_id"]
new_cfg = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}
src_text = cfg.get("text_config", cfg)   # 兼容嵌套/平铺两种格式
for k in KEEP_KEYS:
    if k in src_text:
        new_cfg[k] = src_text[k]
    elif k in cfg:
        new_cfg[k] = cfg[k]
if "rope_scaling" in src_text or "rope_scaling" in cfg:
    print("[note] 摘除 rope_scaling(mrope)——纯文本下三路位置同值等价于标准 1D RoPE")
json.dump(new_cfg, open(f"{DST}/config.json", "w"), indent=2)
print("config: 层数", new_cfg.get("num_hidden_layers"), "| KV头", new_cfg.get("num_key_value_heads"),
      "| vocab", new_cfg.get("vocab_size"), "| tie_embeddings", new_cfg.get("tie_word_embeddings"))

# ---- 4. 拷 tokenizer / generation 配置 ----
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "generation_config.json"]:
    p = f"{SRC}/{f}"
    if os.path.exists(p):
        shutil.copy(p, f"{DST}/{f}")
        print(f"copied {f}")

has_lm_head = any(k == "lm_head.weight" for k in new_map)
print(f"lm_head.weight 在权重中: {has_lm_head}（tie_word_embeddings 时可能不在，convert 会处理）")
print("DONE strip")
