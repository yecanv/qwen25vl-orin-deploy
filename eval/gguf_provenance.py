# -*- coding: utf-8 -*-
"""核查两个 GGUF 的血统与元数据差异(公平对比的前提)"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect("192.168.1.2", username="nvidia", timeout=20)

def run(cmd, cap=2500, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace").strip()
    print(f"$ {cmd[:110]}")
    if o: print(o[:cap])
    print("-" * 58)
    return o

print("== 两个 GGUF 的文件与时间 ==")
run("ls -lh ~/models/gguf/")

print("== 元数据对比(架构/词表/chat template 指纹/训练上下文) ==")
py = r'''
import sys, hashlib
from gguf import GGUFReader
for p in ["/home/nvidia/models/gguf/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
          "/home/nvidia/models/gguf/qwen25vl-3b-text-f16.gguf"]:
    r = GGUFReader(p)
    kv = {}
    for k, f in r.fields.items():
        try:
            v = f.parts[f.data[0]] if f.data else None
            if hasattr(v, "tolist"):
                v = v.tolist()
                if isinstance(v, list) and len(v) > 8: v = v[:8]
            kv[k] = v
        except Exception:
            pass
    name = p.split("/")[-1]
    print("###", name)
    for key in ["general.architecture", "general.name", "general.file_type",
                "qwen2vl.context_length", "qwen2vl.embedding_length",
                "qwen2vl.block_count", "qwen2vl.attention.head_count",
                "qwen2vl.attention.head_count_kv", "qwen2vl.rope.freq_base",
                "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"]:
        if key in kv: print(f"   {key} = {kv[key]}")
    # chat template 指纹
    for k, f in r.fields.items():
        if "chat_template" in k:
            try:
                s = bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
            except Exception:
                s = str(f.parts[f.data[0]])[:200]
            print(f"   chat_template len={len(s)} md5={hashlib.md5(s.encode()).hexdigest()[:12]}")
            print(f"   head: {s[:120]!r}")
    # 词表大小
    for k, f in r.fields.items():
        if k == "tokenizer.ggml.tokens":
            print(f"   vocab_size = {len(f.data)}")
    print(f"   tensor_count = {len(r.tensors)}")
'''
run(f"python3 -c '{py}'", cap=4000, t=180)
cli.close()
