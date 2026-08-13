#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实 run_vl.py 多图路径执行测试(桌面,引擎/LLM 打桩)
与 test_multiimg_glue.py 的区别:那个验证的是逻辑复刻,这个执行的是
runtime/run_vl.py 文件本身的 generate()——桩只替换硬件依赖:
  vit 桩   : 输出行编码 [图号, 图内下标, ...],可逐行溯源
  压缩桩   : 每段确定性丢第 1、3 个 token(触发 shrink/裁列/delta 重算全路径)
  llm 桩   : 记录收到的 fake_ids/ptable/pos_ids/delta,回吐固定 token
最后对 llm 桩收到的实物做契约断言。
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from runtime import run_vl
from runtime.mrope import IMAGE_PAD_ID
from runtime.mrope_multiimg_check import make_imgs, MODEL_DIR


class FakeVit:
    def __init__(self):
        self.calls = []
    def __call__(self, pix, grid):
        i = len(self.calls)
        self.calls.append((pix.shape, grid.tolist()))
        n_tok = pix.shape[0] // 4
        out = np.zeros((n_tok, 8), dtype=np.float16)
        out[:, 0] = i
        out[:, 1] = np.arange(n_tok)
        return out


class FakeLLM:
    def __init__(self):
        self.received = None
    def generate(self, batch_ids, **kw):
        self.received = {"ids": batch_ids[0].numpy(), **{k: v for k, v in kw.items()}}
        n = len(self.received["ids"])
        import torch
        yield {"output_ids": torch.tensor([[list(self.received["ids"]) + [100, 101]]])[0][None]}


def main():
    from transformers import AutoProcessor, AutoTokenizer

    # 绕过 __init__(不加载真引擎),手工装配 VLPipeline 实例
    pipe = run_vl.VLPipeline.__new__(run_vl.VLPipeline)
    pipe.proc = AutoProcessor.from_pretrained(MODEL_DIR)
    pipe.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pipe.vit = FakeVit()
    pipe.llm = FakeLLM()
    pipe.vocab_size = 151936
    pipe.token_keep_ratio = 1.0
    pipe._tm = None

    # 压缩桩:绑定到实例,每段确定性丢第 1、3 个
    def fake_compress(self, vis):
        n = vis.shape[0]
        keep = np.ones(n, dtype=bool)
        if n > 4:
            keep[[1, 3]] = False
        idx = np.nonzero(keep)[0]
        return vis[idx], idx
    pipe._compress_visual = types.MethodType(fake_compress, pipe)

    # processor 兼容:新版 fast processor 不支持 np,包一层转换
    real_proc = pipe.proc
    class ProcShim:
        def __getattr__(self, k): return getattr(real_proc, k)
        def __call__(self, **kw):
            kw["return_tensors"] = "pt"
            out = real_proc(**kw)
            return {k: v.numpy() for k, v in out.items()}
    pipe.proc = ProcShim()

    img_a, img_b = make_imgs()
    import tempfile
    pa = os.path.join(tempfile.gettempdir(), "mi_a.png"); img_a.save(pa)
    pb = os.path.join(tempfile.gettempdir(), "mi_b.png"); img_b.save(pb)

    r = pipe.generate([pa, pb], "分别描述两张图。", max_new_tokens=4, stream=False)

    # ---- 契约断言(对 llm 桩收到的实物)----
    rec = pipe.llm.received
    ids, ptable = rec["ids"], rec["prompt_table"].numpy()
    pos_ids = rec["mrope_position_ids"].numpy()
    delta = int(rec["mrope_position_deltas"][0])

    # ① ViT 被逐图调用,形状正确
    assert len(pipe.vit.calls) == 2, f"ViT 调用次数 {len(pipe.vit.calls)} != 2"
    assert pipe.vit.calls[0][0][0] == 4096 and pipe.vit.calls[1][0][0] == 1024
    # ② 每段丢 2 个:视觉 token 2560-4=2556;表行数一致
    n_fake = int((ids >= 151936).sum())
    assert n_fake == 1276 == ptable.shape[0], f"fake id {n_fake} / 表 {ptable.shape[0]}"
    # ③ 表行溯源:fake id 段内行必须属于正确的图
    fake_rows_img = ptable[:, 0]
    seg_a = int((np.asarray(fake_rows_img) == 0).sum())
    seg_b = int((np.asarray(fake_rows_img) == 1).sum())
    assert seg_a == 1022 and seg_b == 254, f"段行数 {seg_a}/{seg_b} != 1022/254"
    # ④ 位置矩阵与序列同宽,delta 按压缩后长度重算
    assert pos_ids.shape[1] == len(ids)
    assert delta == int(pos_ids.max()) + 1 - pos_ids.shape[1]
    # ⑤ 返回结构完整
    assert r["n_visual_tokens"] == 1276 and r["n_visual_tokens_raw"] == 1280

    print(f"ViT 逐图调用: 4096patch + 1024patch  压缩后视觉 token 1276(1022+254)")
    print(f"序列 {len(ids)}  表 {ptable.shape[0]} 行  delta={delta}")
    print("RUN_VL_MULTIIMG_EXEC_PASS")


if __name__ == "__main__":
    main()
