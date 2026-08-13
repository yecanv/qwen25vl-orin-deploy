#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 VL 推理（Python 参考实现）
================================
先用这个跑通、验证正确性，再看 C++ 版本（runtime/src/）。
Python 版是"对不对"的基准，C++ 版是"快不快"的实现。

链路：
  image → ViT TRT engine (FP16) → vision_embeds
  text  → tokenizer → input_ids (含 image_pad 占位)
  两者 → prompt_table + M-RoPE position_ids → LLM TRT-LLM engine → 流式输出
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np


class VitEngine:
    """ViT TensorRT engine 封装（dynamic shape）。"""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa

        self.trt = trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"engine 反序列化失败: {engine_path}\n"
                f"最常见原因：engine 是在别的机器/别的 TensorRT 版本上 build 的。\n"
                f"TensorRT engine 绑定 SM 架构与 TRT 版本，必须在本机重新 build。"
            )
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def __call__(self, pixel_values: np.ndarray, grid_thw: np.ndarray) -> np.ndarray:
        cuda = self.cuda
        self.ctx.set_input_shape("pixel_values", pixel_values.shape)
        self.ctx.set_input_shape("grid_thw", grid_thw.shape)

        out_shape = tuple(self.ctx.get_tensor_shape("vision_embeds"))
        if any(d < 0 for d in out_shape):
            # 输出形状是**数据依赖**的（由 grid_thw 的值而非形状决定），
            # TRT 在 enqueue 前报 -1，正规解法是 IOutputAllocator 运行时分配。
            # 但本模型输出行数是可预知的：n_patches / merge²（2x2 merger → /4），
            # hidden 维在 engine 里是静态的——直接按已知语义预分配，省掉 allocator。
            n_tok = pixel_values.shape[0] // 4
            hidden = out_shape[-1] if out_shape[-1] > 0 else 2048
            out_shape = (n_tok, hidden)
        d_pix = cuda.mem_alloc(pixel_values.nbytes)
        d_grid = cuda.mem_alloc(grid_thw.nbytes)
        out = np.empty(out_shape, dtype=np.float16)
        d_out = cuda.mem_alloc(out.nbytes)

        cuda.memcpy_htod_async(d_pix, np.ascontiguousarray(pixel_values), self.stream)
        cuda.memcpy_htod_async(d_grid, np.ascontiguousarray(grid_thw), self.stream)
        self.ctx.set_tensor_address("pixel_values", int(d_pix))
        self.ctx.set_tensor_address("grid_thw", int(d_grid))
        self.ctx.set_tensor_address("vision_embeds", int(d_out))
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(out, d_out, self.stream)
        self.stream.synchronize()
        return out


class VLPipeline:
    def __init__(self, model_id: str, vit_engine: str, llm_engine: str):
        from transformers import AutoProcessor
        from tensorrt_llm.runtime import ModelRunnerCpp

        self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.tok = self.proc.tokenizer

        # ★ fake prompt id 的基准必须是 **engine 里 embedding 表的行数**，
        #   也就是 config.vocab_size（Qwen2.5-VL-3B 是 151936；
        #   7B/72B 才是 152064——以下面动态读到的实际值为准），
        #   不是 len(tokenizer)（约 151665）。
        #
        #   TRT-LLM 的 PromptTuningEmbedding 判据是 `tokens >= vocab_size`，
        #   这个 vocab_size 来自 build engine 时的 embedding 层配置。
        #   用 len(tokenizer) 会让 fake id 落进 [len(tok), config.vocab_size)
        #   这段**真实但未使用**的 embedding 行里，engine 会走正常 lookup，
        #   拿到垃圾 embedding —— 现象是输出乱码，且极难定位。
        # ⚠️ 严格说，权威来源是 **llm_engine 目录下的 config.json**（engine 实际
        #   烧进去的 embedding 行数，TP 切分对齐时可能再 padding）。这里读 HF
        #   config 是离线可跑的近似；上板后若两者不一致，必须以 engine 值为准。
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", cfg)
        self.vocab_size = int(getattr(text_cfg, "vocab_size"))
        if self.vocab_size < len(self.tok):
            raise RuntimeError(
                f"config.vocab_size({self.vocab_size}) < len(tokenizer)"
                f"({len(self.tok)})，配置异常")
        print(f"[init] vocab_size={self.vocab_size} "
              f"(tokenizer {len(self.tok)}，差值 {self.vocab_size - len(self.tok)} 为 padding)")

        print("[init] 加载 ViT engine")
        self.vit = VitEngine(vit_engine)

        print("[init] 加载 LLM engine")
        t0 = time.perf_counter()
        self.llm = ModelRunnerCpp.from_dir(
            engine_dir=llm_engine,
            rank=0,
            max_output_len=512,
            kv_cache_free_gpu_memory_fraction=0.55,
        )
        self.cold_start_s = time.perf_counter() - t0
        print(f"[init] LLM 冷启动 {self.cold_start_s:.2f}s")

        # ---- 视觉 token 压缩（kernels/ 模块）----
        self.token_keep_ratio = 1.0
        self._tm = None
        try:
            import token_merge_cuda
            self._tm = token_merge_cuda
            print("[init] 视觉 token 压缩算子已加载")
        except ImportError:
            print("[init] 未安装 token_merge_cuda，token 压缩不可用"
                  "（CUDA_ARCH=87 python kernels/setup.py install）")

    def set_token_keep_ratio(self, ratio: float):
        """
        设置视觉 token 保留比例。1.0 = 不压缩。

        压缩发生在 ViT 输出之后、送入 LLM 之前——这是唯一正确的插入点：
          - 放在 ViT 之前：还没有语义特征，没法算相似度
          - 放在 LLM 之后：token 已经进了 attention，省不下计算量
        """
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio 需在 (0, 1]，收到 {ratio}")
        if ratio < 1.0 and self._tm is None:
            raise RuntimeError(
                "token 压缩需要先编译 CUDA 算子：\n"
                "  cd kernels && CUDA_ARCH=87 python setup.py install")
        self.token_keep_ratio = ratio

    def _compress_visual(self, vis):
        """
        对 ViT 输出的 vision embeds 做压缩。

        返回 (compressed_vis, kept_idx)：
          compressed_vis : [N_kept, D]
          kept_idx       : [N_kept] int，**存活 token 在原序列中的下标**

        ★ kept_idx 必须返回，不能假设"存活的是前 N_kept 个"。
          被合并掉的 A token 是按相似度 top-r 选的，在序列里**散布**，
          不是集中在尾部。用"删尾部"的假设去裁 position_ids，
          会让每个保留 token 都拿到别人的二维坐标——
          不报错，但图像的空间结构全乱，输出质量下降且极难归因。
        """
        import numpy as np
        import torch

        if self.token_keep_ratio >= 1.0 or self._tm is None:
            return vis, np.arange(vis.shape[0])

        N = vis.shape[0]
        n_drop = int(N * (1.0 - self.token_keep_ratio))
        if n_drop < 1:
            return vis, np.arange(N)

        t = torch.from_numpy(vis).cuda()
        idx, val = self._tm.fused_match(t)               # kernel 1

        Na = (N + 1) // 2
        r = min(n_drop, Na)
        # top-r：按相似度降序，只合并最相似的那批
        order = torch.argsort(val, descending=True)
        dst = torch.full((Na,), -1, dtype=torch.int32, device=t.device)
        dst[order[:r]] = idx[order[:r]]

        # 计算输出槽位
        keep = torch.ones(N, dtype=torch.bool, device=t.device)
        merged_a = order[:r] * 2                          # A token 在原序列的下标
        keep[merged_a] = False
        slot = torch.full((N,), -1, dtype=torch.int32, device=t.device)
        slot[keep] = torch.arange(int(keep.sum()), dtype=torch.int32,
                                  device=t.device)

        out = self._tm.merge_tokens(t, dst, slot, int(keep.sum()))  # kernel 2+3
        kept_idx = torch.nonzero(keep, as_tuple=True)[0].cpu().numpy()
        assert len(kept_idx) == out.shape[0], \
            f"存活下标数 {len(kept_idx)} != 输出 token 数 {out.shape[0]}"
        return out.cpu().numpy(), kept_idx

    @staticmethod
    def _shrink_after_compress(input_ids, n_raw: int, kept_vis_idx):
        """
        压缩后同步收缩 input_ids，并返回 keep_mask 供裁剪 position_ids。

        参数
        ----
        kept_vis_idx : 存活视觉 token 在**原视觉序列**中的下标（升序、散布）

        返回 (shrunk_ids, keep_mask)

        ★ 必须按实际存活下标裁，不能假设"存活的是前 N_kept 个"。
          这样每个保留的视觉 token 拿到的仍是它自己原本的二维坐标，
          图像的空间结构不被破坏。
        """
        import numpy as np
        from runtime.mrope import IMAGE_PAD_ID
        pos = np.where(input_ids == IMAGE_PAD_ID)[0]
        if len(pos) != n_raw:
            raise RuntimeError(
                f"占位符数 {len(pos)} != ViT 输出 {n_raw}，检查 processor 配置")
        keep = np.ones(len(input_ids), dtype=bool)
        dropped = np.setdiff1d(np.arange(n_raw), kept_vis_idx)
        keep[pos[dropped]] = False
        return input_ids[keep], keep

    def generate(self, image_path: str, prompt: str,
                 max_new_tokens: int = 256, stream: bool = True):
        from PIL import Image
        from runtime.mrope import compute_mrope_position_ids, build_prompt_table

        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img], return_tensors="np")

        input_ids = inputs["input_ids"][0]
        input_ids_raw = input_ids.copy()      # 压缩前的原始 ids，算 M-RoPE 用
        grid = inputs["image_grid_thw"]

        # ---- 阶段 1：视觉编码 ----
        t_vit0 = time.perf_counter()
        vis = self.vit(inputs["pixel_values"].astype(np.float16), grid.astype(np.int64))
        t_vit = (time.perf_counter() - t_vit0) * 1000

        # ---- 阶段 1.5：视觉 token 压缩 ----
        n_vis_raw = int(vis.shape[0])
        t_cmp0 = time.perf_counter()
        vis, kept_vis_idx = self._compress_visual(vis)
        t_compress = (time.perf_counter() - t_cmp0) * 1000
        n_vis_kept = int(vis.shape[0])

        # ★ 压缩后必须同时处理三件事，少一件就崩：
        #   ① input_ids 里的占位符数量同步减少
        #   ② position_ids 也要删掉对应列 —— 不能用压缩后的 ids 重算，
        #      因为 grid_thw 描述的仍是原始网格，重算会按原 token 数写越界，
        #      覆盖掉图后面的文本 token 的位置
        #   ③ 保留的视觉 token 沿用**原始的二维坐标**（语义上正确：
        #      它们在图里的位置没变）
        keep_mask = None
        if n_vis_kept != n_vis_raw:
            input_ids, keep_mask = self._shrink_after_compress(
                input_ids, n_vis_raw, kept_vis_idx)

        # ---- 阶段 2：桥接 ----
        # position_ids 用**压缩前**的 ids + 原始 grid 算，再按 keep_mask 裁列
        pos_ids_full, mrope_delta = compute_mrope_position_ids(input_ids_raw, grid)
        pos_ids = pos_ids_full[:, keep_mask] if keep_mask is not None else pos_ids_full
        if keep_mask is not None:
            # ★ 压缩后序列变短，decode 续算用的 delta 必须按压缩后长度重算：
            #   delta = (序列内最大位置 + 1) - 序列长度（见 mrope.py decode_step_position）。
            #   compute_mrope_position_ids 返回的 delta 是按压缩前长度算的，
            #   直接沿用会让首个生成 token 的位置落回 prompt 尾部文本的位置区间
            #   （每压掉 d 个 token 就重叠 d 个位置）——不报错但输出劣化、极难归因。
            mrope_delta = int(pos_ids.max()) + 1 - pos_ids.shape[1]

        fake_ids, ptable = build_prompt_table(input_ids, vis, self.vocab_size)

        assert pos_ids.shape[1] == len(fake_ids), (
            f"position_ids 列数 {pos_ids.shape[1]} != input_ids 长度 {len(fake_ids)}")

        # ---- 阶段 3：LLM 生成 ----
        import torch
        t_llm0 = time.perf_counter()
        first_token_at = None
        pieces = []

        # ⚠️ 适配警告（上板必改）：mrope_position_ids / mrope_position_deltas
        #   这两个 kwarg 是**期望接口的占位写法**，已发布的 TensorRT-LLM 版本
        #   都不接受它们（传入即 TypeError）。真实路线（TRT-LLM ≥0.15 的
        #   Qwen2-VL 多模态示例）：把三路 position_ids 预计算成 rotary
        #   cos/sin 表，经 MropeParams(mrope_rotary_cos_sin=...,
        #   mrope_position_deltas=...) 传入。本调用只表达数据流意图，
        #   详见 AUDIT_RESPONSE.md「未验证」节。
        gen = self.llm.generate(
            [torch.from_numpy(fake_ids.astype(np.int32))],
            prompt_table=torch.from_numpy(ptable),
            mrope_position_ids=torch.from_numpy(pos_ids),
            mrope_position_deltas=torch.tensor([mrope_delta]),
            max_new_tokens=max_new_tokens,
            end_id=self.tok.eos_token_id,
            pad_id=self.tok.pad_token_id or self.tok.eos_token_id,
            streaming=stream,
        )

        for out in gen:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            ids = out["output_ids"][0][0].tolist()
            piece = self.tok.decode(ids[len(fake_ids):], skip_special_tokens=True)
            pieces.append(piece)
            if stream:
                # piece 是**累计**解码串，增量 = 相对上一次累计串的后缀。
                # （不能用 "".join(pieces[:-1])：那是所有历史累计串的拼接，
                #   长度超线性增长，第 3 块起切片起点越过串尾，增量恒为空）
                prev = pieces[-2] if len(pieces) > 1 else ""
                print(piece[len(prev):], end="", flush=True)

        t_total = (time.perf_counter() - t_llm0) * 1000
        ttft = (first_token_at - t_llm0) * 1000 if first_token_at else float("nan")
        n_out = len(self.tok(pieces[-1] if pieces else "")["input_ids"])

        return {
            "text": pieces[-1] if pieces else "",
            "n_visual_tokens": n_vis_kept,
            "n_visual_tokens_raw": n_vis_raw,
            "token_keep_ratio": round(n_vis_kept / max(n_vis_raw, 1), 3),
            "compress_ms": round(t_compress, 2),
            "n_prompt_tokens": int(len(input_ids)),
            "n_output_tokens": n_out,
            "vit_ms": round(t_vit, 2),
            "ttft_ms": round(t_vit + ttft, 2),      # 端到端 TTFT 含视觉编码
            "llm_ttft_ms": round(ttft, 2),
            "total_ms": round(t_vit + t_total, 2),
            "decode_tok_s": round(n_out / (t_total / 1000 - ttft / 1000), 2)
                            if n_out > 1 else 0.0,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--vit-engine", default="engines/vit_fp16.engine")
    ap.add_argument("--llm-engine", default="engines/llm_int4awq")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="详细描述这张图片的内容。")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    pipe = VLPipeline(args.model, args.vit_engine, args.llm_engine)
    r = pipe.generate(args.image, args.prompt, args.max_new_tokens)

    print("\n" + "-" * 60)
    for k, v in r.items():
        if k != "text":
            print(f"{k:20s} {v}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
