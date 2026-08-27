# 板端 VLM HTTP 服务(OpenAI 兼容)

把"能跑"变成"能请求"的最后一公里。实测通过。

## 启动(板端一条命令)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gguf/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  --mmproj ~/models/gguf/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
  -ngl 99 --host 0.0.0.0 --port 8080 -c 4096 -np 2 --metrics
```

- `--mmproj` 挂上视觉投影模块 → 服务从纯文本变成图文
- `-np 2` 两个并发槽(n_ctx_slot=2048 each)
- `--metrics` 开 Prometheus 指标端点 `/metrics`

就绪判定:`curl http://192.168.1.2:8080/health` → `{"status":"ok"}`

## 请求(任意机器,局域网内)

```bash
curl -s http://192.168.1.2:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-vl-3b","max_tokens":128,
       "messages":[{"role":"user","content":[
         {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<BASE64>"}},
         {"type":"text","text":"详细描述这张图片的内容。"}]}]}'
```

Python 版见 `serving/vlm_client_demo.py`(自动读图转 base64、打印用量与速率)。

## 实测(桌面 → 板端局域网)

| 轮次 | 内容 | 整轮耗时 | prompt/completion tokens |
|---|---|---|---|
| 第 1 轮(冷) | "详细描述这张图片的内容" | **7.33 s** | 915 / 92 |
| 第 2 轮(同图换问题) | "图中有文字吗?原样写出来" | **0.79 s** | — / 11 |

- 第 1 轮 serving 口径整体速率 ≈ **12.6 tok/s**(含视觉编码 + 网络 + 服务开销,
  与 CLI 口径 decode 26.56 tok/s 不是同一个量,不可混报)
- 第 2 轮快 9 倍的原因:**prompt cache 命中**——同一张图的视觉编码结果被服务端复用,
  省掉约 2.6 s 的 mmproj 编码 + 915 token 的 prefill。
  这是 serving 相对 CLI 的结构性优势,也是多轮对话场景的真实收益。
- 答案质量抽样:图片为 DrupalCon Copenhagen 横幅,模型正确识别横幅文字、
  水滴造型与配色,第二轮 OCR 正确输出 "DRUPALCON COPENHAGEN"。

原始证据:`results/verified/serving_demo.json`

## 与已有 benchmark 的口径关系

| 口径 | 数字 | 说明 |
|---|---|---|
| CLI decode(llama.cpp) | 26.56 tok/s | 纯 decode,不含视觉与服务开销 |
| CLI decode(TRT-LLM) | 45.9 tok/s | A 轨,文本干引擎 |
| **serving 整轮(本页)** | **12.6 tok/s** | 端到端用户视角:视觉编码+prefill+decode+网络 |
| serving 热轮 | 0.79 s / 11 tok | prompt cache 命中后的多轮体验 |

**报数原则**:serving 口径永远比 CLI 低,因为它含全部真实开销——
面试报数字时必须说清是哪个口径,否则就是在骗人。

## 尚未做的(产品化差距,诚实列出)

- 无鉴权、无限流、无 TLS(局域网内演示级)
- 无进程守护/自动重启(systemd unit 未写)
- 无请求日志与监控落盘(`--metrics` 端点开着但没接采集)
- C++ 自研网关未做(当前直接用 llama-server)
