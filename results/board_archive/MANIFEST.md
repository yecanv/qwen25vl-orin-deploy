# 板卡备份清单(board_archive)

- **备份日期**:2026-08-04
- **来源**:租用 Orin NX 16GB(192.168.1.2,JetPack r36.4,SM87)
- **核对方式**:板端 md5sum ↔ 本地 hashlib 逐文件比对,**55/55 通过**

## 已备份内容

| 目录 | 内容 | 说明 |
|---|---|---|
| `scripts/` | 板上 10 个实验脚本 | 6 个 ViT 验证(verify→verify4/fp32/st,六步排障链)、vit_perm_diag(排列诊断)、cold_server(冷启动)、longrun_client(长稳)、strip_text_backbone(剥文本干) |
| `data/` | vit_golden_4096.npz(13MB) | 桌面 golden 输入/输出,cos 0.999853 对比的基准数据 |
| `raw/` | 全部实验原始输出 28 份 | 含 **vit_orin_wrong_output.npy**(cos 0.1247 翻车物证)、**ncu_vit_st.ncu-rep + 2254 行明细**(2026-08-04 采样,vit_fp16_st 引擎 30 kernel,--set basic) |
| `verified_boardside/` | 桌面预验证期产物 11 份 | 敏感度散点图/JSON、校准探针日志、kernel 桌面对拍、ONNX 导出日志 |
| `logs/` | 构建日志 | vit_build.log.gz(28MB verbose→1.4MB,253,366 行,L4 fp16 失败那次=第三课教材)、static/st 构建日志、ncu 运行日志 |
| `run_vl_board.py` | 板上改过 720 行的 runtime | 与仓库 runtime/run_vl.py 的分叉版本,VLM 实测实际用的代码 |

## 未备份(可重建,记录在案)

| 资产 | 体积 | 重建方式 | 耗时 |
|---|---|---|---|
| vit_fp16/static/st.engine ×3 | 1.3GB×3 | trtexec 命令在各 build log 抬头 | L2≈53min(2026-08-04 复测 56min)/L3≈55min/L4≈119min |
| llm-int4awq engine | 2.6GB | trtllm-build(命令在 qbank 03) | 2m57s |
| trtllm-ckpt-int4awq | 2.6GB | docker 内 AWQ 校准(quantize.py) | ≈13min |
| HF 模型/剥离版/GGUF | 7+6.17+2.6GB | HF 下载 / strip脚本 / convert脚本 | 网速决定 |
| llama.cpp 构建产物 | — | cmake(板上源码目录仍在) | ≈20min |
| token_merge_ref(ELF) | 32KB | 桌面仓库 kernels 源码重编 | 秒级 |

## 待补(2026-08-04 过夜任务出结果后)

- [x] vit_st_L4/L5.engine 构建日志(engines 本体不备份)→ `logs/vit_st_L4_build.log`、`logs/vit_st_L5_build.log`
- [x] L2 vs L4 vs L5 推理延迟对比数据 → `raw/vit_st{,_L4,_L5}_timing.txt`:GPU Compute mean 746.7 / 740.8 / 745.3 ms(L4 两小时买 0.8%,L5 归零)——收益递减实证,L2 已近天花板
