# 实测数据台账(截至 2026-08-11 寄板)

> 用途:每个对外引用的数字 → 原始文件 → 口径 → 副本位置,一查即达。
> 副本约定:**仓库** = 本 results/(git 版本化);**E 盘** = E:\jetson_board_backup_2026-08-11\(全量快照 + post_backup_additions 补充);**存档** = results/board_archive/(带 md5 清单的早期归档)。
> 板卡 2026-08-11 寄回,板上数据此后不再是副本。

## 一、engine 端(TensorRT / TensorRT-LLM 链路)

| 数字 | 口径 | 原始文件 | 副本 |
|---|---|---|---|
| decode 45.9 tok/s | TRT-LLM INT4-AWQ 引擎,文本 decode | verified/orin/llm_int4awq_run.txt | 仓库+E+存档 |
| 视觉编码 750ms / P50 743 | ViT TRT 引擎纯 GPU 前向 | verified/orin/vit_orin_verify.json(0.999853 版) | 仓库+E+存档 |
| golden 余弦 0.999853 | 板端引擎 vs 桌面 PyTorch | 同上;08-11 用 st 引擎复验一致(记录于 docs/07 第11步) | 同上 |
| 历史 bug 值 0.1247 | stronglyTyped 修复前动态引擎 | vit_orin_verify.txt/verify2.txt;08-11 复现记录 E 盘 post_backup_additions/vit_orin_verify_OVERWRITTEN_BY_DIAG_0.1247.json | 仓库+E+存档 |
| 五点构建曲线 55/53/55/119/124min | trtexec L1-L5,推理无差异 | verified/orin/vit_orin_verify_st.txt、vit_st_*timing(存档);构建日志见存档 | 仓库+E+存档 |
| TTFT 组件拆解 | 750ms+桥接+首步 | verified/orin/vlm_ttft_decompose.txt | 仓库+E+存档 |
| ncu 剖析 | token_merge 与 ViT st 引擎 | 存档 ncu_*.ncu-rep + *_details.txt | E+存档 |

## 二、llama.cpp 链路(engine 端)

| 数字 | 口径 | 原始文件 | 副本 |
|---|---|---|---|
| tg 26.56 / pp 1009 | llama-bench 文本 | verified/orin/llamacpp_bench_text.txt | 仓库+E+存档 |
| mmproj 2595ms | mtmd-cli 单图端到端日志 | verified/orin/llamacpp_vlm_e2e.txt | 仓库+E+存档 |
| PPL 11.589(f16)/11.915(Q4,+2.8%) | wikitext-2,llama-perplexity | 存档 ppl_f16.txt / ppl_q4km.txt | E+存档 |
| TextVQA 72.33/71.17(-1.16,p=0.727) | 200 题,McNemar | verified/textvqa_result.json(+preds) | 仓库+E |

## 三、服务端(llama-server)

| 数字 | 口径 | 原始文件 | 副本 |
|---|---|---|---|
| 单路 P50 26.47/26.5 | 30min 串行 371 请求首窗 | verified/orin/longrun_summary.json + longrun_seq.csv | 仓库+E+存档 |
| 并发曲线 26.1/47.5/60.4/68.9 | 1/2/4/8 路,-np 8,风扇拉满 | verified/orin/fanfix30/conc_sweep_np8.json | 仓库+E |
| 30min 漂移 0.0% | 风扇拉满复跑 385 请求 | verified/orin/fanfix30/longrun_summary.json | 仓库+E |
| 历史 4 路 54.8 | 热饱和态、-np 4 实例(考古定案) | verified/orin/longrun_summary.json phase_c;考古记录见 4_ 文档战区表与课堂笔记 | 仓库+E+存档 |
| -6.6% 漂移与热定位 | 默认风扇,99°C DVFS | longrun_summary.json + tegrastats_longrun.log.gz + throttle_analysis.md | 仓库+E+存档 |
| 风扇重配 10min 验证 | 2min×5 窗 | verified/orin/fan_verify.json | 仓库+E+存档 |
| TTFT 稳态 3.5~4.2s | 流式首 chunk,三路互证 | verified/ttft_direct.json | 仓库+E |
| serving 12.6 / 缓存命中 0.79s | OpenAI API 跨机实测 | verified/serving_demo.json | 仓库+E |
| 冷启动 4.53s | server 起立至可服务 | verified/orin/cold_start.txt | 仓库+E+存档 |
| 内存曲线 空闲3.1/中位4.8/并发峰值5.97GB | tegrastats 1Hz 整机口径 | verified/orin/memory_profile_from_tegrastats.json(挖掘自 tegrastats_longrun.log.gz) | 仓库+E+存档 |
| 首请求预热 ≈2.8s | 首发 6.30s vs 稳态 3.5s | verified/ttft_direct.json | 仓库+E |
| 延迟/TPOT 分位数与失败率(两组长稳) | TPOT=每请求均值口径 | verified/orin/latency_quantiles_longrun.json(存档 CSV 现算) | 仓库+E |
| 能效 0.95 tok/J / 功耗三档 | 均功耗 27.9W;三档实录 MAXN_SUPER 26.5 / 25W 10.85 / 15W 8.27 tok/s | longrun_summary.json + verified/orin/power_sweep.txt | 仓库+E+存档 |

## 三-补、寄板前 24h 补测(2026-08-12,return_day2/)

| 数字 | 口径 | 文件 |
|---|---|---|
| ViT INT8 引擎尝试:五墙未竟,教训归档 | 含 FP32 校准契约/DDS 杀执行器/inf 卡熵校准等 | verified/orin/return_day2/vit_int8_attempt_lessons.json |
| 文本首字 0.16~2.80s(32~2048tok 四档直测)+ pp2048 990 | llama-server 流式,cache 关 | verified/orin/return_day2/text_ttft_card.json |
| 量化提速实测 F16 12.85 → Q4 26.55(2.07×) | llama.cpp 同栈背靠背;字节比3.21×kernel效率0.64 | verified/orin/return_day2/fp16_vs_q4_decode.json | 仓库+E |
| reboot-to-ready ≈51.5s(44+7+0.5) | 软重启,非物理断电 | verified/orin/return_day2/phase1_reboot_to_ready.json |
| 内存六点法 + 切换双占 MemFree 156MB | meminfo+smaps+tegrastats | verified/orin/return_day2/phase2_smaps_switch.json |
| VLM TTFT P50/P95/P99=3.59/3.60/3.63s(n=50 全冷) | 50 互异合成图流式首 chunk | verified/orin/return_day2/phase3_ttft_energy.json |
| 每请求 115.5J / 全请求 3.16 J/token / idle 5.46W | VDD_IN 10Hz 积分,板钟 | 同上 |
| 默认风扇 30min 复跑:漂移 -0.04%(未复现,环境温度条件性) | 同协议顺序对调,26°C 空调房,板温峰值≈63°C | verified/orin/return_day2/repeat_defaultfan/ + repeat_defaultfan_analysis.json |

## 四、量化与消融(桌面伪量化口径)

| 数字 | 文件 |
|---|---|
| AWQ 保护 α U 曲线(谷底 0.0216@0.2) | verified/alpha_scan_desktop.json |
| SmoothQuant α(0.0234→0.0062@0.5) | verified/sq_alpha_scan_desktop.json |
| 校准四象限(直觉未复现) | verified/calib_ablation_desktop.json + calib_ablation_a*.json |
| 敏感度/outlier 1.59e6@llm_mlp_down | verified/sensitivity_desktop.json |
| 融合算子 bench | verified/kernel_bench_desktop.json + orin/kernel_bench_orin.json |

## 五、多图与混合端到端(2026-08-11)

| 内容 | 文件 |
|---|---|
| llama.cpp 串扰三次修订(触发条件+解法) | verified/orin/fanfix30/multiimg_synthetic_test.json |
| HF 参考双图零串扰 | verified/hf_multiimg_e2e.json |
| 混合端到端(板端 TRT 特征+HF LLM,余弦 0.9997) | verified/hybrid_trt_vit_multiimg.json + multiimg_vit_embeds.npz |
| **混合端到端实测(TRT ViT→llama.cpp LLM,板上原生,2026-08-12)**:TTFT≈1.9s(741ms ViT+1133ms prefill,未含预处理)vs 纯链 3.59s;decode 25.8~26.0;双图判据 PASS 零串扰 | verified/orin/return_day2/hybrid_llamacpp_e2e.json + runtime/hybrid_driver.cpp + stderr_a/b.log |
| 引擎输入(复现实验用) | E 盘 post_backup_additions/hybrid_inputs.npz |
| 位置编码对拍/胶水/执行测试 | runtime/mrope_multiimg_check.py、test_multiimg_glue.py、test_run_vl_multiimg_exec.py(脚本即记录,输出见 git 提交说明) |

## 六、事故与环境

| 内容 | 文件 |
|---|---|
| GPU 视觉数值故障 20 步排障 | docs/07_incident_llamacpp_vision_numerical.md |
| 重编译/最新版构建日志 | E 盘 post_backup_additions/rebuild.log、build_latest.log |
| 探针产物(问号 json、测试图、算子脚本) | E 盘 post_backup_additions/tmp_artifacts/ |
| 板端环境指纹 | E 盘 version_info/board_env.txt + verified/orin/device_info.json + probe_orin_with_bw.txt |
| 实验时间线 | 本仓库 git log(每次实验一提交,含口径说明) |
