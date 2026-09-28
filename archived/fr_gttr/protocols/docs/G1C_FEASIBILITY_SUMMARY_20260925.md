# G1c 1B Feasibility 结果摘要（2026-09-25 整理轮，依据冻结制品摘录）

来源：`outputs/generalization_g1c_feasibility_v01/`（协议
`G1C_FEASIBILITY_PROTOCOL.md`，预注册冻结于 2026-09-17T02:15:50Z）。
**本 gate 不产生任何泛化结论/规模声明；正式 G1c 科学实验从未授权。**

## 判定链

| 阶段 | 判定 | 关键数字 |
|---|---|---|
| Phase 0（step950000 恢复门） | **PASS**（13/13 checks ok，38.3s） | model/optim/train 三件可恢复，SHA256 存档 |
| Phase 1（step950000×stream1×H=1 smoke） | **PASS** | replay 位级一致（GPU loss gap=0；CPU fp32 全梯度 gap=0）；H=1 恒等 \|B1−C1\|=0；vs FD 相对误差 eps 1.70e-4 / eps÷2 7.60e-5；HVP 峰值显存 20.2GB、646s |
| Phase 2（3 checkpoint × 2 stream × H{1,4,8} = 18 cells） | **未完成，中途中止**（2026-09-18 20:07，无 DECISION/汇总报告） | 首个 cell（step380000×stream1）：B 切线 h1–h8 稳定，chi −1.85009 → −10.6971；C 分支 h=7 处 NaN（wte 冻结 guard 触发） |

## C 分支失败诊断（G1C_C_FAILURE_DIAGNOSTIC_V01，2026-09-19）

四分类裁决：**主因 C（scale-dependent limitation）**，近因 B（fp16 溢出）；
A（实现 bug）与 D（optimizer 状态异常）排除。依据要点：失败跨 2 次在跑重复
与 4 次独立 launch 位级确定；同代码路径 B 稳定至 H=8；fp32 输入时同一
horizon 有限（chi=−34158）但呈相同定性轨迹（h=6 符号翻转、h=8 chi→+3.7e6、
tangent 范数→3e11），而 26M/100M 同协议保持 O(1)（p50 0.47/0.65）。机制：
wte 稀有 token 行存在精确零 `exp_avg_sq` 坐标 → AdamW 分母落到 eps 地板
（实测 1e-4）→ 逐坐标自适应增益 ~1e4 → 在 1B 的深 clip 区（系数 0.003–0.012
vs 26M/100M 的 0.7–1.0）C 递归自 h≈3 起超线性放大 → h=7 HVP 内 fp16 上溢。

## 与其他实验的关系

- 输入来源：官方 OLMo-2-0425-1B（peteish1）standard checkpoint（复核制品
  `outputs/generalization_g1c_checkpoint_recheck_v01/`，
  FULL_STANDARD_CHECKPOINT_AVAILABLE）。
- checkpoint 复选/网络诊断：`generalization_g1c_checkpoint_recheck_v01/`、
  `generalization_g1c_external_1b_v01/`。
- 大文件处置：8 个 .pt（3 组 model/optim + 2 个 q 张量，≈61G）已于
  2026-09-25 删除，恢复方法见 `docs/CLEANUP_G1C_LARGE_FILES_20260925.md`。
- F0 及任何 GTTR 轮次不依赖本目录任何文件（协议与代码零引用，2026-09-25 核验）。
