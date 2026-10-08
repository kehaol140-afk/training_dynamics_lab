# F3 OpenCode Takeover Audit（运行态 + 项目态接管审计）

- 审计时间：2026-09-28 23:48–23:55 CST（本文件生成本轮，只读审计，未启动任何任务、未 kill 任何进程、未修改任何既有输出）。
- 审计者：opencode（接管 Claude/Kimi 会话中断后的 F3 compatibility / reproduction 工作）。
- 触发原因：上一会话报 `Background shell command didn't finish before the previous session ended`。

---

## 1. 系统状态

| 项 | 值 |
|---|---|
| 平台 | WSL2（DESKTOP-I00SAUH，kernel 6.18.33.2-microsoft-standard-WSL2，systemd enabled） |
| 当前 boot | boot 0：2026-09-28 22:53:07 CST 起（`/proc/uptime` ≈ 57 min，journalctl --list-boots 确认） |
| 上一 boot | boot −1：16:46:19 → **21:53:44 结束**（约 5h08m） |
| 更早 boots | 16:22、16:24、16:41 三次短命 boot（WSL 冷启动抖动） |
| `who -b` / `last` | 无记录 / 命令不存在（WSL 特性，非权限问题） |
| journalctl | 有权限，已读 |
| OOM killer | 无记录 |
| NVIDIA Xid | 无（无 NVRM/Xid 条目） |
| GPU 异常 | 仅 2 条 dxgkrnl WARNING（22:59:04、23:34:00，PID 3960，WSL GPU 半虚拟化 sync-object 等待警告，非致命，进程已消失） |
| 内存 | 125Gi total，118Gi free（无压力） |
| 磁盘 | / 已用 829G/1007G，剩 128G；`outputs/f3_curvature_mechanism` 49G；`outputs/f3_compatibility` 496K |

**结论：Claude 会话中断伴随一次真实的 host 级重启**（WSL VM 于 21:53:44 终止，22:53:07 重新启动，中间 59 分钟停机）。此外 21:53:05（重启前 39 秒）刚完成一次 commit（ae2ab31），21:51:08 启动的 RQ3 任务随 VM 一起死亡。

## 2. 运行中的任务

**没有任何 Claude/F3/compatibility 相关进程存活。**

- `ps` 全表扫描：无 python/torch/f3/compat 进程；仅存的是本 opencode 会话、VS Code server、以及另一用户 yangj 的 modelscope 下载（PID 3103，与本项目无关，勿动）。
- `nvidia-smi`：GPU0 仅 Xwayland 607MiB，GPU1 0MiB，无计算进程；pmon 无活动。
- tmux / screen：均无会话。
- systemd user unit `f3-remaining`（`scripts/f3_status.sh` 中提及）：**不存在**（unit 文件未创建），无自动重启/重复启动风险。
- 60 秒双查（23:49 与 23:50 两次 stat）：`f3_run.log`、`driver2.log`、`rq3_2042_26.out` 大小与 mtime 零变化 → 无 ACTIVE_PROGRESS。
- load average 1/5/15 min = 0.34 / 2.62 / 9.17：15 分钟前的重负载（RQ3 第 4 次尝试）已随会话结束终止。

最后一批任务（第 4 次 RQ3 尝试）的死亡方式：driver2.log 23:24:10 写入两条 START 后无任何 END 行 → driver 及两个 python 子进程被整体杀掉（会话 teardown），与报错信息一致。setsid/nohup 意图未能实现存活。

## 3. Git 状态

- **HEAD：`130b8f7` "F3 RQ1/RQ2 EARLY cell raws (integrity-checked vs frozen reference)"**（2026-09-28 23:23:39 提交，即重启后、最后一次 RQ3 启动前 40 秒）。
- 提交链（新→旧）：`130b8f7` → ae2ab31(status 脚本) → 93cf13d(detached driver) → d9e10ac(raw 完整性检查器) → 22113fd(RQ3 masked arms 实现) → 80241f4(**frozen-reference gate PASS**) → 7510aec(**CURRENT_ENV_REFERENCE_ESTABLISHED**) → 577cb43(compat 分析) → **d63a8e1(amendment，先于一切 compat 结果)** → a2e49f9(F3 BLOCKED) → 658301c → f51ce82 → 420d394(F2.5) …
- working tree：
  - `M outputs/f3_curvature_mechanism/driver2.log`（仅 2 行 START 时间戳被新一轮覆盖，diff 已核对）
  - `M outputs/f3_curvature_mechanism/f3_run.log`（仅追加 2 行 RQ3 banner）
  - `?? raw/rq1_*.state.pt, rq2_*.state.pt`（4 个 11–14GB 大文件，刻意不入库）
  - `?? rq3_2042_26.out, rq3_2043_26.out`（仅含 4 次 banner，无效结果但**保留勿删**）
- **amendment 时序合规**：`d63a8e1`（amendment 文档+gate 代码改造）在 commit 顺序上先于 577cb43/7510aec/80241f4 全部 compat 结果；且 7510aec 内 environment_manifest.json 记录生成时 HEAD=d63a8e1。R0/R1 阈值确系数据前冻结。
- 未做任何 commit（本轮禁止）；无 push。

## 4. Stage 表

| Stage | Status | Evidence | Running PID | Artifact | Notes |
|---|---|---|---|---|---|
| F3 blocked report | COMPLETE | a2e49f9；summary.json status=BLOCKED_AT_REPRODUCTION_GATE | — | docs/F3_CURVATURE_MECHANISM_REPORT.md | 负结果完整保留；ordering bug 修复 + 同日 G1=G2=0 证据在内 |
| F3 reproduction amendment | COMPLETE | d63a8e1，先于结果 commit | — | docs/F3_REPRODUCTION_GATE_AMENDMENT.md | R0 4 gates + R1 3 gates 冻结 |
| environment manifest | COMPLETE | 09-27 09:21 生成，7510aec 提交 | — | outputs/f3_compatibility/environment_manifest.json | torch 2.11.0+cu128, driver 591.44, tf32_matmul=False |
| R0 EARLY 2042:26 ×3 | COMPLETE | rep1/2/3 逐位一致（spread=0.0） | — | raw/compat_2042_26_rep{1,2,3}.json | 9 runs 共 30min，独立进程 |
| R0 EARLY 2043:26 ×3 | COMPLETE | 同上 | — | raw/compat_2043_26_rep{1,2,3}.json | |
| R0 MIDDLE 2042:64 ×3 | COMPLETE | 同上 | — | raw/compat_2042_64_rep{1,2,3}.json | |
| R0 repeatability analysis | COMPLETE | A=0.0，B=1.0，C=3/3，D=×53.79/×8.59，4/4 PASS | — | repeatability.csv | |
| R1 EARLY 2042 B/C | COMPLETE | P1: H16=31.59, H32=86.29 ≥5 | — | phenomenology_gate.csv | |
| R1 EARLY 2043 B/C | COMPLETE | P1: H16=5.447, H32=743.95 | — | phenomenology_gate.csv | |
| R1 MIDDLE 2042 B/C | COMPLETE | G_C/G_B(16)≈0.92（MIDDLE 弱放大，与 archive 定序一致）；P2=31.19 | — | phenomenology_gate.csv | branch_order False 为预期 phenomenology，非 gate |
| R1 phenomenology gate | COMPLETE | P1/P2/P3 全 True（30/30 行） | — | phenomenology_gate.csv | P3: H1 rel=0.0（9/9），H2 ratio 9/9 在 [0.9,1.1] |
| current-env frozen reference | COMPLETE | bundle SHA256 `6f2650e0…8bdf6` | — | frozen_reference/（7 文件）+ CURRENT_ENV_REFERENCE_SHA256.txt | 取 rep1，规则数据前冻结 |
| F3 gate vs frozen ref（复航检查） | COMPLETE | 3/3 cells PASS（80241f4） | — | raw/gate_frozenref_{2042_26,2043_26,2042_64}.json | G1/G2 容差 1e-12，同环境可达 |
| RQ1 alpha sweep EARLY ×2 | COMPLETE | 各 32 h-steps 有限值（10:08–10:43 / 10:0x–10:43）；json 5519 行入库；state.pt 在位 | — | rq1_2042_26.*, rq1_2043_26.* | manifest e40dd…/c3862… 与 frozen 一致；130b8f7 注明已 integrity-check |
| RQ1 MIDDLE 2042:64 | NOT_STARTED | 无文件 | — | — | driver 计划中排在 EARLY 之后 |
| RQ2 source-off EARLY ×2 | COMPLETE | 各 32 h-steps；json 2920 行入库 | — | rq2_2042_26.*, rq2_2043_26.* | state.pt 14.5G×2 未入库（正常） |
| RQ2 MIDDLE 2042:64 | NOT_STARTED | 无文件 | — | — | |
| RQ3 module ablation EARLY ×2 | **FAILED** | 4 次启动（09-28 18:09、19:55、21:51、23:25），全部只打印 banner，0 个 h-step 行、无 raw json/state 输出 | 无（全部死亡） | rq3_2042_26.out / rq3_2043_26.out（仅 4×banner，INVALID 但保留） | #3 死于 21:53:44 VM 重启；#4 死于会话结束（23:25 启动，23:34 仍有 GPU 活动，~23:35–23:45 间被杀）。#1/#2 存续窗口内（最长 ~1h45m）也未出现 h=1（RQ1 h=1 仅 74s）→ **不排除 RQ3 每步极慢或早期 hang，需人工决定先做单 cell 计时探针** |
| RQ3 MIDDLE | NOT_STARTED | 无文件 | — | — | |
| RQ4 spectrum（最贵） | NOT_STARTED | 无文件 | — | — | prereg §10 排最后 |
| RQ5 decomposition | NOT_STARTED | 无 postprocess 输出（RQ1 数据已具备） | — | — | 定义为 RQ1 数据免费计算+后处理 |
| F3 final report | NOT_STARTED | — | — | — | |

## 5. Partial outputs 可复用性

**可复用（全部已 commit，勿重算勿覆盖）：**
- `outputs/f3_compatibility/` 整套（manifest、9 raw runs、两份 gate CSV、frozen reference、SHA256）；
- `gate_frozenref_*.json`（复航 gate PASS 证据）；
- RQ1/RQ2 EARLY 四组 raw json + out + state.pt（state.pt 未入库但完整在磁盘）；
- blocked 报告与 amendment。

**无效（保留、不删除、不覆盖）：**
- `rq3_2042_26.out`、`rq3_2043_26.out`：仅含重复 banner，无任何数据行。重启 RQ3 时应以 append 方式或新文件继续，保留现有内容作 provenance。

**风险文件：** 无 `.lock`/`.pid`/`nohup.out` 残留（driver 用日志非锁文件）。

## 6. 重复运行风险

当前为 **零**：无存活进程、无 systemd unit、无 cron/tmux/screen。唯一注意点：将来重启 RQ3 时 `run_f3_remaining.sh` 的 `: > "$LOG"` 会截断 driver2.log——建议（将来，经人工同意后）改为 append。

## 7. 推荐动作

**BLOCK_FOR_HUMAN_REVIEW**

理由：
1. 无既有任务可等待（WAIT_FOR_EXISTING_JOB 不适用）；
2. compatibility（R0+R1+frozen reference+复航 gate）已全部 COMPLETE 且 PASS，无 missing repeats（RESUME/START_COMPATIBILITY 不适用）；
3. 下一步是 F3 主实验 RQ3（module ablation），但它已连续 4 次异常终止且从未产出第一个 h-step——需要人工决定：(a) 运行载体（tmux/systemd-run --scope，避免再次死于会话 teardown）；(b) 先跑单 cell RQ3 计时探针确认每步耗时/hang 与否；(c) 磁盘预算（MIDDLE 三阶段 raws 预计再增 ~40G，剩 128G）。

在此之前：不启动任何实验，不动 legacy（legacy 自 09-26 后零修改，已核实），不 push。

## 8. 附：关键时间线（2026-09-28）

| 时刻 | 事件 |
|---|---|
| 16:46–21:53 | boot −1（Claude 工作期） |
| 18:09 | RQ3 尝试 #1（直接启动）→ 无输出 |
| 19:53–19:54 | 提交 93cf13d detached driver；19:55 RQ3 尝试 #2 → 无输出 |
| 21:51:08 | RQ3 尝试 #3（driver 版本已入库） |
| 21:52:52–21:53:05 | f3_status.sh + 提交 ae2ab31 |
| **21:53:44** | **WSL VM 终止（= 会话中断的真实原因），杀死 #3** |
| 22:53:07 | WSL VM 重启（boot 0） |
| 23:23:39 | 提交 HEAD 130b8f7（RQ1/RQ2 EARLY raws 入库） |
| 23:24:10 | driver2 启动，RQ3 尝试 #4（2042:26→GPU0，2043:26→GPU1） |
| 23:25:06/08 | 两 cell banner 打印，此后零输出 |
| 23:34 | 最后一次 GPU 活动（dxgkrnl warning, PID 3960） |
| ~23:35–23:45 | 会话结束，后台任务整体被杀（即报错来源） |
| 23:35 | 当前 opencode 会话启动 |
| 23:48–23:55 | 本只读审计 |
