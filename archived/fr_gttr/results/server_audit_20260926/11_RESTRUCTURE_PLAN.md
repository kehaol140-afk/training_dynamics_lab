# 11 — 未来重构方案（仅建议 + 安全命令草案；现在不执行任何 rename/move/delete）

## 1. 目标布局（建议）

```
<new_parent>/                        # 例如 /home/b205/projects_fr（或保留现名）
├── legacy_fluctuation_response/     # 现 fluctuation_response 整体降级为 legacy（原样保留，只读化）
│   ├── （原目录结构不动：research/ docs/ outputs/ minimind/ ...）
├── minimind/                        # 【可选】将 minimind 提升为顶层独立目录并 git init 固定副本
│   └── （补文件级 SHA256 manifest）
├── <new_project>/                   # 新研究项目（新 git 仓库）
│   ├── research/{src,scripts,configs,tests}/   # 从 legacy 选择性 vendor：
│   │       #   - 指标源 a0c_adamw_response_propagation_v01.py（import-only）
│   │       #   - theta_protocol.pyc + 其加载约定
│   │       #   - rng/stream/batch/replay 基础设施
│   │       #   - G 线 config 模板 + runner 三件套模式
│   ├── docs/
│   └── outputs/
└── FULL_SERVER_AUDIT_20260926/      # 本审计（已存在于现父目录）
```

原则：
1. **legacy 目录一律只读化**（chmod -R a-w 或文件系统快照），保住 88 条实验的可追溯性；
2. 新项目**复制/vendor**需要的模块而不是修改 legacy 路径（冻结纪律 §13b 延续）；
3. minimind 若提升，先做文件级 manifest 再移动。

## 2. 分阶段安全命令草案（每步先验证后执行；执行前逐条人工确认）

### 阶段 0 —— 保护（最高优先，可立即做）
```bash
cd /home/b205/projects/fluctuation_response
# 0.1 把 11 项 untracked（F0/F1/CALFIX 代码与包）入库
git add FR_GTTR_F0_CALFIX_Revalidation_v1.md \
        GTTR_F0_Implementation_Consistency_Package_v1.zip \
        research/scripts/f0_calfix_run_v01.py research/scripts/f0_diag_small_T.py \
        research/scripts/f0_ic_adapter_v01.py research/scripts/f0_terminal_rule_run_v01.py \
        research/scripts/f1_closure_run_v01.py research/src/fr/f0_terminal_rule_v01.py \
        research/tests/test_f0_clipjvp_audit_v01.py research/tests/test_f0_terminal_rule_v01.py tools/
git commit -m "F0/CALFIX/F1: 前沿实验代码与独立校验包入库（审计 2026-09-26 补录）"
# 0.2 首次推送（需先确认 SSH 凭据可用；不 force）
git remote -v                      # git@github.com:kehaol140-afk/fluctuation-response.git
git push -u origin master
# 0.3 minimind 文件级清单（不动文件本体）
cd minimind && find . -type f -not -path "./dataset/__pycache__/*" | sort | \
  xargs sha256sum > ../minimind_file_manifest.sha256 && cd ..
```

### 阶段 1 —— 大文件归档（先验证后移动；目标 ~36G → 冷存储）
```bash
# 模板（对每个候选目录执行；先做只读验证）
D=outputs/generalization_g1b_100m_v01/bank
(cd "$D" && sha256sum -c SHA256SUMS.txt)            # 1. 先验证完整性（若目录内有 SUMS）
tar -czf /archive/fr/g1b_bank_states.tar.gz "$D"    # 2. 打包到冷存储
sha256sum /archive/fr/g1b_bank_states.tar.gz        # 3. 记录包哈希
tar -tzf /archive/fr/g1b_bank_states.tar.gz | head  # 4. 可列验证
# 确认以上全部成功后才可考虑删除原大 .pt（结论件/manifest/SHA256SUMS 永不删）
# 同法处理：a0k 快照(~14G)、a0n runs(~7G)、pretrain_hq.jsonl(1.6G)、
#          e3_replication 大件、mv_theory_final/gradient_g.pt(1.1G)
```
> 建议遵循项目已有先例：docs/CLEANUP_G1C_LARGE_FILES_20260925.md 的"SHA256 锚点 + 重获方法 + 代码零依赖核查"三件套。

### 阶段 2 —— 目录重组（一次性、可回滚）
```bash
# 2.1 停止一切训练/IO；确认无进程占用
lsof +D /home/b205/projects/fluctuation_response | head    # 应为空
# 2.2 legacy 降级（重命名而非删除）
mv /home/b205/projects/fluctuation_response \
   /home/b205/projects/legacy_fluctuation_response
# 2.3 只读化（可选但强烈建议）
chmod -R a-w /home/b205/projects/legacy_fluctuation_response/outputs/a0_fr_audit
chmod -R a-w /home/b205/projects/legacy_fluctuation_response/outputs/e3_confirmatory
# 2.4 修正 legacy 内 Git remote 语义（可选：改名为 legacy remote）
git -C /home/b205/projects/legacy_fluctuation_response remote set-url origin <新地址或保留>
```

### 阶段 3 —— minimind 独立化（可选）
```bash
mv /home/b205/projects/legacy_fluctuation_response/minimind /home/b205/projects/minimind
git -C /home/b205/projects/minimind init && git -C ... add -A && git -C ... commit -m "vendor: minimind-3 snapshot (zip sha256 c06cf0b2...)"
# 若新项目仍需 legacy 相对路径引用，用 symlink 过渡：
# ln -s /home/b205/projects/minimind /home/b205/projects/legacy_fluctuation_response/minimind
```

### 阶段 4 —— 新项目脚手架
```bash
mkdir -p /home/b205/projects/<new_project>/{research/{src/fr,scripts,configs,tests},docs,outputs}
cd /home/b205/projects/<new_project> && git init
# vendor 冻结指标与基础设施（复制，不引用跨仓库相对路径）：
cp /home/b205/projects/legacy_fluctuation_response/research/src/fr/a0c_adamw_response_propagation_v01.py research/src/fr/
cp /home/b205/projects/legacy_fluctuation_response/research/src/fr/theta_protocol.pyc research/src/fr/
cp /home/b205/projects/legacy_fluctuation_response/research/src/fr/{rng,stream_rng,batch_stream,batch_manifest,full_state,perturbation,native_adamw_guard,native_schedule,minimind_v03_runtime}.py research/src/fr/
cp -r /home/b205/projects/legacy_fluctuation_response/research/configs research/configs_legacy_templates
git add -A && git commit -m "vendor: FR frozen metric + replay infra from legacy (audit 20260926)"
```

## 3. 绝对不要做的事（红线，与项目 §13b/§14 一致）

1. 不删除/重命名 `outputs/e3_confirmatory/v07/`（56 态 bank）与 `outputs/a0_fr_audit/` 结论件；
2. 不重跑 A0C/A0D/A0E2/A0F/A0F2/H1/mv_theory_consistent（DO NOT REDO）；
3. 不改任何冻结 runner/summary/contract/SHA256SUMS；
4. 不把 G1c 已删大文件视为"需要恢复"（有 SHA 锚点，按 CLEANUP 文档可重获）；
5. 不在 legacy 内做任何"顺手清理"（包括空占位文件 r/rm/project_tree.txt——除非单独一轮专门清理并记录）；
6. 归档删除只针对阶段 1 清单中"验证通过且包哈希已记录"的大 .pt/.jsonl。

## 4. 建议的执行顺序与门槛

| 顺序 | 动作 | 门槛 |
|---|---|---|
| 1 | 阶段 0（git 入库 + push + minimind manifest） | 无风险，立即可做 |
| 2 | 阶段 1（归档 36G） | 磁盘需 <247G 剩余之外另有冷存储空间 |
| 3 | 阶段 2（legacy 化） | 确认新项目已立项、无进程占用 |
| 4 | 阶段 3/4（minimind 独立 + 新项目） | 新项目设计冻结后 |
