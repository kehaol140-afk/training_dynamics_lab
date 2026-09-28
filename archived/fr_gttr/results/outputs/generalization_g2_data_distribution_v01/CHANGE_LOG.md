# CHANGE_LOG — FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01

> 本日志在实验执行后汇总记录；预注册文件（G2_PROTOCOL.md /
> experiment_config.json）在任何结果打开前已冻结，未因结果改动任何阈值、
> 口径或规则。

## §1 预注册与冻结（2026-09-16，任何结果前）

- 新建 `research/configs/g2_data_distribution_generalization_v01.json`
  （G1a 模型/训练字段逐位克隆；B12 四级阈值 + 三个 flag tier 冻结；
  smoke 状态按 B9 定为 seed 2042 / EARLY / step 26，与 G1a 的 MIDDLE 不同）。
- 新建 `outputs/generalization_g2_data_distribution_v01/G2_PROTOCOL.md`（17 节）。
- 新建模块 `research/src/fr/g2_data_distribution_generalization_v01.py`：
  判定分层、cluster bootstrap、pool/partition/manifest 工具、报告渲染；
  主统计口径一律 import 冻结 G1a/a0c，未重新定义。
- 新建 runner `research/scripts/g2_data_distribution_generalization_v01.py`
  （G1a runner 克隆 + `dataset` 阶段 + 双口径机制分析 + D0/D1 对比 + 合并式
  future_stream_manifest）。
- 新建单测 `research/tests/test_g2_data_distribution_generalization_v01.py`
  （31 项；含 pool/partition 确定性、判定层、bootstrap 标注、manifest 校验、
  G1a config 逐字段守卫、D0 哈希守卫）。

## §2 执行记录（2026-09-16，全部一次通过）

- `preregister`：ENVIRONMENT / SOURCE_AND_INPUT_HASHES / run_manifest 写出
  （dataset 阶段前 d1_pool 标记 MISSING，属预期）。
- `dataset`：D0 sha256 核验通过（1e657af9…）；D1 sha256 核验通过
  （82b19375… = HF LFS oid，1,413,103 条记录）；schema 双侧均为
  `{"text": str}` → 无需 adapter；token 统计：D0 均值 114.9 tokens（73%
  截断），D1 均值 126.0（100% 截断，文本显著更长）；重复/重叠诊断：
  D0 全文件 0 精确重复行，D1 820 行（0.058%）；13-gram shingle Jaccard
  均值 1.6e-7（语料几乎不相交）；冻结 `data/d1_pool.jsonl`（4096 条，
  seed 60202601）与 `partition_g2_d1.json`（seed 60202603，32/128/3936）。
- `train`：3 seeds × 3 states，参数量断言 26,681,728 通过；轨迹不触
  probe/eval；dynamics 逐步记录。
- `theta`：2042 先行，2043/2044 随后；future_stream_manifest 以合并方式
  累积至 9 states × 4 streams；全部 rho0-noop 通过。
- `smoke`（seed 2042 / EARLY / step 26）：18/18 检查通过，
  H=1 解析 vs FD 相对误差 1.83e-3；含 B9 专项
  train-loader deterministic replay 与 dataset/pool/partition/state-metadata
  哈希核对。
- `tangents`：36/36 baseline replay 位级精确（shadow 哈希 + X 全等）。
- `mechanism`：seed 2042 × 3 stages × 4 streams；H=1 恒等 gap = 0.0。
- `analyze`：判定 STRONG_DATA_REPLICATION（四级全过，阈值未动）；
  机制双口径（stream-first + official-cell）均报告；D0/D1 matched 对比表、
  cluster bootstrap（20000 resamples, rng 20260916）、报告与 summary 写出。
- `audit`：SHA256SUMS 两段式（255 文件入包 / 25 个大文件仅登记哈希：
  bank checkpoints、theta q、raw D1 语料）；自校验 + 读回验证通过；
  包 SHA256 `96d88d22b1b42670e0096d57c6e3be46180b91581817bdf0e330dcd32ca904f1`。

## §3 实现层修正（执行前发现并修复，未影响任何科学定义）

- runner 草稿中 smoke 分支一处 `stream_seed` 笔误（`int(rec["stream_id"])*0 +
  int(rec["stream_seed"])`）在首次执行前修正为 `int(rec["stream_seed"])`。
- G2 模块初版未含 E2 manifest 校验器；按冻结 `validate_g1a_e2_manifest`
  逐检查镜像新增 `validate_g2_e2_manifest`（仅协议 tag 不同，G1a tag
  会被显式拒绝——有单测覆盖）。
- 单测两处初版失败并修正：(a) bootstrap 退化输入（全部同值）导致 PI
  退化，改用异构合成数据；(b) G1a config model 块含规模实验注记键
  （anchor_unique_params 等 4 个），守卫测试改为科学字段子集逐位相等。
- 审计重入缺陷修复：第二次运行 `audit` 时，旧 bundle 文件被 rglob 扫进
  包含清单并在 tar 创建期间被当作普通文件自包含（自校验随即失败）；
  扫描跳过集现显式排除 `g2_data_distribution_audit_package.tar.gz` 自身
  （G1a 仅单次运行未暴露此问题）。修复后审计复跑通过。
- 终版审计：CHANGE_LOG 全部条目写完后复跑一次 `audit`（256 文件入包 /
  25 个大文件仅登记哈希），自校验 + 读回验证通过；最终 bundle SHA256
  `9553ffa5a4783e273487dc7b3df7cc6b1143cedec4e13ab5bc3d7c88491157a7`
  （见 AUDIT_BUNDLE_SHA256.txt）。自引用说明：该 bundle 覆盖除本行哈希
  数字本身外的全部文件字节；审计自身输出（SHA256SUMS.txt /
  AUDIT_BUNDLE_SHA256.txt / bundle）按两段式规则不入包。

## §4 异常 / 资源

- 无 OOM、无 retry、无 RESOURCE_LIMIT、无 CONTRACT_INCOMPATIBLE。
- GPU0 与外部任务全程共享（外部进程未受影响、未触碰 GPU1）。
- D1 获取路径说明：`pretrain_hq.jsonl` 已从 HF 主分支删除
  （2026-03-26），实验使用删除前官方 revision
  `3a9bbe611244811e45b96661a84047f112c8ed01`（2026-01-17），LFS 哈希
  核验一致；详见 DATASET_PROVENANCE.md。

## §5 冻结声明

本实验未修改任何 A0A–A0G / A0H / G1a / G1b / G3 冻结制品；未启动
1B / G4 / A0G-lite / 新 optimizer / 任何 sweep / 新算法实验。
