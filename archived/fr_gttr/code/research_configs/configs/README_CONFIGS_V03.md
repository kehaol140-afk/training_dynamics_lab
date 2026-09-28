# v0.3 配置说明

`e0_extended_v03.json`：默认约 256 hidden / 4 layers，主要验证重放基础设施能否在明显大于 0.53M smoke 的配置下继续成立。

`e1_pilot_v03.json`：同样采用中等 pilot 配置，默认 R=8；probe 使用子集末尾 8 个样本，并从 stream batch 中显式排除。

`subset_lines=4096` 只是 pilot 加速设置，不应被解释为正式训练数据分布。后续正式阶段检查需要改为全数据或明确的冻结抽样方案。
