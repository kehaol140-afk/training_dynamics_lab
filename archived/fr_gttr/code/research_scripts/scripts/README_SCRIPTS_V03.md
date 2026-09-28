# v0.3 脚本说明

- `check_v03.py`：只检查文件是否完整，不跑 GPU。
- `e0_extended_validate.py`：更大配置、更长 horizon 的无扰动精确重放。
- `e1_natural_fluctuation.py`：构造 q，采样 R 条随机流，计算 sigma_q。

旧版 v0.2 的 smoke 已有 PASS 记录，本包不要求重跑旧 smoke。
