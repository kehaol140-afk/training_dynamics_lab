# v0.4 脚本说明

- `check_v04.py`
  - 只检查文件是否安装完整，不占 GPU。

- `prepare_mid_checkpoint_v04.py`
  - 建立明确的 MID pilot checkpoint；
  - probe/eval 永不进入训练 pool。

- `e1_formal_v04.py`
  - 正式协议对齐 E1；
  - R=16；
  - H=1/8/32；
  - 不人工 perturb。

- `e2_preflight_v04.py`
  - 检查 E1 正式结果是否允许进入 E2。

- `e2_epsilon_linearity_v04.py`
  - 真正执行 baseline / +epsilon / -epsilon；
  - rho=0.10/0.25/0.50/1.00；
  - 只在第一个 optimizer step 注入一次；
  - 输出 odd_error / chi / clipping。
