# A0H-1 审计包 —— 命令、环境、版本

## 环境
- conda env: fr_minimind
- torch: 2.11.0 / CUDA 13.0
- GPU: RTX 5090 ×2（dev 检查用 GPU1 共享；正式计时需独占 gate，本轮未执行）
- PYTHONPATH: research/src

## 命令（按执行顺序）
1. CPU 合成验证: PYTHONPATH=research/src python -m unittest research.tests.test_a0h_adjoint_v01 -v   # 4/4 OK
2. GPU dev 等价+fp32 探针: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=research/src python research/scripts/a0h_adjoint_dev_check_v01.py --device cuda:0 --budget-seconds 1200 --probe-fp32-hvp
   （计时脚本 a0h_cost_timing_v01.py 未运行：协议 §4 gate = 等价 PASS 后才开始）

## 冻结输入（只读，哈希见 hashes/frozen_input_hashes.txt）
- research/configs/e3c_confirmatory_v07.json
- outputs/fr_decision_utility_v01/{lambda_selection.json,level1_local_derivative.csv,level1_finite_action.csv,level2_action_predictions.csv,level3_decision_metrics.csv,data_group_definition.json,validation_split.json}
- outputs/e3_confirmatory/v07/ bank（checkpoint/q/manifest/partition，经 runner 内 sha 校验）
