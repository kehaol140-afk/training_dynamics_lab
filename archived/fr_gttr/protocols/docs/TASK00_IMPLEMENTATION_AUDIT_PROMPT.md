# TASK-00 — MiniMind 实现审计（中文版本）

项目根目录：`/home/b205/projects/fluctuation_response`

正式协议：`docs/Transformer_AdamW_Fluctuation_Response_ResearchPlan_AI_v2.md`

MiniMind：`minimind/`

新实验代码：`research/`

## 当前任务

**只审计，不修改训练代码，不运行正式训练。**

生成：`research/IMPLEMENTATION_AUDIT.md`

必须基于当前服务器真实源码回答：

1. `data loading -> forward -> loss -> backward -> accumulation -> AMP/unscale -> clipping -> AdamW.step -> learning-rate update -> zero_grad -> checkpoint` 的真实执行顺序、文件、函数和关键位置。
2. AdamW class、betas、eps、weight_decay、LR、学习率更新方式、gradient clip；checkpoint 是否保存/恢复 optimizer `m/v/step`。
3. Python / NumPy / Torch CPU / Torch CUDA / DataLoader / sampler / worker / dropout / AMP / compile 等随机源；当前 checkpoint 是否保存/恢复。
4. `pretrain_t2t_mini.jsonl` 如何进入 batch；当前实现能否从同一 checkpoint 重放相同 future batch stream。
5. 对完整状态 `Z_t={theta,m,v,optimizer_step,learning_rate_state,global_step,data/sampler_state,Python RNG,NumPy RNG,Torch CPU RNG,Torch CUDA RNG,AMP}` 逐项标记 `PASS/PARTIAL/MISSING`。
6. **只分析**未来 `g <- g +/- epsilon*q` 的合理插入点，特别说明 accumulation、AMP unscale、clip 和 optimizer moment update 的相对位置。
7. 分析 common-random-number paired branch 是否可行。
8. 最后给出：`E0_READY = YES/NO`。若 NO，只列最少基础设施。

## 禁止事项

- 不修改 `minimind/`。
- 不实现 V2A/V2B。
- 不实现在线控制器。
- 不在没有源码证据时猜测 AMP、clip、resume 行为。
