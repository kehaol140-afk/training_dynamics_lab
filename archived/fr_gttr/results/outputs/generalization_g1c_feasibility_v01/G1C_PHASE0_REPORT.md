# G1C Phase 0 Report — checkpoint restoration gate (step950000)

- gate: FR_G1C_1B_FEASIBILITY_GATE_V01 Phase 0
- verdict: **PASS**
- wall time: 38.3s

## Files (size + SHA256)

| file | size | expected | size_ok | sha256 |
|---|---|---|---|---|
| config.yaml | 163724 | None | True | babefbaa189465ab9c423c178b4a367d23dd6707da1999702df25efc7403a7da |
| model.pt | 5939703038 | 5939703038 | True | 6e7b1690215915dfb9cf697eefd2a22aea1404752ad1cdc531d4c18cae04e40b |
| optim.pt | 11879442406 | 11879442406 | True | ffcb8374c53f635df2fdd22aab11aa1067f19d1285fc6074f2cffd141ccacdb2 |
| train.pt | 17036 | None | True | 7ce75e9a84db628930650b35c03769fbdf7959c67a25ae6851bd0d981fbd8989 |

## Checks

| check | ok | detail |
|---|---|---|
| file:config.yaml | True | size=163724 expected=None sha256=babefbaa189465ab9c423c178b4a367d23dd6707da1999702df25efc7403a7da |
| file:model.pt | True | size=5939703038 expected=5939703038 sha256=6e7b1690215915dfb9cf697eefd2a22aea1404752ad1cdc531d4c18cae04e40b |
| file:optim.pt | True | size=11879442406 expected=11879442406 sha256=ffcb8374c53f635df2fdd22aab11aa1067f19d1285fc6074f2cffd141ccacdb2 |
| file:train.pt | True | size=17036 expected=None sha256=7ce75e9a84db628930650b35c03769fbdf7959c67a25ae6851bd0d981fbd8989 |
| model_restore_keys | True | n_tensors=131 n_elements=1484916736 (expected 1484916736) match=True |
| optim_state_entries | True | state_entries=131 params=131 |
| optim_step_values | True | step unique=[950001.0] min=950001.0 max=950001.0 (expect all == 950000) |
| optim_groups | True | [{"n_params": 130, "lr": 0.00027699376575216877, "betas": [0.9, 0.95], "eps": 1e-08, "weight_decay": 0.1, "initial_lr": 0.0004, "max_grad_norm": 1.0, "amsgrad": false, "foreach": null, "fused": null, "first_name": "_fsdp_wrapped_module.transformer.blocks.0.att_proj.weight", "last_name": "_fsdp_wrapped_module.transformer.ln_f.weight"}, {"n_params": 1, "lr": 0.00027699376575216877, "betas": [0.9, 0.95], "eps": 1e-08, "weight_decay": 0.0, "initial_lr": 0.0004, "max_grad_norm": 1.0, "amsgrad": false, "foreach": null, "fused": null, "first_name": "_fsdp_wrapped_module.transformer.wte.weight", "last_name": "_fsdp_wrapped_module.transformer.wte.weight"}] |
| optim_m_v_finite | True | |m|=0.0153637 mean(v)=6.74971e-12 |
| train_global_step | True | global_step=950000 tokens=1992294400000 |
| train_rng | True | rng flags={'has_rng_cuda': True, 'has_rng_numpy': True, 'has_rng_python': True, 'has_rng_torch': True} |
| scheduler_state | True | has_scheduler_state=False -> constant-lr fallback lr=0.0002769937658 (protocol 6.5) |
| optim_meta_vs_config | True | betas_ok=True eps_ok=True wd_set=[0.0, 0.1] config_optim_keys=[] |

## Notes

- train.pt contains NO scheduler state -> protocol 6.5 constant-lr fallback, lr = restored param_groups value 0.0002769937658.
- optim.pt: 2 param groups (wd=0.1 for blocks/lm_head/ln_f, wd=0.0 for wte) -> versioned adapter g1c_adamw_tangent_step in use (protocol 6.8).
- This is a restoration-feasibility gate, not a science result.