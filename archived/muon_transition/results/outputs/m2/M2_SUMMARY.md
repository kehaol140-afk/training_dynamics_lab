# M2 summary — optimizer × mixture grid (seed 2030, B0=2^26, dev V_Q selection)

## 1. Experiment table (all cells)

| run_id | arm | p | mult | lr | wd | V_Q | NLL_A | NLL_B | ok |
|---|---|---|---|---|---|---|---|---|---|
| m2_adamw_lr0.5_wd0.01_p0.25_s2030 | adamw | 0.25 | 0.5 | 0.00015 | 0.01 | 3.5132 | 3.4443 | 3.5822 | True |
| m2_adamw_lr0.5_wd0.02_p0.25_s2030 | adamw | 0.25 | 0.5 | 0.00015 | 0.02 | 3.5142 | 3.4452 | 3.5832 | True |
| m2_adamw_lr1.0_wd0.01_p0.25_s2030 | adamw | 0.25 | 1 | 0.0003 | 0.01 | 2.9021 | 2.8781 | 2.9261 | True |
| m2_adamw_lr1.0_wd0.02_p0.25_s2030 | adamw | 0.25 | 1 | 0.0003 | 0.02 | 2.9028 | 2.8788 | 2.9269 | True |
| m2_adamw_lr2.0_wd0.01_p0.25_s2030 | adamw | 0.25 | 2 | 0.0006 | 0.01 | 2.6659 | 2.6537 | 2.6781 | True |
| m2_adamw_lr2.0_wd0.02_p0.25_s2030 | adamw | 0.25 | 2 | 0.0006 | 0.02 | 2.6650 | 2.6537 | 2.6763 | True |
| m2_adamw_lr4.0_wd0.01_p0.25_s2030 | adamw | 0.25 | 4 | 0.0012 | 0.01 | 2.6688 | 2.6515 | 2.6861 | True |
| m2_adamw_lr4.0_wd0.02_p0.25_s2030 | adamw | 0.25 | 4 | 0.0012 | 0.02 | 2.6689 | 2.6506 | 2.6872 | True |
| m1_adamw_lr0.5_wd0.01_p0.5_s2030 | adamw | 0.5 | 0.5 | 0.00015 | 0.01 | 3.4901 | 3.2189 | 3.7613 | True |
| m1_adamw_lr0.5_wd0.02_p0.5_s2030 | adamw | 0.5 | 0.5 | 0.00015 | 0.02 | 3.4912 | 3.2200 | 3.7625 | True |
| m1_adamw_lr1.0_wd0.01_p0.5_s2030 | adamw | 0.5 | 1 | 0.0003 | 0.01 | 2.8699 | 2.6732 | 3.0667 | True |
| m1_adamw_lr1.0_wd0.02_p0.5_s2030 | adamw | 0.5 | 1 | 0.0003 | 0.02 | 2.8707 | 2.6737 | 3.0677 | True |
| m1_adamw_lr2.0_wd0.01_p0.5_s2030 | adamw | 0.5 | 2 | 0.0006 | 0.01 | 2.6412 | 2.4881 | 2.7943 | True |
| m1_adamw_lr2.0_wd0.02_p0.5_s2030 | adamw | 0.5 | 2 | 0.0006 | 0.02 | 2.6420 | 2.4888 | 2.7951 | True |
| m2_adamw_lr4.0_wd0.01_p0.5_s2030 | adamw | 0.5 | 4 | 0.0012 | 0.01 | 2.6456 | 2.4941 | 2.7972 | True |
| m2_adamw_lr4.0_wd0.02_p0.5_s2030 | adamw | 0.5 | 4 | 0.0012 | 0.02 | 2.6262 | 2.4818 | 2.7707 | True |
| m2_adamw_lr0.5_wd0.01_p0.75_s2030 | adamw | 0.75 | 0.5 | 0.00015 | 0.01 | 3.5682 | 3.0804 | 4.0559 | True |
| m2_adamw_lr0.5_wd0.02_p0.75_s2030 | adamw | 0.75 | 0.5 | 0.00015 | 0.02 | 3.5692 | 3.0814 | 4.0571 | True |
| m2_adamw_lr1.0_wd0.01_p0.75_s2030 | adamw | 0.75 | 1 | 0.0003 | 0.01 | 2.9821 | 2.5836 | 3.3805 | True |
| m2_adamw_lr1.0_wd0.02_p0.75_s2030 | adamw | 0.75 | 1 | 0.0003 | 0.02 | 2.9832 | 2.5842 | 3.3821 | True |
| m2_adamw_lr2.0_wd0.01_p0.75_s2030 | adamw | 0.75 | 2 | 0.0006 | 0.01 | 2.7186 | 2.4111 | 3.0261 | True |
| m2_adamw_lr2.0_wd0.02_p0.75_s2030 | adamw | 0.75 | 2 | 0.0006 | 0.02 | 2.7184 | 2.4108 | 3.0260 | True |
| m2_adamw_lr4.0_wd0.01_p0.75_s2030 | adamw | 0.75 | 4 | 0.0012 | 0.01 | 2.7113 | 2.4129 | 3.0097 | True |
| m2_adamw_lr4.0_wd0.02_p0.75_s2030 | adamw | 0.75 | 4 | 0.0012 | 0.02 | 2.7041 | 2.4086 | 2.9996 | True |
| m2_muon_lr0.5_wd0.01_p0.25_s2030 | muon | 0.25 | 0.5 | 0.0025 | 0.01 | 2.5169 | 2.5164 | 2.5174 | True |
| m2_muon_lr0.5_wd0.02_p0.25_s2030 | muon | 0.25 | 0.5 | 0.0025 | 0.02 | 2.5130 | 2.5131 | 2.5129 | True |
| m2_muon_lr1.0_wd0.01_p0.25_s2030 | muon | 0.25 | 1 | 0.005 | 0.01 | 2.4217 | 2.4229 | 2.4205 | True |
| m2_muon_lr1.0_wd0.02_p0.25_s2030 | muon | 0.25 | 1 | 0.005 | 0.02 | 2.4231 | 2.4232 | 2.4229 | True |
| m2_muon_lr2.0_wd0.01_p0.25_s2030 | muon | 0.25 | 2 | 0.01 | 0.01 | 2.3852 | 2.3872 | 2.3833 | True |
| m2_muon_lr2.0_wd0.02_p0.25_s2030 | muon | 0.25 | 2 | 0.01 | 0.02 | 2.3807 | 2.3839 | 2.3775 | True |
| m2_muon_lr4.0_wd0.01_p0.25_s2030 | muon | 0.25 | 4 | 0.02 | 0.01 | 2.3791 | 2.3778 | 2.3804 | True |
| m2_muon_lr4.0_wd0.02_p0.25_s2030 | muon | 0.25 | 4 | 0.02 | 0.02 | 2.3715 | 2.3697 | 2.3733 | True |
| m1_muon_lr0.5_wd0.01_p0.5_s2030 | muon | 0.5 | 0.5 | 0.0025 | 0.01 | 2.4921 | 2.3654 | 2.6188 | True |
| m1_muon_lr0.5_wd0.02_p0.5_s2030 | muon | 0.5 | 0.5 | 0.0025 | 0.02 | 2.4893 | 2.3619 | 2.6167 | True |
| m1_muon_lr1.0_wd0.01_p0.5_s2030 | muon | 0.5 | 1 | 0.005 | 0.01 | 2.3972 | 2.2715 | 2.5229 | True |
| m1_muon_lr1.0_wd0.02_p0.5_s2030 | muon | 0.5 | 1 | 0.005 | 0.02 | 2.3942 | 2.2694 | 2.5189 | True |
| m1_muon_lr2.0_wd0.01_p0.5_s2030 | muon | 0.5 | 2 | 0.01 | 0.01 | 2.3567 | 2.2359 | 2.4775 | True |
| m1_muon_lr2.0_wd0.02_p0.5_s2030 | muon | 0.5 | 2 | 0.01 | 0.02 | 2.3512 | 2.2312 | 2.4713 | True |
| m2_muon_lr4.0_wd0.01_p0.5_s2030 | muon | 0.5 | 4 | 0.02 | 0.01 | 2.3513 | 2.2310 | 2.4717 | True |
| m2_muon_lr4.0_wd0.02_p0.5_s2030 | muon | 0.5 | 4 | 0.02 | 0.02 | 2.3437 | 2.2256 | 2.4618 | True |
| m2_muon_lr0.5_wd0.01_p0.75_s2030 | muon | 0.75 | 0.5 | 0.0025 | 0.01 | 2.5426 | 2.2797 | 2.8054 | True |
| m2_muon_lr0.5_wd0.02_p0.75_s2030 | muon | 0.75 | 0.5 | 0.0025 | 0.02 | 2.5390 | 2.2763 | 2.8018 | True |
| m2_muon_lr1.0_wd0.01_p0.75_s2030 | muon | 0.75 | 1 | 0.005 | 0.01 | 2.4359 | 2.1891 | 2.6827 | True |
| m2_muon_lr1.0_wd0.02_p0.75_s2030 | muon | 0.75 | 1 | 0.005 | 0.02 | 2.4315 | 2.1849 | 2.6781 | True |
| m2_muon_lr2.0_wd0.01_p0.75_s2030 | muon | 0.75 | 2 | 0.01 | 0.01 | 2.3950 | 2.1500 | 2.6400 | True |
| m2_muon_lr2.0_wd0.02_p0.75_s2030 | muon | 0.75 | 2 | 0.01 | 0.02 | 2.3887 | 2.1449 | 2.6324 | True |
| m2_muon_lr4.0_wd0.01_p0.75_s2030 | muon | 0.75 | 4 | 0.02 | 0.01 | 2.3947 | 2.1490 | 2.6403 | True |
| m2_muon_lr4.0_wd0.02_p0.75_s2030 | muon | 0.75 | 4 | 0.02 | 0.02 | 2.3916 | 2.1481 | 2.6351 | True |

## 2. Optimizer × mixture performance (best cell per arm × p)

| arm | p | best run | mult | wd | V_Q | NLL_A | NLL_B |
|---|---|---|---|---|---|---|---|
| adamw | 0.25 | m2_adamw_lr2.0_wd0.02_p0.25_s2030 | 2 | 0.02 | 2.6650 | 2.6537 | 2.6763 |
| adamw | 0.5 | m2_adamw_lr4.0_wd0.02_p0.5_s2030 | 4 | 0.02 | 2.6262 | 2.4818 | 2.7707 |
| adamw | 0.75 | m2_adamw_lr4.0_wd0.02_p0.75_s2030 | 4 | 0.02 | 2.7041 | 2.4086 | 2.9996 |
| muon | 0.25 | m2_muon_lr4.0_wd0.02_p0.25_s2030 | 4 | 0.02 | 2.3715 | 2.3697 | 2.3733 |
| muon | 0.5 | m2_muon_lr4.0_wd0.02_p0.5_s2030 | 4 | 0.02 | 2.3437 | 2.2256 | 2.4618 |
| muon | 0.75 | m2_muon_lr2.0_wd0.02_p0.75_s2030 | 2 | 0.02 | 2.3887 | 2.1449 | 2.6324 |

## 2b. LR search resolution per (arm, p) (addendum §1.3)

| arm | p | best mult | status |
|---|---|---|---|
| adamw | 0.25 | 2 | RESOLVED |
| adamw | 0.5 | 4 | LR_RANGE_UNRESOLVED |
| adamw | 0.75 | 4 | LR_RANGE_UNRESOLVED |
| muon | 0.25 | 4 | LR_RANGE_UNRESOLVED |
| muon | 0.5 | 4 | LR_RANGE_UNRESOLVED |
| muon | 0.75 | 2 | RESOLVED |

## 3. Selected mixtures

- p*_AdamW = **0.5**  (V_Q = 2.6262)
- p*_Muon  = **0.5**  (V_Q = 2.3437)

## 4. Migration cost

- R = Muon(p*_AdamW) − Muon(p*_Muon) = **0.0000**
- R^AdamW (symmetric diagnostic) = AdamW(p*_Muon) − AdamW(p*_AdamW) = **0.0000**

## 5. Stop rule verdict (addendum §4)

- verdict: **NO_SIGNAL_STOP_LINE**
- detail: p* identical=True; adv_muon=0.0000 vs proxy_muon=0.1539; adv_adamw=0.0000 vs proxy_adamw=0.8652
- M3 recommendation: do not enter M3; negative result is terminal
