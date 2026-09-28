"""Test 2 — Beta modification correctness（docs/EXPERIMENT_PROTOCOL.md §6.2）。

(a) 陷阱存在性：`optimizer.load_state_dict` 会用 checkpoint 的 param_groups
    覆盖新 optimizer 构造时的 betas —— 改 β 必须发生在 load 之后；
(b) set_betas 在 load 之后生效，且 m/v/step 逐位不被触碰；
(c) 手推 AdamW 单步数学（torch 2.x 原生更新序）验证 β2=0.99 真正参与更新；
(d) apply_arm 全臂语义：清零通道精确为 0，保留通道哈希不变，step 处理正确；
(e) config 臂表与代码臂表（唯一定义）一致。
"""
import _bootstrap  # noqa: F401
import json
import unittest
from pathlib import Path

import torch

from optimizer_utils import set_betas
from state_adapter import ARM_SPECS, _state_channel_hashes, apply_arm, check_arms_table_consistency


def _fresh_param():
    torch.manual_seed(3)
    return torch.nn.Parameter(torch.tensor([0.5, -1.25, 2.0]))


def _grad_step(p, opt, g):
    opt.zero_grad(set_to_none=True)
    p.grad = g.clone()
    opt.step()


class TestBetaTrapAndSet(unittest.TestCase):
    def test_a_load_state_dict_restores_old_betas(self):
        """陷阱：load 后 betas 是 checkpoint 的 (0.9,0.999)，不是构造的 (0.9,0.99)。"""
        p1 = _fresh_param()
        opt1 = torch.optim.AdamW([p1], lr=1e-3, betas=(0.9, 0.999))
        _grad_step(p1, opt1, torch.tensor([0.1, -0.2, 0.3]))
        saved = opt1.state_dict()

        p2 = _fresh_param()
        opt2 = torch.optim.AdamW([p2], lr=1e-3, betas=(0.9, 0.99))  # 构造时改成 0.99
        opt2.load_state_dict(saved)
        got = tuple(float(x) for x in opt2.param_groups[0]["betas"])
        self.assertEqual(got, (0.9, 0.999), "load_state_dict 应恢复 checkpoint 的 betas（陷阱存在）")

    def test_b_set_betas_after_load_touches_no_state(self):
        p1 = _fresh_param()
        opt1 = torch.optim.AdamW([p1], lr=1e-3, betas=(0.9, 0.999))
        for g in (torch.tensor([0.1, -0.2, 0.3]), torch.tensor([0.05, 0.1, -0.1])):
            _grad_step(p1, opt1, g)
        saved = opt1.state_dict()

        p2 = _fresh_param()
        opt2 = torch.optim.AdamW([p2], lr=1e-3)
        opt2.load_state_dict(saved)
        h_before = _state_channel_hashes(opt2)

        set_betas(opt2, 0.9, 0.99)
        self.assertEqual(tuple(float(x) for x in opt2.param_groups[0]["betas"]), (0.9, 0.99))
        h_after = _state_channel_hashes(opt2)
        for ch in ("m", "v", "step"):
            self.assertEqual(h_before[ch], h_after[ch], f"set_betas 不得触碰 {ch}")

    def test_c_beta2_actually_used_in_update_math(self):
        """手推 torch AdamW 单步（mul_(1-lr*wd) → addcdiv_），β2=0.99 参与更新。"""
        lr, eps, wd = 1e-3, 1e-8, 0.01
        g1 = torch.tensor([0.1, -0.2, 0.3])
        g2 = torch.tensor([0.2, 0.1, -0.4])
        g3 = torch.tensor([-0.1, 0.05, 0.2])

        # 3 步 β2=0.999 建立 state
        p1 = _fresh_param()
        opt1 = torch.optim.AdamW([p1], lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=wd)
        for g in (g1, g2, g3):
            _grad_step(p1, opt1, g)
        saved = opt1.state_dict()
        m3 = saved["state"][0]["exp_avg"].clone()
        v3 = saved["state"][0]["exp_avg_sq"].clone()

        # 迁移：fresh param + load + set_betas(0.9,0.99) + 一步 g4
        p2 = _fresh_param()
        opt2 = torch.optim.AdamW([p2], lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=wd)
        opt2.load_state_dict(saved)
        set_betas(opt2, 0.9, 0.99)
        g4 = torch.tensor([0.15, -0.05, 0.1])
        p2_before = p2.detach().clone()
        _grad_step(p2, opt2, g4)

        # ---- 手推（torch 2.x AdamW 单参数单步更新序）----
        b1, b2 = 0.9, 0.99
        m4 = b1 * m3 + (1 - b1) * g4
        v4 = b2 * v3 + (1 - b2) * g4 * g4
        t = 4.0
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        p_manual = p2_before * (1 - lr * wd)
        denom = (v4 / bc2).sqrt() + eps
        p_manual = p_manual - (lr / bc1) * m4 / denom

        s = opt2.state_dict()["state"][0]
        self.assertTrue(torch.equal(s["exp_avg"], m4), "exp_avg 更新必须用 β1=0.9（与旧 β 相同，验证精度）")
        self.assertTrue(torch.equal(s["exp_avg_sq"], v4), "exp_avg_sq 更新必须用新 β2=0.99")
        self.assertEqual(float(s["step"]), 4.0)
        self.assertTrue(torch.allclose(p2.detach(), p_manual, atol=1e-7),
                        f"参数更新未按 β2=0.99 数学执行: {p2.detach()} vs {p_manual}")


class TestApplyArmSemantics(unittest.TestCase):
    def _loaded_opt(self):
        p = _fresh_param()
        opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.999))
        for g in (torch.tensor([0.1, -0.2, 0.3]), torch.tensor([0.05, 0.1, -0.1]),
                  torch.tensor([0.2, -0.05, 0.15])):
            _grad_step(p, opt, g)
        saved = opt.state_dict()
        p2 = _fresh_param()
        opt2 = torch.optim.AdamW([p2], lr=1e-3)
        opt2.load_state_dict(saved)
        return p2, opt2

    def test_d_arm_semantics(self):
        for arm, spec in ARM_SPECS.items():
            p, opt = self._loaded_opt()
            h_pre = _state_channel_hashes(opt)
            rec = apply_arm(opt, arm)

            self.assertEqual(rec["arm"], arm)
            got_betas = tuple(float(x) for x in opt.param_groups[0]["betas"])
            self.assertEqual(got_betas, tuple(float(x) for x in spec["betas"]))

            h_post = _state_channel_hashes(opt)
            entry = opt.state[p]
            if not spec["reset_m"]:
                self.assertEqual(h_pre["m"], h_post["m"], f"{arm}: m 必须逐位保留")
            else:
                self.assertTrue(bool(entry["exp_avg"].eq(0).all()), f"{arm}: m 必须精确为 0")
            if not spec["reset_v"]:
                self.assertEqual(h_pre["v"], h_post["v"], f"{arm}: v 必须逐位保留")
            else:
                self.assertTrue(bool(entry["exp_avg_sq"].eq(0).all()), f"{arm}: v 必须精确为 0")
            if not spec["reset_step"]:
                self.assertEqual(h_pre["step"], h_post["step"], f"{arm}: step 必须保留")
                self.assertEqual(float(entry["step"]), 3.0)
            else:
                self.assertEqual(float(entry["step"]), 0.0, f"{arm}: step 必须归零")

    def test_e_config_arms_table_matches_code(self):
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "smoke_v01.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        check_arms_table_consistency(cfg["arms"])  # 不一致即抛错

    def test_unknown_arm_rejected(self):
        p, opt = self._loaded_opt()
        with self.assertRaises(ValueError):
            apply_arm(opt, "E")


if __name__ == "__main__":
    unittest.main()
