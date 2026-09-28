import unittest
from fr.native_schedule import set_native_micro_lr

class _Opt:
    def __init__(self):
        self.param_groups=[{'lr':0.0},{'lr':0.0}]

class TestNativeSchedule(unittest.TestCase):
    def test_each_micro_step_updates_lr(self):
        opt=_Opt()
        def get_lr(step,total,base): return base*(1-step/total)
        a=set_native_micro_lr(opt,get_lr,1,10,1.0)
        b=set_native_micro_lr(opt,get_lr,2,10,1.0)
        self.assertNotEqual(a,b)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], b)
        self.assertAlmostEqual(opt.param_groups[1]['lr'], b)
