"""hashing.py 的基础单元测试。

验证两件事：
1. dict 的 key 插入顺序不会改变 hash；
2. 状态内容发生变化时 hash 会变化。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fr.hashing import stable_hash


class TestHashing(unittest.TestCase):
    def test_dict_key_order_is_stable(self):
        """相同字典内容、不同 key 顺序应产生相同 hash。"""
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))

    def test_change_changes_hash(self):
        """状态内容改变后 hash 必须改变。"""
        self.assertNotEqual(stable_hash({"a": 1}), stable_hash({"a": 2}))


if __name__ == "__main__":
    unittest.main()
