"""tests 共用 bootstrap：把 src/ 注入 sys.path（测试从 continuation_adaptation/ 目录运行）。"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
LAB_ROOT = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
