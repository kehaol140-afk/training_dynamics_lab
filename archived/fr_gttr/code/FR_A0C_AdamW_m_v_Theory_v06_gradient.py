import torch
import json
from pathlib import Path

from research.src.fr.direction import collect_current_gradients


root=Path("/home/b205/projects/fluctuation_response")


# 这里只做接口验证
# 下一步接theta_protocol replay


print("READY_FOR_GRADIENT_EXPORT")
print("Need E1/E2 replay batch")
