# 服务器上传、解压与安装（不需要 unzip）

假设本地压缩包：

`E:\涨落响应\FR_Experiment_Framework_v0.1.zip`

且你的 SSH config 可以直接使用 `win-wsl`：

```powershell
scp "E:\涨落响应\FR_Experiment_Framework_v0.1.zip" `
  win-wsl:/home/b205/projects/fluctuation_response/
```

服务器：

```bash
cd /home/b205/projects/fluctuation_response
rm -rf _fr_framework_incoming
mkdir -p _fr_framework_incoming

python3 -m zipfile -e \
  FR_Experiment_Framework_v0.1.zip \
  _fr_framework_incoming

bash _fr_framework_incoming/FR_Experiment_Framework_v0.1/install_into_project.sh \
  /home/b205/projects/fluctuation_response
```

安装脚本不会写入或删除 `minimind/`。

然后：

```bash
source /home/b205/miniforge3/etc/profile.d/conda.sh
conda activate fr_minimind
cd /home/b205/projects/fluctuation_response

python research/scripts/check_scaffold.py
CUDA_VISIBLE_DEVICES=1 python research/scripts/check_environment.py \
  --project-root /home/b205/projects/fluctuation_response

PYTHONPATH=research/src python -m unittest discover -s research/tests -v
```

预期：scaffold tests PASS；`e0_replay_skeleton.py` **故意输出 E0_BLOCKED_BY_DESIGN**，直到 TASK-00 审计并实现 adapter。
