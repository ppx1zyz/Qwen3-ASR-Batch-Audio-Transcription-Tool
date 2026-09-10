# Qwen3-ASR 批量音频转文字工具

基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的音频转文字工具，支持：

- 使用 NVIDIA GPU 加速推理
- Qwen3-ASR-1.7B 和 Qwen3-ASR-0.6B 模型
- 选择输入文件夹
- 批量转换音频文件
- 输出同名 `.txt` 文件到指定目录
- 检测已有文本文件并逐个选择覆盖或跳过
- 流式显示当前识别结果
- 显示当前文件字数
- 显示文件和 Chunk 处理进度
- 预计剩余时间
- 显示 GPU 显存使用情况
- 记忆上次使用的路径和参数

---

## 目录结构

建议项目目录结构如下：

```text
qwen3-asr/
├── asr_front.py          # 图形化前端程序
├── qsr_server.py         # ASR 识别核心代码
└── README.md
```

模型建议单独保存，例如：

```text
models/
├── Qwen3-ASR-1.7B/
└── Qwen3-ASR-0.6B/
```

---

## 环境要求

建议环境：

- Python 3.12
- NVIDIA GPU
- 已安装 NVIDIA 驱动
- CUDA 驱动版本与 PyTorch 版本兼容
- Conda 或 Miniconda

如果没有 NVIDIA GPU，也可以使用 CPU，但推理速度会明显降低。

---

## 1. 检查 NVIDIA 和 CUDA 环境

首先确认 NVIDIA 驱动是否正常：

```bash
nvidia-smi
```

输出中通常会看到类似内容：

```text
Driver Version: 535.xx.xx
CUDA Version: 12.x
```

例如：

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
+-----------------------------------------------------------------------------+
```

这里的 `CUDA Version` 表示当前 NVIDIA 驱动所支持的最高 CUDA 运行时版本。

> 注意：`nvidia-smi` 显示的 CUDA 版本不一定等于本地安装的 CUDA Toolkit 版本。安装 PyTorch 时，主要根据 PyTorch 官网提供的安装命令选择对应版本。

---

## 2. 创建 Conda 环境

创建 Python 3.12 环境：

```bash
conda create -n qwen3-asr python=3.12 -y
```

激活环境：

```bash
conda activate qwen3-asr
```

确认当前 Python 版本：

```bash
python --version
```

预期输出类似：

```text
Python 3.12.x
```

---

## 3. 安装 PyTorch

请先打开 PyTorch 官网：

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

根据以下选项选择安装命令：

- PyTorch Build：Stable
- Operating System：你的操作系统
- Package：Pip
- Language：Python
- Compute Platform：根据 NVIDIA 驱动支持的 CUDA 版本选择

官网会生成类似下面的安装命令：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

实际命令以 PyTorch 官网生成的内容为准。

例如，CUDA 12.4 版本可以使用：

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

如果需要安装 CUDA 11.8 版本，可以使用：

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

> 不建议直接执行未经确认的 CUDA 安装命令。请根据自己的 NVIDIA 驱动版本和 PyTorch 官网当前提供的版本选择。

---

## 4. 安装项目依赖

安装 Qwen3-ASR：

```bash
pip install -U qwen-asr
```

安装模型下载工具：

```bash
pip install -U modelscope
```

安装音频处理和进度显示依赖：

```bash
pip install -U numpy librosa tqdm
```

如果项目中还使用了其他依赖，可以一并安装：

```bash
pip install -U torch qwen-asr modelscope numpy librosa tqdm
```

---

## 5. 验证 PyTorch 是否可以使用 GPU

执行：

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU 正常时，输出类似：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

如果输出：

```text
CUDA available: False
GPU: CPU
```

说明 PyTorch 当前没有检测到 CUDA。可以检查：

1. NVIDIA 驱动是否正常；
2. 是否激活了正确的 Conda 环境；
3. PyTorch 是否安装成了 CPU 版本；
4. PyTorch 的 CUDA 版本是否与驱动兼容；
5. 是否在远程服务器或容器中正确映射了 GPU。

也可以使用以下命令查看 PyTorch 版本：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

---

## 6. 下载 Qwen3-ASR 模型

进入模型保存目录：

```bash
cd /path/to/your/models
```

Windows 示例：

```powershell
cd D:\models
```

Linux 示例：

```bash
cd ~/models
```

下载 1.7B 模型：

```bash
modelscope download \
    --model Qwen/Qwen3-ASR-1.7B \
    --local_dir ./Qwen3-ASR-1.7B
```

下载 0.6B 模型：

```bash
modelscope download \
    --model Qwen/Qwen3-ASR-0.6B \
    --local_dir ./Qwen3-ASR-0.6B
```

下载完成后，目录结构类似：

```text
models/
├── Qwen3-ASR-1.7B/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── Qwen3-ASR-0.6B/
    ├── config.json
    ├── model.safetensors
    └── ...
```

模型大小较大，下载时间取决于网络速度和磁盘性能。

---

## 7. 配置模型路径

在 `asr_front.py` 或项目配置中设置模型路径。

例如：

```python
MODEL_PATH = r"D:\models\Qwen3-ASR-1.7B"
```

Linux 示例：

```python
MODEL_PATH = "/home/user/models/Qwen3-ASR-1.7B"
```

如果程序支持在图形界面中选择模型目录，则启动程序后直接选择对应模型目录即可。

---

## 8. 启动图形化前端

确认已经激活 Conda 环境：

```bash
conda activate qwen3-asr
```

进入项目目录：

```bash
cd /path/to/qwen3-asr
```

启动前端：

```bash
python asr_front.py
```

Windows 示例：

```powershell
python asr_front.py
```

也可以使用：

```bash
python -m asr_front
```

---

## 9. 使用步骤

启动程序后：

### 选择输入文件夹

选择需要转换的音频文件目录。

支持的音频格式包括：

```text
.wav
.mp3
.m4a
.flac
.ogg
.aac
.wma
```

### 选择输出文件夹

程序会将识别结果保存到该目录。

例如：

```text
输入：
meeting_001.wav

输出：
meeting_001.txt
```

### 选择模型目录

选择已经下载好的模型目录，例如：

```text
Qwen3-ASR-1.7B
```

### 开始转换

点击：

```text
开始批量转换
```

程序会按照文件名排序，逐个处理音频文件。

---

## 10. 输出文件规则

假设输入文件夹为：

```text
audio/
├── meeting_001.wav
├── meeting_002.mp3
└── meeting_003.flac
```

输出文件夹为：

```text
txt/
```

转换后：

```text
txt/
├── meeting_001.txt
├── meeting_002.txt
└── meeting_003.txt
```

输出文件名会使用原音频文件的文件名，只替换扩展名：

```text
example.wav  -> example.txt
example.mp3  -> example.txt
example.m4a  -> example.txt
```

---

## 11. 已有 TXT 文件的处理

如果输出目录中已经存在同名文本文件，程序会逐个弹窗确认。

例如：

```text
meeting_001.txt 已经存在。

是否覆盖？

是：覆盖原文件
否：跳过当前文件
```

目前每个文件单独询问，可以避免误覆盖已有结果。

后续可以扩展为：

- 全部覆盖
- 全部跳过
- 仅覆盖空文件
- 根据文件修改时间决定是否覆盖
- 自动备份旧 TXT 文件

---

## 12. 界面状态信息

程序运行时会显示以下信息。

### 文件进度

例如：

```text
文件进度：3/20
```

表示当前已经完成 3 个文件，总共有 20 个文件。

### Chunk 进度

例如：

```text
Chunk：5/18
```

表示当前音频被分成 18 个 Chunk，目前已完成第 5 个。

### 当前识别结果

识别过程中，界面会实时显示已经完成的 Chunk 的文本内容。

### 当前字数

例如：

```text
当前文件字数：1256
```

### 预计剩余时间

例如：

```text
当前文件预计剩余：00:02:35
```

文件完成后会显示整个批处理任务的预计剩余时间：

```text
全部任务预计剩余：00:18:42
```

预计时间会随着实际处理速度动态变化。

---

## 13. GPU 显存显示

如果模型运行在 CUDA GPU 上，界面会显示类似：

```text
NVIDIA GeForce RTX 3090 |
进程使用: 3.21 GiB |
缓存: 4.02 GiB |
GPU占用: 8.15 GiB / 24.00 GiB |
剩余: 15.85 GiB
```

各项含义如下：

| 项目     | 含义                                   |
| -------- | -------------------------------------- |
| 进程使用 | 当前 Python/PyTorch 进程实际分配的显存 |
| 缓存     | PyTorch 预留的缓存显存                 |
| GPU 占用 | 整张显卡当前已经使用的显存             |
| 总显存   | 显卡的总显存                           |
| 剩余     | 显卡当前可用显存                       |

如果显示：

```text
GPU 不可用
```

说明当前模型可能运行在 CPU 上，或者 PyTorch 没有正确识别 CUDA。

---

## 14. 重要代码依赖

项目核心代码会使用以下模块：

```python
import os
import time
import json
import queue
import threading

from pathlib import Path
from dataclasses import dataclass
from tkinter import *

import numpy as np
import librosa
import torch

from tqdm import tqdm
from qwen_asr import Qwen3ASRModel
```

对应的主要依赖为：

```bash
pip install -U numpy librosa torch tqdm qwen-asr modelscope
```

---

## 15. 可选：创建 `requirements.txt`

可以在项目根目录创建 `requirements.txt`：

```text
numpy
librosa
tqdm
qwen-asr
modelscope
```

PyTorch 建议按照 PyTorch 官网命令单独安装，不直接固定在这个文件中，因为不同 GPU 和 CUDA 环境需要使用不同的 PyTorch 安装命令。

安装其他依赖：

```bash
pip install -r requirements.txt
```

---

## 16. 常见问题

### 16.1 `torch.cuda.is_available()` 返回 `False`

执行：

```bash
nvidia-smi
```

如果 `nvidia-smi` 本身无法运行，说明 NVIDIA 驱动没有正确安装。

如果 `nvidia-smi` 正常，但 PyTorch 返回 `False`，检查 PyTorch 安装版本：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

重新按照 PyTorch 官网命令安装 CUDA 版本 PyTorch。

---

### 16.2 `torch.cuda.get_device_name(0)` 报错

如果当前没有 GPU，直接执行：

```python
torch.cuda.get_device_name(0)
```

可能会报错。

建议使用：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### 16.3 找不到 `qwen_asr`

如果出现：

```text
ModuleNotFoundError: No module named 'qwen_asr'
```

确认当前环境是否正确：

```bash
conda activate qwen3-asr
```

然后重新安装：

```bash
pip install -U qwen-asr
```

---

### 16.4 找不到模型文件

请确认 `MODEL_PATH` 指向模型的实际目录，而不是上一级目录。

正确示例：

```text
D:\models\Qwen3-ASR-1.7B
```

错误示例：

```text
D:\models
```

模型目录中通常应该存在：

```text
config.json
model.safetensors
```

等文件。

---

### 16.5 显存不足

如果 1.7B 模型显存不足，可以尝试：

1. 使用 0.6B 模型；
2. 使用 `float16`；
3. 减小 Chunk 长度；
4. 关闭其他占用显存的程序；
5. 检查是否有其他 Python 进程占用 GPU。

查看显存：

```bash
nvidia-smi
```

---

### 16.6 Tkinter 无法导入

Linux 下安装：

```bash
sudo apt install python3-tk
```

Windows 通常随 Python 一起安装。如果没有，需要重新安装 Python，并勾选 Tcl/Tk 组件。

---

## 17. 开发说明

程序采用前后端分离结构：

```text
qsr_hurmaness.py
```

负责：

- 音频加载
- 模型加载
- 单个 Chunk 识别
- 流式识别
- 文本生成

```text
asr_front.py
```

负责：

- 图形界面
- 文件夹选择
- 批量任务管理
- 覆盖或跳过提示
- 进度显示
- ETA 计算
- 显存显示
- 配置记忆

后台识别使用独立线程，界面通过消息队列接收识别进度：

```text
识别线程 -> queue.Queue -> Tkinter 主线程
```

这样可以避免识别过程中界面卡死。

---


## 19. 最简启动流程

```bash
# 创建环境
conda create -n qwen3-asr python=3.12 -y

# 激活环境
conda activate qwen3-asr

# 安装 PyTorch
# 请使用 PyTorch 官网生成的命令

# 安装依赖
pip install -U qwen-asr modelscope numpy librosa tqdm

# 检查 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 下载模型
modelscope download \
    --model Qwen/Qwen3-ASR-1.7B \
    --local_dir ./Qwen3-ASR-1.7B

# 启动前端
python asr_front.py
```

启动后选择输入文件夹、输出文件夹和模型目录，即可开始批量转换。
