# -*- coding: utf-8 -*-

"""
Qwen3-ASR-1.7B
Windows + Transformers 本地流式/分段转写

特点：
    1. 不使用 vLLM
    2. 使用 Transformers backend
    3. 音频按 chunk 分段
    4. 每处理一个 chunk 就立即输出结果
    5. 显示整体处理进度
    6. 最后自动合并成完整文本
    7. 保存 txt

注意：
    这不是 vLLM 的 token-level streaming。
    而是 Windows 下更加容易部署的：
        chunk -> inference -> output
    分段流式显示方案。
"""


# ============================================================
#                    用户参数区域
# ============================================================

# ------------------------------------------------------------
# 1. 音频文件
# ------------------------------------------------------------

AUDIO_PATH = r"C:\Users\ZhengAMD\Downloads\20260907.mp3"

# 最终转写文本保存位置
OUTPUT_PATH = r"transcription.txt"


# ------------------------------------------------------------
# 2. Qwen3-ASR 模型
# ------------------------------------------------------------

# 推荐 Transformers 专用 HF 模型
MODEL_PATH = r"C:\Users\ZhengAMD\Desktop\Project\2_Python\Qwen3-ASR-1.7B"

# 如果没有下载到本地，也可以直接：
#
# MODEL_PATH = "Qwen/Qwen3-ASR-1.7B-hf"


# ------------------------------------------------------------
# 3. GPU 设置
# ------------------------------------------------------------

# "cuda"  = NVIDIA GPU
# "cpu"   = CPU
DEVICE = "cuda"

# 推荐：
# NVIDIA GPU 使用 bfloat16
#
# 如果你的 GPU 不支持 BF16，可以改：
# torch.float16
#
# CPU 建议：
# torch.float32
DTYPE = "bfloat16"


# ------------------------------------------------------------
# 4. 语言
# ------------------------------------------------------------

# 中文：
LANGUAGE = "Chinese"

# 自动识别：
# LANGUAGE = None

# 例如：
# LANGUAGE = "English"
# LANGUAGE = "Japanese"


# ------------------------------------------------------------
# 5. Chunk 大小
# ------------------------------------------------------------

# 每次送给 ASR 的音频长度，单位：秒
#
# 例如：
#   0.5  -> 延迟低，但是推理次数多
#   1.0  -> 推荐
#   2.0  -> 推荐，稳定
#   5.0  -> 延迟高，但是效率较好
#
CHUNK_SIZE_SEC = 20.0


# ------------------------------------------------------------
# 6. Chunk 重叠
# ------------------------------------------------------------

# 两个 chunk 之间重叠多少秒
#
# 例如：
#
# chunk 1：
# 0 ~ 2 s
#
# chunk 2：
# 1.5 ~ 3.5 s
#
# overlap = 0.5
#
# 重叠可以减少句子被切断的问题。
#
CHUNK_OVERLAP_SEC = 3


# ------------------------------------------------------------
# 7. 最大生成 Token
# ------------------------------------------------------------

# 每一个 chunk 最多生成多少 token
#
# 普通中文：
# 64   推荐
#
# 语速较快：
# 128
#
# 如果 chunk 很短：
# 32
#
MAX_NEW_TOKENS = 128


# ------------------------------------------------------------
# 8. Context / 专业词汇
# ------------------------------------------------------------

# 可以放专业词汇，提高专业场景识别效果。
#
# 例如你做 UWB / 微等离子体：
#
# PROMPT = """
# 专业词汇：
# UWB、超宽带、脉冲源、微等离子体、
# MEMS、HRTIM、STM32、Qwen、半导体
# """
#
# 如果没有专业词汇：
PROMPT = None


# ------------------------------------------------------------
# 9. 是否显示每个 Chunk 的详细信息
# ------------------------------------------------------------

SHOW_CHUNK_INFO = True


# ------------------------------------------------------------
# 10. 是否清理每段文本两端空格
# ------------------------------------------------------------

STRIP_TEXT = True


# ============================================================
#                    Python 导入
# ============================================================

import os
import time

import numpy as np
import librosa
import torch

from tqdm import tqdm

from qwen_asr import Qwen3ASRModel

def format_bytes(num_bytes):
    """
    将字节数转换成易读格式。
    """
    if num_bytes is None:
        return "N/A"

    num_bytes = float(num_bytes)

    units = ["B", "MiB", "GiB", "TiB"]

    for unit in units:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024

    return f"{num_bytes:.2f} PiB"
def get_gpu_memory_info(device=None):
    """
    返回 GPU 显存信息。

    返回：
        allocated：当前 PyTorch 进程实际使用的显存
        reserved：PyTorch 缓存池占用的显存
        used：整个 GPU 已使用显存
        free：整个 GPU 剩余显存
        total：GPU 总显存
    """

    if not torch.cuda.is_available():
        return {
            "allocated": None,
            "reserved": None,
            "used": None,
            "free": None,
            "total": None,
        }

    if device is None:
        device = torch.cuda.current_device()

    # 当前 PyTorch 进程实际分配的显存
    allocated = torch.cuda.memory_allocated(device)

    # PyTorch reserved/cache 显存
    reserved = torch.cuda.memory_reserved(device)

    # 整个 GPU 的显存状态
    free, total = torch.cuda.mem_get_info(device)

    used = total - free

    return {
        "allocated": allocated,
        "reserved": reserved,
        "used": used,
        "free": free,
        "total": total,
    }

# ============================================================
#                    参数检查
# ============================================================
class qsr_hurmaness:
    def __init__(self,
                 audio_path=AUDIO_PATH,
                 model_path=MODEL_PATH,
                 device=DEVICE,
                 dtype=DTYPE,
                 language=LANGUAGE,
                 chunk_size_sec=CHUNK_SIZE_SEC,
                 chunk_overlap_sec=CHUNK_OVERLAP_SEC,
                 max_new_tokens=MAX_NEW_TOKENS,
                 show_chunk_info=SHOW_CHUNK_INFO):
        self.audio_path = audio_path
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.language = language
        self.chunk_size_sec = chunk_size_sec
        self.chunk_overlap_sec = chunk_overlap_sec
        self.max_new_tokens = max_new_tokens
        self.show_chunk_info = show_chunk_info
    def check_config(self):

        print("=" * 70)
        print("Qwen3-ASR Windows Transformers")
        print("=" * 70)

        print(f"Audio       : {self.audio_path}")
        print(f"Model       : {self.model_path}")
        print(f"Device      : {self.device}")
        print(f"Dtype       : {self.dtype}")
        print(f"Language    : {self.language}")
        print(f"Chunk       : {self.chunk_size_sec} s")
        print(f"Overlap     : {self.chunk_overlap_sec} s")
        print(f"Max tokens  : {self.max_new_tokens}")

        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(
                f"\n找不到音频文件：\n{self.audio_path}"
            )

        if not os.path.exists(self.model_path):

            # 如果 MODEL_PATH 不是本地路径，
            # 可以直接交给 HuggingFace。
            if "/" in self.model_path:
                print("\n使用 HuggingFace 模型：")
                print(self.model_path)

            else:
                raise FileNotFoundError(
                    f"\n找不到模型目录：\n{self.model_path}"
                )


    # ============================================================
    #                    加载音频
    # ============================================================

    def load_audio(self):

        print("\n[1/4] 加载音频...")

        # 统一转成 16 kHz Mono
        #
        # Qwen3-ASR 官方处理器支持音频输入，
        # 这里提前转换可以保证 chunk 处理的一致性。
        audio, sr = librosa.load(
            self.audio_path,
            sr=16000,
            mono=True
        )

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        duration = len(audio) / 16000.0

        print(f"采样率：16000 Hz")
        print(f"声道：Mono")
        print(f"时长：{duration:.2f} 秒")
        print(f"采样点：{len(audio):,}")

        return audio, duration


    # ============================================================
    #                    加载 Qwen3-ASR
    # ============================================================

    def load_model(self):

        print("\n[2/4] 加载 Qwen3-ASR-1.7B...")

        start = time.perf_counter()

        # --------------------------------------------------------
        # Dtype
        # --------------------------------------------------------

        if DTYPE == "bfloat16":
            dtype = torch.bfloat16

        elif DTYPE == "float16":
            dtype = torch.float16

        elif DTYPE == "float32":
            dtype = torch.float32

        else:
            raise ValueError(
                "DTYPE 必须是："
                "bfloat16 / float16 / float32"
            )

        # --------------------------------------------------------
        # Device
        # --------------------------------------------------------

        if DEVICE == "cuda":

            if not torch.cuda.is_available():

                print(
                    "\n===警告：CUDA 不可用，自动切换到 CPU===="
                )

                device_map = "cpu"
                dtype = torch.float32

            else:

                device_map = "cuda:0"

                print(
                    f"GPU：{torch.cuda.get_device_name(0)}"
                )

                print(
                    f"显存："
                    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                )

        else:

            device_map = "cpu"
            dtype = torch.float32


        # --------------------------------------------------------
        # 创建模型
        # --------------------------------------------------------

        model = Qwen3ASRModel.from_pretrained(

            self.model_path,

            # 自动将模型放到 GPU
            device_map=device_map,

            # FP16 / BF16
            dtype=dtype,

            # 最大推理 batch
            #
            # 我们这里每次只处理一个 chunk，
            # 因此设置 1 即可。
            max_inference_batch_size=1,

            # 最大生成 token
            max_new_tokens=self.max_new_tokens,

        )

        elapsed = time.perf_counter() - start

        print(
            f"模型加载完成，耗时：{elapsed:.2f} 秒"
        )

        return model


    # ============================================================
    #                    单 Chunk 推理
    # ============================================================

    def transcribe_chunk(
        self,
        model,
        audio_chunk
    ):

        """
        对一个音频 Chunk 进行识别。

        audio_chunk：
            np.ndarray
            float32
            16 kHz
            Mono
        """

        result = model.transcribe(

            audio=(audio_chunk, 16000),

            # 专业词汇提示
            context=PROMPT,

            # 指定语言
            language=LANGUAGE,

            # 不需要时间戳
            return_time_stamps=False,
        )

        text = result[0].text

        if STRIP_TEXT:
            text = text.strip()

        return text


    # ============================================================
    #                    流式处理
    # ============================================================

    def streaming_transcribe(self,
        model,
        audio,
        duration,
        chunk_callback=None
    ):

        print("\n[3/4] 开始流式识别")
        print("=" * 70)

        # --------------------------------------------------------
        # Chunk 参数
        # --------------------------------------------------------

        sample_rate = 16000

        chunk_samples = int(
            self.chunk_size_sec * sample_rate
        )

        overlap_samples = int(
            self.chunk_overlap_sec * sample_rate
        )

        hop_samples = (
            chunk_samples - overlap_samples
        )

        if hop_samples <= 0:

            raise ValueError(
                "CHUNK_OVERLAP_SEC 必须小于 CHUNK_SIZE_SEC"
            )

        total_samples = len(audio)

        # --------------------------------------------------------
        # 计算 chunk 数量
        # --------------------------------------------------------

        chunk_count = int(
            np.ceil(
                max(
                    0,
                    total_samples - overlap_samples
                )
                / hop_samples
            )
        )

        print(
            f"总 Chunk 数：{chunk_count}"
        )

        print(
            f"每 Chunk：{self.chunk_size_sec:.2f} s"
        )

        print(
            f"Chunk 重叠：{self.chunk_overlap_sec:.2f} s"
        )

        print()

        # --------------------------------------------------------
        # 保存所有识别结果
        # --------------------------------------------------------

        all_text = []

        # --------------------------------------------------------
        # 开始计时
        # --------------------------------------------------------

        start_time = time.perf_counter()
        # --------------------------------------------------------
        # 确定模型所在设备
        # --------------------------------------------------------

        gpu_device = None

        try:
            model_device = next(model.parameters()).device

            if model_device.type == "cuda":
                gpu_device = model_device

        except Exception:
            pass


        # --------------------------------------------------------
        # tqdm
        # --------------------------------------------------------

        progress = tqdm(
        total=total_samples,
        desc="处理进度",
        unit="sample",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| "
            "{n_fmt}/{total_fmt} {percentage:3.0f}% "
            "[已用时: {elapsed}, 剩余: {remaining}] "
            "{postfix}"
            )
        )   

        # --------------------------------------------------------
        # Chunk 循环
        # --------------------------------------------------------

        start_sample = 0

        chunk_id = 0

        while start_sample < total_samples:

            chunk_id += 1

            end_sample = min(
                start_sample + chunk_samples,
                total_samples
            )

            audio_chunk = audio[
                start_sample:end_sample
            ]

            # 当前时间范围
            start_sec = start_sample / sample_rate
            end_sec = end_sample / sample_rate

            # ----------------------------------------------------
            # 推理
            # ----------------------------------------------------

            chunk_start_time = time.perf_counter()

            text = self.transcribe_chunk(
                model,
                audio_chunk
            )

            chunk_elapsed = (
                time.perf_counter()
                - chunk_start_time
            )

            # ----------------------------------------------------
            # 保存
            # ----------------------------------------------------

            if text:

                all_text.append(text)

            # ----------------------------------------------------
            # 实时输出
            # ----------------------------------------------------

            if self.show_chunk_info:

                print(
                    f"\n"
                    f"[Chunk {chunk_id:03d}] "
                    f"{start_sec:7.2f}s -> "
                    f"{end_sec:7.2f}s"
                )

                print(
                    f"识别结果：{text}"
                )

                print(
                    f"本 Chunk 推理："
                    f"{chunk_elapsed:.2f}s"
                )

            else:

                print(
                    text,
                    flush=True
                )

            # --------------------------------------------------------
            # 更新进度
            # --------------------------------------------------------

            processed_samples = end_sample - start_sample

            progress.update(processed_samples)

            processed = progress.n

            # --------------------------------------------------------
            # 计算预计剩余时间
            # --------------------------------------------------------

            eta_seconds = None

            if processed > 0:
                elapsed_so_far = time.perf_counter() - start_time
                remaining_samples = max(
                    0,
                    total_samples - processed
                )

                eta_seconds = (
                    elapsed_so_far
                    * remaining_samples
                    / processed
                )

            # --------------------------------------------------------
            # 获取显存信息
            # --------------------------------------------------------

            memory_info = get_gpu_memory_info(gpu_device)

            if gpu_device is not None:
                allocated_text = format_bytes(
                    memory_info["allocated"]
                )

                reserved_text = format_bytes(
                    memory_info["reserved"]
                )

                used_text = format_bytes(
                    memory_info["used"]
                )

                free_text = format_bytes(
                    memory_info["free"]
                )

                total_text = format_bytes(
                    memory_info["total"]
                )

                memory_postfix = (
                    f"进程:{allocated_text} "
                    f"占用:{used_text}/{total_text} "
                    f"剩余:{free_text}"
                )

            else:
                memory_postfix = "GPU:N/A"

            # --------------------------------------------------------
            # 格式化 ETA
            # --------------------------------------------------------

            if eta_seconds is None:
                eta_text = "计算中"

            elif eta_seconds < 3600:
                minutes = int(eta_seconds // 60)
                seconds = int(eta_seconds % 60)

                eta_text = f"{minutes:02d}:{seconds:02d}"

            else:
                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                seconds = int(eta_seconds % 60)

                eta_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # --------------------------------------------------------
            # 更新 tqdm 后缀
            # --------------------------------------------------------

            progress.set_postfix_str(
                f"{memory_postfix} ETA:{eta_text}",
                refresh=True
            )


            # ----------------------------------------------------
            # 下一 Chunk
            # ----------------------------------------------------

            start_sample += hop_samples
            # --------------------------------------------------------
            # 通知前端当前 Chunk 已完成
            # --------------------------------------------------------

            if chunk_callback is not None:
                processed_chunks = chunk_id

                elapsed_so_far = (
                    time.perf_counter()
                    - start_time
                )

                remaining_chunks = max(
                    0,
                    chunk_count - processed_chunks
                )

                if processed_chunks > 0:
                    eta_seconds = (
                        elapsed_so_far
                        * remaining_chunks
                        / processed_chunks
                    )
                else:
                    eta_seconds = None

                chunk_callback(
                    {
                        "chunk_id": chunk_id,
                        "chunk_count": chunk_count,

                        "start_sec": start_sec,
                        "end_sec": end_sec,

                        "text": text or "",
                        "text_so_far": "\n".join(all_text),
                        "char_count": len("\n".join(all_text)),

                        "chunk_elapsed": chunk_elapsed,
                        "elapsed": elapsed_so_far,
                        "eta_seconds": eta_seconds,
                    }
                )


        # --------------------------------------------------------
        # 结束进度条
        # --------------------------------------------------------

        progress.close()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # --------------------------------------------------------
        # 合并最终文本
        # --------------------------------------------------------

        final_text = "\n".join(
            all_text
        )

        # --------------------------------------------------------
        # 性能统计
        # --------------------------------------------------------

        print("\n" + "=" * 70)

        print("[4/4] 识别完成")

        print("=" * 70)

        print(
            f"音频长度：{duration:.2f} 秒"
        )

        print(
            f"处理时间：{elapsed:.2f} 秒"
        )

        if elapsed > 0:

            rtf = elapsed / duration

            speed = duration / elapsed

            print(
                f"RTF：{rtf:.3f}"
            )

            print(
                f"处理速度：{speed:.2f} × 实时速度"
            )

        # --------------------------------------------------------
        # 最终文本
        # --------------------------------------------------------

        print("\n【最终完整文本】")
        print("-" * 70)

        print(
            final_text
        )

        print("-" * 70)

        return final_text


    # ============================================================
    #                    保存文本
    # ============================================================

    def save_text(self, text):

        if self.output_path is None:
            return

        print(
            f"\n正在保存：{self.output_path}"
        )

        with open(
            self.output_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(text)

        print("保存完成。")


# ============================================================
#                    Main
# ============================================================

def main():
    qsr = qsr_hurmaness(
        audio_path=AUDIO_PATH,
        model_path=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
        language=LANGUAGE,
        chunk_size_sec=CHUNK_SIZE_SEC,
        chunk_overlap_sec=CHUNK_OVERLAP_SEC,
        max_new_tokens=MAX_NEW_TOKENS
    )
    qsr.check_config()

    # --------------------------------------------------------
    # 加载音频
    # --------------------------------------------------------

    audio, duration = qsr.load_audio()

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model = qsr.load_model()

    # --------------------------------------------------------
    # 流式识别
    # --------------------------------------------------------

    final_text = qsr.streaming_transcribe(
        model,
        audio,
        duration
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    qsr.save_text(
        final_text
    )

    print("\n程序结束。")


# ============================================================
#                    程序入口
# ============================================================

if __name__ == "__main__":
    main()