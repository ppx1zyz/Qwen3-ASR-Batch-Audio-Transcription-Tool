import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    TOP,
    X,
    Y,
    Button,
    Entry,
    Frame,
    Label,
    Listbox,
    Menu,
    Message,
    StringVar,
    Tk,
    Text,
    filedialog,
    messagebox,
    ttk,
)

import torch

from asr_server import qsr_hurmaness


# ============================================================
# 基础配置
# ============================================================

CONFIG_FILE = (
    Path(__file__).parent
    / 
    ".qsr_transcriber_config.json"
)

DEFAULT_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class AppConfig:
    input_dir: str = ""
    output_dir: str = ""
    model_path: str = ""

    device: str = "cuda"
    dtype: str = "float16"
    language: str = "zh"

    chunk_size_sec: float = 30.0
    chunk_overlap_sec: float = 5.0
    max_new_tokens: int = 256

    extensions: str = ",".join(DEFAULT_EXTENSIONS)


@dataclass
class FileTask:
    input_path: Path
    output_path: Path


# ============================================================
# 工具函数
# ============================================================

def format_bytes(num_bytes):
    """
    将字节数转换为易读格式。
    """
    if num_bytes is None:
        return "N/A"

    value = float(num_bytes)

    for unit in ("B", "MiB", "GiB", "TiB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PiB"


def format_seconds(seconds):
    """
    将秒数格式化为 HH:MM:SS。
    """
    if seconds is None:
        return "--:--:--"

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_gpu_memory_text():
    """
    获取当前 GPU 显存信息。

    进程使用：
        当前 Python/PyTorch 进程实际分配的显存

    GPU占用：
        整张显卡当前已经使用的显存

    剩余：
        整张显卡当前剩余显存
    """
    if not torch.cuda.is_available():
        return "GPU 不可用"

    try:
        device_index = torch.cuda.current_device()

        free_memory, total_memory = torch.cuda.mem_get_info(
            device_index
        )

        used_memory = total_memory - free_memory
        process_memory = torch.cuda.memory_allocated(
            device_index
        )
        reserved_memory = torch.cuda.memory_reserved(
            device_index
        )

        device_name = torch.cuda.get_device_name(
            device_index
        )

        return (
            f"{device_name} | "
            f"进程使用: {format_bytes(process_memory)} | "
            f"缓存: {format_bytes(reserved_memory)} | "
            f"GPU占用: {format_bytes(used_memory)} / "
            f"{format_bytes(total_memory)} | "
            f"剩余: {format_bytes(free_memory)}"
        )

    except Exception as exc:
        return f"显存读取失败：{exc}"


def get_dtype(dtype_name):
    """
    将界面中的字符串转换为 PyTorch dtype。
    """
    dtype_name = dtype_name.lower().strip()

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    if dtype_name not in mapping:
        raise ValueError(
            "dtype 必须是 float16、float32 或 bfloat16"
        )

    return mapping[dtype_name]


def load_config():
    """
    读取上次保存的配置。
    """
    config = AppConfig()

    if not CONFIG_FILE.exists():
        return config

    try:
        data = json.loads(
            CONFIG_FILE.read_text(
                encoding="utf-8"
            )
        )

        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    except Exception as exc:
        print("读取配置失败：", exc)

    return config


def save_config(config):
    """
    保存当前配置，用于下次启动时记忆。
    """
    try:
        CONFIG_FILE.write_text(
            json.dumps(
                config.__dict__,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    except Exception as exc:
        print("保存配置失败：", exc)


def parse_extensions(value):
    """
    解析扩展名，例如：

    .wav,.mp3,.flac
    """
    result = set()

    for item in value.split(","):
        item = item.strip().lower()

        if not item:
            continue

        if not item.startswith("."):
            item = "." + item

        result.add(item)

    return result


def find_audio_files(input_dir, extensions):
    """
    获取输入目录下的音频文件。

    当前只处理当前目录，不递归子目录。
    如果以后需要递归，可以改成 rglob。
    """
    input_dir = Path(input_dir)

    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    ]

    return sorted(
        files,
        key=lambda path: path.name.lower()
    )


class StopRequested(Exception):
    """
    用户点击停止时抛出的内部异常。
    """
    pass


# ============================================================
# 主界面
# ============================================================

class TranscriberApp:

    def __init__(self, root):
        self.root = root
        self.root.title("批量音频转文字工具")
        self.root.geometry("1100x760")
        self.root.minsize(900, 650)

        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None

        self.config = load_config()

        self.create_variables()
        self.create_widgets()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

        self.root.after(
            100,
            self.poll_events
        )

    # --------------------------------------------------------
    # 界面变量
    # --------------------------------------------------------

    def create_variables(self):
        config = self.config

        self.input_dir_var = StringVar(
            value=config.input_dir
        )

        self.output_dir_var = StringVar(
            value=config.output_dir
        )

        self.model_path_var = StringVar(
            value=config.model_path
        )

        self.device_var = StringVar(
            value=config.device
        )

        self.dtype_var = StringVar(
            value=config.dtype
        )

        self.language_var = StringVar(
            value=config.language
        )

        self.chunk_size_var = StringVar(
            value=str(config.chunk_size_sec)
        )

        self.chunk_overlap_var = StringVar(
            value=str(config.chunk_overlap_sec)
        )

        self.max_new_tokens_var = StringVar(
            value=str(config.max_new_tokens)
        )

        self.extensions_var = StringVar(
            value=config.extensions
        )

        self.status_var = StringVar(
            value="等待开始"
        )

        self.file_progress_var = StringVar(
            value="文件进度：0/0"
        )

        self.chunk_progress_var = StringVar(
            value="Chunk：0/0"
        )

        self.char_count_var = StringVar(
            value="当前字数：0"
        )

        self.eta_var = StringVar(
            value="预计剩余：--:--:--"
        )

        self.memory_var = StringVar(
            value="显存：未检测"
        )

    # --------------------------------------------------------
    # 创建界面
    # --------------------------------------------------------

    def create_widgets(self):
        self.create_path_area()
        self.create_parameter_area()
        self.create_control_area()
        self.create_progress_area()
        self.create_output_area()

    def create_path_area(self):
        frame = ttk.LabelFrame(
            self.root,
            text="路径设置"
        )

        frame.pack(
            fill=X,
            padx=10,
            pady=8
        )

        self.add_path_row(
            frame,
            row=0,
            label="输入文件夹：",
            variable=self.input_dir_var,
            command=self.select_input_dir
        )

        self.add_path_row(
            frame,
            row=1,
            label="输出文件夹：",
            variable=self.output_dir_var,
            command=self.select_output_dir
        )

        self.add_path_row(
            frame,
            row=2,
            label="模型目录：",
            variable=self.model_path_var,
            command=self.select_model_dir
        )

    def add_path_row(
        self,
        parent,
        row,
        label,
        variable,
        command
    ):
        ttk.Label(
            parent,
            text=label,
            width=14
        ).grid(
            row=row,
            column=0,
            padx=5,
            pady=5,
            sticky="w"
        )

        entry = ttk.Entry(
            parent,
            textvariable=variable
        )

        entry.grid(
            row=row,
            column=1,
            padx=5,
            pady=5,
            sticky="ew"
        )

        ttk.Button(
            parent,
            text="选择",
            command=command
        ).grid(
            row=row,
            column=2,
            padx=5,
            pady=5
        )

        parent.columnconfigure(
            1,
            weight=1
        )

    def create_parameter_area(self):
        frame = ttk.LabelFrame(
            self.root,
            text="识别参数"
        )

        frame.pack(
            fill=X,
            padx=10,
            pady=5
        )

        self.add_parameter(
            frame,
            row=0,
            column=0,
            label="设备",
            variable=self.device_var
        )

        self.add_parameter(
            frame,
            row=0,
            column=2,
            label="数据类型",
            variable=self.dtype_var
        )

        self.add_parameter(
            frame,
            row=0,
            column=4,
            label="语言",
            variable=self.language_var
        )

        self.add_parameter(
            frame,
            row=1,
            column=0,
            label="Chunk秒数",
            variable=self.chunk_size_var
        )

        self.add_parameter(
            frame,
            row=1,
            column=2,
            label="重叠秒数",
            variable=self.chunk_overlap_var
        )

        self.add_parameter(
            frame,
            row=1,
            column=4,
            label="最大Token",
            variable=self.max_new_tokens_var
        )

        ttk.Label(
            frame,
            text="音频扩展名"
        ).grid(
            row=2,
            column=0,
            padx=5,
            pady=5,
            sticky="w"
        )

        ttk.Entry(
            frame,
            textvariable=self.extensions_var,
            width=75
        ).grid(
            row=2,
            column=1,
            columnspan=5,
            padx=5,
            pady=5,
            sticky="ew"
        )

        for column in (1, 3, 5):
            frame.columnconfigure(
                column,
                weight=1
            )

    def add_parameter(
        self,
        parent,
        row,
        column,
        label,
        variable
    ):
        ttk.Label(
            parent,
            text=label
        ).grid(
            row=row,
            column=column,
            padx=5,
            pady=5,
            sticky="w"
        )

        ttk.Entry(
            parent,
            textvariable=variable,
            width=15
        ).grid(
            row=row,
            column=column + 1,
            padx=5,
            pady=5,
            sticky="ew"
        )

    def create_control_area(self):
        frame = ttk.Frame(
            self.root
        )

        frame.pack(
            fill=X,
            padx=10,
            pady=8
        )

        self.start_button = ttk.Button(
            frame,
            text="开始批量转换",
            command=self.start_conversion
        )

        self.start_button.pack(
            side=LEFT,
            padx=5
        )

        self.stop_button = ttk.Button(
            frame,
            text="停止",
            command=self.stop_conversion,
            state="disabled"
        )

        self.stop_button.pack(
            side=LEFT,
            padx=5
        )

        ttk.Label(
            frame,
            textvariable=self.status_var
        ).pack(
            side=LEFT,
            padx=20
        )

    def create_progress_area(self):
        frame = ttk.LabelFrame(
            self.root,
            text="转换状态"
        )

        frame.pack(
            fill=X,
            padx=10,
            pady=5
        )

        self.progress_bar = ttk.Progressbar(
            frame,
            orient="horizontal",
            mode="determinate"
        )

        self.progress_bar.pack(
            fill=X,
            padx=8,
            pady=8
        )

        info_frame = ttk.Frame(
            frame
        )

        info_frame.pack(
            fill=X,
            padx=8,
            pady=5
        )

        ttk.Label(
            info_frame,
            textvariable=self.file_progress_var
        ).pack(
            side=LEFT,
            padx=10
        )

        ttk.Label(
            info_frame,
            textvariable=self.chunk_progress_var
        ).pack(
            side=LEFT,
            padx=10
        )

        ttk.Label(
            info_frame,
            textvariable=self.char_count_var
        ).pack(
            side=LEFT,
            padx=10
        )

        ttk.Label(
            info_frame,
            textvariable=self.eta_var
        ).pack(
            side=LEFT,
            padx=10
        )

        ttk.Label(
            frame,
            textvariable=self.memory_var
        ).pack(
            fill=X,
            padx=18,
            pady=5,
            anchor="w"
        )

    def create_output_area(self):
        frame = ttk.LabelFrame(
            self.root,
            text="实时识别结果"
        )

        frame.pack(
            fill=BOTH,
            expand=True,
            padx=10,
            pady=8
        )

        self.result_text = Text(
            frame,
            wrap="word",
            height=18
        )

        self.result_text.pack(
            side=LEFT,
            fill=BOTH,
            expand=True
        )

        scrollbar = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=self.result_text.yview
        )

        scrollbar.pack(
            side=RIGHT,
            fill=Y
        )

        self.result_text.configure(
            yscrollcommand=scrollbar.set
        )

    # --------------------------------------------------------
    # 路径选择
    # --------------------------------------------------------

    def select_input_dir(self):
        path = filedialog.askdirectory(
            title="选择输入音频文件夹"
        )

        if path:
            self.input_dir_var.set(path)

            if not self.output_dir_var.get().strip():
                self.output_dir_var.set(
                    str(Path(path) / "txt_output")
                )

    def select_output_dir(self):
        path = filedialog.askdirectory(
            title="选择输出文件夹"
        )

        if path:
            self.output_dir_var.set(path)

    def select_model_dir(self):
        path = filedialog.askdirectory(
            title="选择模型目录"
        )

        if path:
            self.model_path_var.set(path)

    # --------------------------------------------------------
    # 配置读取与校验
    # --------------------------------------------------------

    def collect_config(self):
        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        model_path = self.model_path_var.get().strip()

        if not input_dir:
            raise ValueError("请选择输入文件夹")

        if not output_dir:
            raise ValueError("请选择输出文件夹")

        if not model_path:
            raise ValueError("请选择模型目录")

        input_path = Path(input_dir)

        if not input_path.exists():
            raise ValueError(
                f"输入文件夹不存在：{input_path}"
            )

        return AppConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=model_path,
            device=self.device_var.get().strip(),
            dtype=self.dtype_var.get().strip(),
            language=self.language_var.get().strip(),
            chunk_size_sec=float(
                self.chunk_size_var.get().strip()
            ),
            chunk_overlap_sec=float(
                self.chunk_overlap_var.get().strip()
            ),
            max_new_tokens=int(
                self.max_new_tokens_var.get().strip()
            ),
            extensions=self.extensions_var.get().strip()
        )

    # --------------------------------------------------------
    # 生成任务列表
    # --------------------------------------------------------

    def prepare_tasks(self, config):
        input_dir = Path(config.input_dir)
        output_dir = Path(config.output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        extensions = parse_extensions(
            config.extensions
        )

        audio_files = find_audio_files(
            input_dir,
            extensions
        )

        if not audio_files:
            raise ValueError(
                "输入文件夹中没有找到支持的音频文件"
            )

        tasks = []

        for audio_path in audio_files:
            output_path = (
                output_dir
                / f"{audio_path.stem}.txt"
            )

            if output_path.exists():
                overwrite = messagebox.askyesno(
                    "发现已有文本文件",
                    (
                        f"文件：\n{output_path.name}\n\n"
                        "是否覆盖？\n\n"
                        "选择“是”：覆盖\n"
                        "选择“否”：跳过"
                    )
                )

                if not overwrite:
                    self.append_log(
                        f"[跳过] {audio_path.name}"
                    )
                    continue

            tasks.append(
                FileTask(
                    input_path=audio_path,
                    output_path=output_path
                )
            )

        return tasks

    # --------------------------------------------------------
    # 启动转换
    # --------------------------------------------------------

    def start_conversion(self):
        if (
            self.worker_thread is not None
            and self.worker_thread.is_alive()
        ):
            messagebox.showinfo(
                "提示",
                "当前已经有任务正在运行"
            )
            return

        try:
            config = self.collect_config()
            save_config(config)

            tasks = self.prepare_tasks(config)

            if not tasks:
                messagebox.showinfo(
                    "提示",
                    "没有需要转换的文件"
                )
                return

        except Exception as exc:
            messagebox.showerror(
                "配置错误",
                str(exc)
            )
            return

        self.stop_event.clear()

        self.progress_bar["maximum"] = len(tasks)
        self.progress_bar["value"] = 0

        self.file_progress_var.set(
            f"文件进度：0/{len(tasks)}"
        )

        self.chunk_progress_var.set(
            "Chunk：0/0"
        )

        self.char_count_var.set(
            "当前字数：0"
        )

        self.eta_var.set(
            "预计剩余：--:--:--"
        )

        self.memory_var.set(
            "显存：等待模型加载"
        )

        self.result_text.delete(
            "1.0",
            END
        )

        self.start_button.configure(
            state="disabled"
        )

        self.stop_button.configure(
            state="normal"
        )

        self.status_var.set(
            f"准备转换 {len(tasks)} 个文件"
        )

        self.worker_thread = threading.Thread(
            target=self.worker_main,
            args=(tasks, config),
            daemon=True
        )

        self.worker_thread.start()

    def stop_conversion(self):
        if (
            self.worker_thread is not None
            and self.worker_thread.is_alive()
        ):
            self.stop_event.set()

            self.status_var.set(
                "正在停止，请等待当前 Chunk 结束..."
            )

    # --------------------------------------------------------
    # 后台工作线程
    # --------------------------------------------------------

    def worker_main(self, tasks, config):
        model = None
        qsr = None

        completed_count = 0
        batch_start = time.perf_counter()

        try:
            for index, task in enumerate(tasks, start=1):
                if self.stop_event.is_set():
                    raise StopRequested()

                self.events.put(
                    (
                        "file_start",
                        {
                            "index": index,
                            "total": len(tasks),
                            "input_name": task.input_path.name,
                            "output_name": task.output_path.name,
                        }
                    )
                )

                # ------------------------------------------------
                # 第一个文件时初始化识别器并加载模型
                # 后续文件复用模型
                # ------------------------------------------------

                if qsr is None:
                    qsr = qsr_hurmaness(
                        audio_path=str(task.input_path),
                        model_path=config.model_path,
                        device=config.device,
                        dtype=get_dtype(config.dtype),
                        language=config.language,
                        chunk_size_sec=config.chunk_size_sec,
                        chunk_overlap_sec=config.chunk_overlap_sec,
                        max_new_tokens=config.max_new_tokens
                    )

                    qsr.check_config()

                    self.events.put(
                        (
                            "status",
                            "正在加载模型..."
                        )
                    )

                    model = qsr.load_model()

                    self.events.put(
                        (
                            "status",
                            "模型加载完成"
                        )
                    )

                else:
                    # 切换当前音频文件
                    qsr.audio_path = str(
                        task.input_path
                    )

                # ------------------------------------------------
                # 加载当前音频
                # ------------------------------------------------

                self.events.put(
                    (
                        "status",
                        f"正在加载：{task.input_path.name}"
                    )
                )

                audio, duration = qsr.load_audio()

                # ------------------------------------------------
                # 清空当前文件显示区域
                # ------------------------------------------------

                self.events.put(
                    (
                        "clear_result",
                        None
                    )
                )

                # ------------------------------------------------
                # 接收每个 Chunk 的识别结果
                # ------------------------------------------------

                def on_chunk(info):
                    if self.stop_event.is_set():
                        raise StopRequested()

                    memory_text = (
                        get_gpu_memory_text()
                    )

                    event_data = dict(info)

                    event_data.update(
                        {
                            "file_index": index,
                            "file_total": len(tasks),
                            "file_name": task.input_path.name,
                            "memory_text": memory_text,
                            "batch_elapsed": (
                                time.perf_counter()
                                - batch_start
                            ),
                        }
                    )

                    self.events.put(
                        (
                            "chunk",
                            event_data
                        )
                    )

                # ------------------------------------------------
                # 流式识别
                # ------------------------------------------------

                final_text = qsr.streaming_transcribe(
                    model,
                    audio,
                    duration,
                    chunk_callback=on_chunk
                )

                if self.stop_event.is_set():
                    raise StopRequested()

                # ------------------------------------------------
                # 保存文本
                # 不调用 qsr.save_text，直接保存到指定目录
                # ------------------------------------------------

                task.output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                task.output_path.write_text(
                    final_text or "",
                    encoding="utf-8"
                )

                completed_count += 1

                self.events.put(
                    (
                        "file_done",
                        {
                            "index": index,
                            "total": len(tasks),
                            "input_name": task.input_path.name,
                            "output_path": str(
                                task.output_path
                            ),
                            "char_count": len(
                                final_text or ""
                            ),
                            "batch_elapsed": (
                                time.perf_counter()
                                - batch_start
                            ),
                        }
                    )
                )

            self.events.put(
                (
                    "all_done",
                    {
                        "completed": completed_count,
                        "total": len(tasks),
                    }
                )
            )

        except StopRequested:
            self.events.put(
                (
                    "stopped",
                    {
                        "completed": completed_count,
                        "total": len(tasks),
                    }
                )
            )

        except Exception as exc:
            self.events.put(
                (
                    "error",
                    {
                        "message": repr(exc),
                        "completed": completed_count,
                        "total": len(tasks),
                    }
                )
            )

    # --------------------------------------------------------
    # 事件处理
    # --------------------------------------------------------

    def poll_events(self):
        try:
            while True:
                event_type, data = (
                    self.events.get_nowait()
                )

                self.handle_event(
                    event_type,
                    data
                )

        except queue.Empty:
            pass

        self.root.after(
            100,
            self.poll_events
        )

    def handle_event(self, event_type, data):
        if event_type == "status":
            self.status_var.set(data)
            self.append_log(data)

        elif event_type == "file_start":
            self.file_progress_var.set(
                f"文件进度：{data['index'] - 1}/"
                f"{data['total']}，"
                f"当前：{data['input_name']}"
            )

            self.chunk_progress_var.set(
                "Chunk：0/0"
            )

            self.char_count_var.set(
                "当前字数：0"
            )

            self.eta_var.set(
                "预计剩余：--:--:--"
            )

            self.append_log(
                f"\n开始处理：{data['input_name']}"
            )

        elif event_type == "clear_result":
            self.result_text.delete(
                "1.0",
                END
            )

        elif event_type == "chunk":
            chunk_id = data["chunk_id"]
            chunk_count = data["chunk_count"]

            self.chunk_progress_var.set(
                f"Chunk：{chunk_id}/{chunk_count}"
            )

            self.char_count_var.set(
                f"当前字数：{data['char_count']}"
            )

            self.eta_var.set(
                "当前文件预计剩余："
                + format_seconds(
                    data["eta_seconds"]
                )
            )

            self.memory_var.set(
                data["memory_text"]
            )

            # 使用当前累计结果刷新文本区域
            self.result_text.delete(
                "1.0",
                END
            )

            self.result_text.insert(
                END,
                data["text_so_far"]
            )

            self.result_text.see(
                END
            )

            self.status_var.set(
                (
                    f"{data['file_name']} - "
                    f"Chunk {chunk_id}/{chunk_count}"
                )
            )

        elif event_type == "file_done":
            self.progress_bar["value"] = (
                data["index"]
            )

            self.file_progress_var.set(
                f"文件进度：{data['index']}/"
                f"{data['total']}"
            )

            self.char_count_var.set(
                f"当前文件字数：{data['char_count']}"
            )

            # 使用已经完成的文件数估算整个批次剩余时间
            completed = data["index"]
            total = data["total"]
            elapsed = data["batch_elapsed"]

            if completed > 0:
                eta = (
                    elapsed
                    * (total - completed)
                    / completed
                )
            else:
                eta = None

            self.eta_var.set(
                "全部任务预计剩余："
                + format_seconds(eta)
            )

            self.append_log(
                f"[完成] {data['input_name']} -> "
                f"{data['output_path']}，"
                f"字数：{data['char_count']}"
            )

        elif event_type == "all_done":
            self.start_button.configure(
                state="normal"
            )

            self.stop_button.configure(
                state="disabled"
            )

            self.status_var.set(
                f"全部完成：{data['completed']}/"
                f"{data['total']}"
            )

            self.append_log(
                "\n全部文件转换完成"
            )

            messagebox.showinfo(
                "完成",
                (
                    f"转换完成\n\n"
                    f"成功处理：{data['completed']} 个文件"
                )
            )

        elif event_type == "stopped":
            self.start_button.configure(
                state="normal"
            )

            self.stop_button.configure(
                state="disabled"
            )

            self.status_var.set(
                "任务已停止"
            )

            self.append_log(
                "\n用户停止了任务"
            )

        elif event_type == "error":
            self.start_button.configure(
                state="normal"
            )

            self.stop_button.configure(
                state="disabled"
            )

            self.status_var.set(
                "发生错误"
            )

            self.append_log(
                f"\n错误：{data['message']}"
            )

            messagebox.showerror(
                "转换失败",
                data["message"]
            )

    def append_log(self, text):
        self.result_text.insert(
            END,
            f"\n\n{text}"
        )

        self.result_text.see(
            END
        )

    # --------------------------------------------------------
    # 关闭窗口
    # --------------------------------------------------------

    def on_close(self):
        if (
            self.worker_thread is not None
            and self.worker_thread.is_alive()
        ):
            answer = messagebox.askyesno(
                "确认退出",
                "当前仍有任务运行，确定退出吗？"
            )

            if not answer:
                return

            self.stop_event.set()

        try:
            config = self.collect_config()
            save_config(config)
        except Exception:
            pass

        self.root.destroy()


# ============================================================
# 程序入口
# ============================================================

def main():
    root = Tk()

    # 使用 ttk 原生主题
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    app = TranscriberApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()
