"""MLX Whisper Studio - DAW style Tkinter GUI."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional

from output_files import OutputPlan, cleanup_outputs, prepare_outputs, publish_outputs

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    TkinterDnD = None
    DND_FILES = None


BG_MAIN = "#1a1a1a"
BG_CARD = "#2d2d2d"
BG_BORDER = "#404040"
ACCENT = "#00ffcc"
ACCENT_HOVER = "#00e6b8"
ACCENT_ACTIVE = "#00d4a3"
SUCCESS = "#00ff88"
ERROR = "#ff4444"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#a0a0a0"
TEXT_DISABLED = "#666666"

FONT_TITLE = ("Helvetica Neue", 22, "bold")
FONT_LABEL = ("Helvetica Neue", 14, "bold")
FONT_BUTTON = ("Helvetica Neue", 13, "bold")
FONT_INPUT = ("SF Mono", 12)
FONT_SMALL = ("Helvetica Neue", 11)


MODEL_OPTIONS = [
    ("Tiny", "mlx-community/whisper-tiny-mlx"),
    ("Base", "mlx-community/whisper-base-mlx"),
    ("Small", "mlx-community/whisper-small-mlx"),
    ("Medium", "mlx-community/whisper-medium-mlx"),
    ("Large", "mlx-community/whisper-large-mlx"),
    ("Large v2", "mlx-community/whisper-large-v2-mlx"),
    ("Large v3", "mlx-community/whisper-large-v3-mlx"),
    ("Custom...", "custom"),
]

LANGUAGE_OPTIONS = [
    ("Auto (detect)", "auto"),
    ("Chinese (zh)", "zh"),
    ("English (en)", "en"),
    ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Spanish (es)", "es"),
    ("Portuguese (pt)", "pt"),
    ("Italian (it)", "it"),
    ("Thai (th)", "th"),
    ("Vietnamese (vi)", "vi"),
    ("Indonesian (id)", "id"),
    ("Malay (ms)", "ms"),
    ("Russian (ru)", "ru"),
    ("Arabic (ar)", "ar"),
    ("Hindi (hi)", "hi"),
    ("Custom...", "custom"),
]

STATUS_STYLES = {
    "Idle": ("•", TEXT_SECONDARY),
    "Queued": ("•", TEXT_SECONDARY),
    "Processing": ("⟳", ACCENT),
    "Done": ("✓", SUCCESS),
    "Failed": ("✗", ERROR),
    "Cancelled": ("✗", ERROR),
}

DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ja-zh"


@dataclass
class FileEntry:
    path: str
    size_mb: float
    status: str = "Idle"


class MLXWhisperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MLX Whisper Studio")
        self.root.geometry("1040x760")
        self.root.minsize(900, 640)
        self.root.configure(bg=BG_MAIN)

        self.queue: List[str] = []
        self.queue_entries: List[FileEntry] = []
        self.queue_results: List[str] = []
        self.file_entries: List[FileEntry] = []
        self.current_index = -1
        self.current_process: Optional[subprocess.Popen] = None
        self.current_plan: Optional[OutputPlan] = None
        self.cancel_requested = False
        self.is_running = False
        self.closing = False
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.options_visible = True
        self.log_visible = False

        self._apply_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_logs()

    def _apply_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Primary.TButton",
            background=ACCENT,
            foreground=BG_MAIN,
            font=FONT_BUTTON,
            padding=(12, 8),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", ACCENT_HOVER), ("pressed", ACCENT_ACTIVE)],
            foreground=[("disabled", TEXT_DISABLED)],
        )
        style.configure(
            "Secondary.TButton",
            background=BG_CARD,
            foreground=ACCENT,
            font=FONT_BUTTON,
            padding=(8, 6),
            borderwidth=1,
            relief="solid",
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#353535"), ("pressed", "#2a2a2a")],
            foreground=[("disabled", TEXT_DISABLED)],
        )
        style.configure(
            "Ghost.TButton",
            background=BG_MAIN,
            foreground=TEXT_SECONDARY,
            font=FONT_LABEL,
            padding=(4, 2),
            borderwidth=0,
        )
        style.map(
            "Ghost.TButton",
            foreground=[("active", ACCENT)],
            background=[("active", BG_MAIN)],
        )
        style.configure(
            "Neon.Horizontal.TProgressbar",
            troughcolor=BG_CARD,
            background=ACCENT,
            thickness=10,
        )
        style.configure(
            "TCombobox",
            fieldbackground=BG_CARD,
            background=BG_CARD,
            foreground=TEXT_PRIMARY,
            arrowcolor=ACCENT,
            bordercolor=BG_BORDER,
            font=FONT_INPUT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG_CARD)],
            foreground=[("readonly", TEXT_PRIMARY)],
        )
        style.configure(
            "TCheckbutton",
            background=BG_CARD,
            foreground=TEXT_PRIMARY,
            font=FONT_INPUT,
        )
        style.map(
            "TCheckbutton",
            foreground=[("disabled", TEXT_DISABLED)],
        )

    def _build_ui(self):
        container = tk.Frame(self.root, bg=BG_MAIN)
        container.pack(fill=tk.BOTH, expand=True, padx=24, pady=24)

        title = tk.Label(
            container,
            text="MLX Whisper Studio",
            fg=TEXT_PRIMARY,
            bg=BG_MAIN,
            font=FONT_TITLE,
        )
        title.pack(anchor="w")

        hint_frame = tk.Frame(
            container,
            bg=BG_CARD,
            highlightbackground=BG_BORDER,
            highlightthickness=1,
        )
        hint_frame.pack(fill=tk.X, pady=(10, 12))
        hint_label = tk.Label(
            hint_frame,
            text="Drop files here to transcribe",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        hint_label.pack(anchor="w", padx=14, pady=10)

        file_section = tk.Frame(container, bg=BG_MAIN)
        file_section.pack(fill=tk.BOTH, expand=False, pady=(0, 12))

        file_card = tk.Frame(
            file_section,
            bg=BG_CARD,
            highlightbackground=BG_BORDER,
            highlightthickness=1,
        )
        file_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.file_list = tk.Listbox(
            file_card,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            selectbackground=BG_BORDER,
            selectforeground=TEXT_PRIMARY,
            highlightbackground=BG_BORDER,
            highlightcolor=ACCENT,
            highlightthickness=2,
            relief="flat",
            activestyle="none",
            font=FONT_INPUT,
            height=8,
        )
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0), pady=12)
        scrollbar = ttk.Scrollbar(file_card, orient=tk.VERTICAL, command=self.file_list.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=12)
        self.file_list.configure(yscrollcommand=scrollbar.set)

        self.drop_hint = tk.Label(
            file_card,
            text="Release to add files",
            fg=ACCENT,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        self.drop_hint.place_forget()

        button_card = tk.Frame(
            file_section,
            bg=BG_CARD,
            highlightbackground=BG_BORDER,
            highlightthickness=1,
        )
        button_card.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 0))

        self.add_button = ttk.Button(
            button_card, text="Add Files", style="Primary.TButton", command=self._select_files
        )
        self.add_button.pack(fill=tk.X, padx=14, pady=(14, 8))
        self.remove_button = ttk.Button(
            button_card,
            text="Remove",
            style="Secondary.TButton",
            command=self._remove_selected,
        )
        self.remove_button.pack(fill=tk.X, padx=14, pady=8)
        self.clear_button = ttk.Button(
            button_card, text="Clear", style="Secondary.TButton", command=self._clear_files
        )
        self.clear_button.pack(fill=tk.X, padx=14, pady=(8, 14))

        if TkinterDnD is not None:
            self.file_list.drop_target_register(DND_FILES)
            self.file_list.dnd_bind("<<DropEnter>>", self._on_drag_enter)
            self.file_list.dnd_bind("<<DropLeave>>", self._on_drag_leave)
            self.file_list.dnd_bind("<<Drop>>", self._on_drop)

        self.options_toggle = ttk.Button(
            container,
            text="▼ Advanced Options",
            style="Ghost.TButton",
            command=self._toggle_options,
        )
        self.options_toggle.pack(fill=tk.X, pady=(4, 0))

        self.options_frame = tk.Frame(
            container,
            bg=BG_CARD,
            highlightbackground=BG_BORDER,
            highlightthickness=1,
        )
        self.options_frame.pack(fill=tk.X, pady=(6, 16))
        self.options_frame.columnconfigure(1, weight=1)
        self.options_frame.columnconfigure(3, weight=1)

        model_label = tk.Label(
            self.options_frame,
            text="Model",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        model_label.grid(row=0, column=0, sticky="w", padx=14, pady=10)
        model_names = [label for label, _ in MODEL_OPTIONS]
        self.model_combo = ttk.Combobox(
            self.options_frame, values=model_names, state="readonly", width=22
        )
        self.model_combo.configure(font=FONT_INPUT)
        self.model_combo.current(len(MODEL_OPTIONS) - 2)
        self.model_combo.grid(row=0, column=1, sticky="w", padx=14, pady=10)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        language_label = tk.Label(
            self.options_frame,
            text="Language",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        language_label.grid(row=0, column=2, sticky="w", padx=14, pady=10)
        lang_names = [label for label, _ in LANGUAGE_OPTIONS]
        self.language_combo = ttk.Combobox(
            self.options_frame, values=lang_names, state="readonly", width=22
        )
        self.language_combo.configure(font=FONT_INPUT)
        self.language_combo.current(0)
        self.language_combo.grid(row=0, column=3, sticky="w", padx=14, pady=10)
        self.language_combo.bind("<<ComboboxSelected>>", self._on_language_change)

        custom_model_label = tk.Label(
            self.options_frame,
            text="Custom Model",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        custom_model_label.grid(row=1, column=0, sticky="w", padx=14, pady=8)
        self.custom_model_var = tk.StringVar()
        self.custom_model_entry = tk.Entry(
            self.options_frame,
            textvariable=self.custom_model_var,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="solid",
            borderwidth=1,
            highlightbackground=BG_BORDER,
            highlightcolor=ACCENT,
            font=FONT_INPUT,
        )
        self.custom_model_entry.grid(row=1, column=1, sticky="ew", padx=14, pady=8, ipady=4)
        self.custom_model_entry.configure(state="disabled")

        custom_lang_label = tk.Label(
            self.options_frame,
            text="Custom Code",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        custom_lang_label.grid(row=1, column=2, sticky="w", padx=14, pady=8)
        self.custom_lang_var = tk.StringVar()
        self.custom_lang_entry = tk.Entry(
            self.options_frame,
            textvariable=self.custom_lang_var,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="solid",
            borderwidth=1,
            highlightbackground=BG_BORDER,
            highlightcolor=ACCENT,
            font=FONT_INPUT,
        )
        self.custom_lang_entry.grid(row=1, column=3, sticky="ew", padx=14, pady=8, ipady=4)
        self.custom_lang_entry.configure(state="disabled")

        prompt_label = tk.Label(
            self.options_frame,
            text="Prompt",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        prompt_label.grid(row=2, column=0, sticky="nw", padx=14, pady=8)
        self.prompt_text = tk.Text(
            self.options_frame,
            height=5,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="solid",
            borderwidth=1,
            highlightbackground=BG_BORDER,
            highlightcolor=ACCENT,
            font=FONT_INPUT,
        )
        self.prompt_text.grid(row=2, column=1, columnspan=3, sticky="ew", padx=14, pady=8)

        format_label = tk.Label(
            self.options_frame,
            text="Output",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        format_label.grid(row=3, column=0, sticky="w", padx=14, pady=8)
        self.format_txt = tk.BooleanVar(value=True)
        self.format_srt = tk.BooleanVar(value=True)
        self.format_vtt = tk.BooleanVar(value=False)
        self.format_json = tk.BooleanVar(value=False)
        formats_frame = tk.Frame(self.options_frame, bg=BG_CARD)
        formats_frame.grid(row=3, column=1, columnspan=3, sticky="w", padx=14, pady=8)
        ttk.Checkbutton(formats_frame, text="TXT", variable=self.format_txt).pack(
            side=tk.LEFT
        )
        ttk.Checkbutton(formats_frame, text="SRT", variable=self.format_srt).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Checkbutton(formats_frame, text="VTT", variable=self.format_vtt).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Checkbutton(formats_frame, text="JSON", variable=self.format_json).pack(
            side=tk.LEFT, padx=8
        )

        translate_label = tk.Label(
            self.options_frame,
            text="Translate",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        translate_label.grid(row=4, column=0, sticky="w", padx=14, pady=8)
        self.translate_var = tk.BooleanVar(value=False)
        self.translate_check = ttk.Checkbutton(
            self.options_frame,
            text="Chinese (offline)",
            variable=self.translate_var,
            command=self._on_translate_change,
        )
        self.translate_check.grid(row=4, column=1, sticky="w", padx=14, pady=8)
        self.translation_model_var = tk.StringVar()
        self.translation_model_entry = tk.Entry(
            self.options_frame,
            textvariable=self.translation_model_var,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="solid",
            borderwidth=1,
            highlightbackground=BG_BORDER,
            highlightcolor=ACCENT,
            font=FONT_INPUT,
        )
        self.translation_model_entry.grid(row=4, column=2, columnspan=2, sticky="ew", padx=14, pady=8)
        self.translation_model_entry.insert(0, DEFAULT_TRANSLATION_MODEL)
        self.translation_model_entry.configure(state="disabled")

        output_label = tk.Label(
            self.options_frame,
            text="Output Folder",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        output_label.grid(row=5, column=0, sticky="w", padx=14, pady=8)
        self.output_dir_var = tk.StringVar()
        self.output_dir_entry = tk.Entry(
            self.options_frame,
            textvariable=self.output_dir_var,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="solid",
            borderwidth=1,
            highlightbackground=BG_BORDER,
            highlightcolor=ACCENT,
            font=FONT_INPUT,
        )
        self.output_dir_entry.grid(row=5, column=1, sticky="ew", padx=14, pady=8, ipady=4)
        self.output_dir_button = ttk.Button(
            self.options_frame,
            text="Browse",
            style="Secondary.TButton",
            command=self._select_output_dir,
        )
        self.output_dir_button.grid(row=5, column=2, sticky="w", padx=14, pady=8)

        self.control_frame = tk.Frame(container, bg=BG_MAIN)
        self.control_frame.pack(fill=tk.X, pady=(0, 12))
        self.start_button = ttk.Button(
            self.control_frame,
            text="▶ START TRANSCRIPTION",
            style="Primary.TButton",
            command=self._start_queue,
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=12)
        self.cancel_button = ttk.Button(
            self.control_frame,
            text="Cancel",
            style="Secondary.TButton",
            command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(12, 0), ipady=12)
        self.retry_button = ttk.Button(
            self.control_frame,
            text="Retry Failed / Cancelled",
            style="Secondary.TButton",
            command=lambda: self._start_queue(retry_only=True),
            state="disabled",
        )
        self.retry_button.pack(side=tk.LEFT, padx=(12, 0), ipady=12)

        progress_frame = tk.Frame(
            container,
            bg=BG_CARD,
            highlightbackground=BG_BORDER,
            highlightthickness=1,
        )
        progress_frame.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(
            progress_frame,
            text="Idle",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_LABEL,
        )
        self.status_label.pack(anchor="w", padx=14, pady=(12, 6))

        self.progress = ttk.Progressbar(
            progress_frame, mode="indeterminate", style="Neon.Horizontal.TProgressbar"
        )
        self.progress.pack(fill=tk.X, padx=14)

        self.current_file_label = tk.Label(
            progress_frame,
            text="",
            fg=TEXT_SECONDARY,
            bg=BG_CARD,
            font=FONT_SMALL,
        )
        self.current_file_label.pack(anchor="w", padx=14, pady=(8, 6))

        self.log_toggle = ttk.Button(
            progress_frame,
            text="Show Logs ▾",
            style="Ghost.TButton",
            command=self._toggle_logs,
        )
        self.log_toggle.pack(anchor="w", padx=14, pady=(0, 8))

        self.log_frame = tk.Frame(progress_frame, bg=BG_CARD)
        self.log_text = tk.Text(
            self.log_frame,
            height=9,
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="solid",
            borderwidth=1,
            highlightbackground=BG_BORDER,
            highlightcolor=ACCENT,
            font=FONT_INPUT,
            state="disabled",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _toggle_options(self):
        self.options_visible = not self.options_visible
        if self.options_visible:
            self.options_frame.pack(fill=tk.X, pady=(6, 14), before=self.control_frame)
            self.options_toggle.configure(text="▼ Advanced Options")
        else:
            self.options_frame.pack_forget()
            self.options_toggle.configure(text="▶ Advanced Options")

    def _toggle_logs(self):
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
            self.log_toggle.configure(text="Hide Logs ▴")
        else:
            self.log_frame.pack_forget()
            self.log_toggle.configure(text="Show Logs ▾")

    def _select_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[
                (
                    "Audio/Video",
                    "*.mp3 *.wav *.m4a *.mp4 *.mov *.aac *.flac *.ogg",
                ),
                ("All files", "*.*"),
            ],
        )
        self._add_files(list(paths))

    def _add_files(self, files: List[str]):
        if self.is_running or self.closing:
            return
        existing = {entry.path for entry in self.file_entries}
        for path in files:
            if not path or not os.path.isfile(path):
                continue
            if path in existing:
                continue
            size_mb = os.path.getsize(path) / (1024 * 1024)
            entry = FileEntry(path=path, size_mb=size_mb)
            self.file_entries.append(entry)
            self.file_list.insert(tk.END, self._format_entry(entry))
            self.file_list.itemconfig(tk.END, foreground=STATUS_STYLES["Idle"][1])
            existing.add(path)
        if files:
            self._log("Added files.")

    def _format_entry(self, entry: FileEntry) -> str:
        icon, _color = STATUS_STYLES.get(entry.status, ("•", TEXT_SECONDARY))
        size = f"{entry.size_mb:.1f} MB"
        name = os.path.basename(entry.path)
        return f"🎵 {name} | {size} | {icon} {entry.status}"

    def _remove_selected(self):
        if self.is_running or self.closing:
            return
        selection = list(self.file_list.curselection())
        for index in reversed(selection):
            self.file_list.delete(index)
            if 0 <= index < len(self.file_entries):
                self.file_entries.pop(index)
        self._refresh_retry_button()

    def _clear_files(self):
        if self.is_running or self.closing:
            return
        self.file_list.delete(0, tk.END)
        self.file_entries.clear()
        self._refresh_retry_button()

    def _select_output_dir(self):
        directory = filedialog.askdirectory(title="Select output folder")
        if directory:
            self.output_dir_var.set(directory)

    def _on_drag_enter(self, _event=None):
        self._set_drop_highlight(True)

    def _on_drag_leave(self, _event=None):
        self._set_drop_highlight(False)

    def _set_drop_highlight(self, active: bool):
        if active:
            self.file_list.configure(highlightbackground=ACCENT, highlightthickness=2)
            self.drop_hint.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.file_list.configure(highlightbackground=BG_BORDER, highlightthickness=2)
            self.drop_hint.place_forget()

    def _on_drop(self, event):
        self._set_drop_highlight(False)
        if not event.data:
            return
        paths = self.root.tk.splitlist(event.data)
        self._add_files(list(paths))

    def _on_language_change(self, _event=None):
        code = self._language_code()
        if code == "custom":
            self.custom_lang_entry.configure(state="normal")
        else:
            self.custom_lang_entry.configure(state="disabled")

    def _on_model_change(self, _event=None):
        if self._selected_model_label() == "Custom...":
            self.custom_model_entry.configure(state="normal")
        else:
            self.custom_model_entry.configure(state="disabled")

    def _on_translate_change(self):
        if self.translate_var.get():
            self.translation_model_entry.configure(state="normal")
        else:
            self.translation_model_entry.configure(state="disabled")

    def _selected_model_label(self) -> str:
        return self.model_combo.get()

    def _language_code(self) -> str:
        selection = self.language_combo.get()
        for label, code in LANGUAGE_OPTIONS:
            if label == selection:
                return code
        return "auto"

    def _selected_formats(self) -> List[str]:
        formats = []
        if self.format_txt.get():
            formats.append("txt")
        if self.format_srt.get():
            formats.append("srt")
        if self.format_vtt.get():
            formats.append("vtt")
        if self.format_json.get():
            formats.append("json")
        return formats

    def _start_queue(self, retry_only=False):
        if self.is_running or self.closing or (self.worker_thread and self.worker_thread.is_alive()):
            return
        if self.file_list.size() == 0:
            messagebox.showwarning("No files", "Please add at least one file.")
            return
        formats = self._selected_formats()
        if not formats:
            messagebox.showwarning("No formats", "Select at least one output format.")
            return

        entries = [entry for entry in self.file_entries
                   if not retry_only or entry.status in {"Failed", "Cancelled"}]
        if not entries:
            return
        self.queue_entries = entries
        self.queue = [entry.path for entry in entries]
        self.queue_results = []
        # Snapshot Tk values on the UI thread. Changing options mid-run must not
        # change the worker's files or make cleanup target a different output set.
        self.queue_formats = list(formats)
        self.queue_output_root = self.output_dir_var.get().strip()
        self.queue_worker_args = self._build_worker_args("", "")
        self.current_index = -1
        self.cancel_requested = False
        self.is_running = True
        for entry in entries:
            entry.status = "Queued"
        for idx in range(len(entries)):
            self._mark_item_status(idx, "Queued")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.retry_button.configure(state="disabled")
        for button in (self.add_button, self.remove_button, self.clear_button):
            button.configure(state="disabled")
        self.progress.start(10)
        self._set_status("Processing", ACCENT)
        self._log("Starting queue...")

        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        try:
            self.worker_thread.start()
        except RuntimeError as exc:
            self._log(f"Could not start transcription queue: {exc}")
            self.queue_results = ["Failed"] * len(self.queue)
            for idx in range(len(self.queue)):
                self._mark_item_status(idx, "Failed")
            self._finish_queue()

    def _process_queue(self):
        try:
            for idx, file_path in enumerate(self.queue):
                if self.cancel_requested:
                    break
                self.current_index = idx
                self.current_plan = None
                self._post_ui(
                    lambda idx=idx, file_path=file_path: self._update_current_file(
                        f"Processing {idx + 1}/{len(self.queue)}: {os.path.basename(file_path)}"
                    )
                )
                self._mark_item_status(idx, "Processing")
                status = "Failed"
                try:
                    self.current_plan = self._build_output_plan(file_path)
                    success = self._run_worker(file_path, self.current_plan.staging_dir)
                    if success and not self.cancel_requested:
                        outputs = publish_outputs(self.current_plan)
                        self.log_queue.put("Saved: " + ", ".join(outputs))
                        status = "Done"
                    elif self.cancel_requested:
                        status = "Cancelled"
                except Exception as exc:
                    self.log_queue.put(f"Transcription failed: {exc}")
                finally:
                    try:
                        self._cleanup_outputs(self.current_plan)
                    except OSError as exc:
                        self.log_queue.put(f"Could not remove temporary output: {exc}")
                    self.current_plan = None
                self._mark_item_status(idx, status)
                self.queue_results.append(status)
        finally:
            # Unstarted jobs in a cancelled queue must remain retryable too.
            for idx in range(len(self.queue_results), len(self.queue)):
                status = "Cancelled" if self.cancel_requested else "Failed"
                self.queue_results.append(status)
                self._mark_item_status(idx, status)
            self._post_ui(self._finish_queue)

    def _run_worker(self, file_path: str, output_dir: str) -> bool:
        args = list(self.queue_worker_args)
        args[args.index("--input") + 1] = file_path
        args[args.index("--output-dir") + 1] = output_dir
        if self.cancel_requested:
            return False
        process = subprocess.Popen(
            [sys.executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=(os.name == "posix"),
        )
        self.current_process = process
        assert process.stdout is not None

        def read_logs():
            for line in process.stdout:
                self.log_queue.put(line.rstrip())

        reader = threading.Thread(target=read_logs, daemon=True)
        reader.start()
        try:
            while True:
                if self.cancel_requested:
                    self._stop_worker(process)
                    return False
                try:
                    return process.wait(timeout=0.1) == 0
                except subprocess.TimeoutExpired:
                    continue
        finally:
            # Reap the worker before cleanup; otherwise it can recreate files
            # after cancellation or overwrite files belonging to another run.
            if process.poll() is None:
                self._stop_worker(process)
            reader.join(timeout=1)
            if reader.is_alive():
                # ffmpeg or another descendant may still own the stdout pipe,
                # even when the worker itself has exited.
                self._stop_worker(process)
                reader.join(timeout=1)
            if not reader.is_alive():
                process.stdout.close()
            else:
                self.log_queue.put("Worker log stream did not close; stopped waiting.")
            self.current_process = None

    @staticmethod
    def _stop_worker(process):
        def signal_group(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        if os.name == "posix":
            # The worker starts its own session; only its descendants belong to
            # this group. Stop decoders too, without touching unrelated apps.
            signal_group(signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                signal_group(signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        finally:
            if os.name == "posix":
                # A child may ignore SIGTERM even if the group leader exited.
                signal_group(signal.SIGKILL)

    def _build_worker_args(self, file_path: str, output_dir: str) -> List[str]:
        formats = ",".join(self._selected_formats())
        model_label = self._selected_model_label()
        model = dict(MODEL_OPTIONS).get(model_label, MODEL_OPTIONS[-2][1])
        if model == "custom":
            model = self.custom_model_var.get().strip() or MODEL_OPTIONS[-2][1]
        language_code = self._language_code()
        if language_code == "custom":
            language_code = self.custom_lang_var.get().strip() or "auto"
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        args = [
            os.path.join(os.path.dirname(__file__), "mlx_worker.py"),
            "--input",
            file_path,
            "--model",
            model,
            "--language",
            language_code,
            "--formats",
            formats,
            "--output-dir",
            output_dir,
        ]
        if prompt:
            args.extend(["--prompt", prompt])
        if self.translate_var.get():
            args.extend(["--translate-to", "zh"])
            model_override = self.translation_model_var.get().strip()
            if model_override:
                args.extend(["--translation-model", model_override])
        return args

    def _build_output_plan(self, file_path: str) -> OutputPlan:
        return prepare_outputs(file_path, self.queue_formats, self.queue_output_root)

    def _cleanup_outputs(self, plan: OutputPlan | None):
        cleanup_outputs(plan)

    def _cancel_processing(self):
        self.cancel_requested = True
        self._log("Cancel requested. Waiting for the worker to stop...")

    def _on_close(self):
        if self.closing:
            return
        self.closing = True
        if self.is_running or (self.worker_thread and self.worker_thread.is_alive()):
            self._cancel_processing()
            self._set_status("Stopping before closing…", TEXT_SECONDARY)
            self.start_button.configure(state="disabled")
            self.retry_button.configure(state="disabled")
            self.cancel_button.configure(state="disabled")
        self._wait_for_close()

    def _wait_for_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(50, self._wait_for_close)
        else:
            self.root.destroy()

    def _refresh_retry_button(self):
        retryable = any(entry.status in {"Failed", "Cancelled"} for entry in self.file_entries)
        enabled = retryable and not self.is_running and not self.closing
        self.retry_button.configure(state="normal" if enabled else "disabled")

    def _finish_queue(self):
        succeeded = self.queue_results.count("Done")
        failed = self.queue_results.count("Failed")
        cancelled = self.queue_results.count("Cancelled")
        summary = f"{succeeded} succeeded, {failed} failed, {cancelled} cancelled"
        if cancelled:
            self._set_status(f"Cancelled: {summary}", ERROR)
        elif failed:
            self._set_status(f"Finished with errors: {summary}", ERROR)
        else:
            self._set_status(f"Completed: {summary}", SUCCESS)
        self._log(summary + ".")
        self.progress.stop()
        self.is_running = False
        self.start_button.configure(state="disabled" if self.closing else "normal")
        self.cancel_button.configure(state="disabled")
        for button in (self.add_button, self.remove_button, self.clear_button):
            button.configure(state="disabled" if self.closing else "normal")
        self._refresh_retry_button()
        self.current_process = None
        self.cancel_requested = False
        self.current_plan = None
        self.current_file_label.configure(text="")

    def _mark_item_status(self, index: int, status: str):
        entry = self.queue_entries[index]

        def update():
            if entry in self.file_entries:
                # Retry queues contain only a subset of rows. Resolve the entry
                # instead of applying a retry index to the full visible list.
                row = self.file_entries.index(entry)
                entry.status = status
                label = self._format_entry(entry)
                self.file_list.delete(row)
                self.file_list.insert(row, label)
                color = STATUS_STYLES.get(status, ("•", TEXT_SECONDARY))[1]
                self.file_list.itemconfig(row, foreground=color)

        self._post_ui(update)

    def _set_status(self, text: str, color: str):
        self.status_label.configure(text=text, fg=color)

    def _update_current_file(self, text: str):
        self.current_file_label.configure(text=text)

    def _post_ui(self, func):
        self.root.after(0, func)

    def _poll_logs(self):
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            self._log(line)
        self.root.after(150, self._poll_logs)

    def _log(self, message: str):
        if not message:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")


def main():
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = MLXWhisperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
