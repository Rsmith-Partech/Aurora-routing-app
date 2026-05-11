#!/usr/bin/env python3
VERSION = "1.2.0"

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue
import numpy as np
import sounddevice as sd
import sys
import traceback
import time
import datetime

# ----------------------------
# Configuration defaults
# ----------------------------
DEFAULT_SAMPLERATE = 16000
DEFAULT_BLOCKSIZE  = 256
POLL_INTERVAL_MS   = 1000
ROUTE_QUEUE_SIZE   = 32
PRINT_DEBUG        = True

# Silence / starvation detection tuning
SILENCE_THRESHOLD_RMS = 0.001   # below this RMS → silence
SILENCE_WARN_SECONDS  = 3.0     # seconds of silence before warning
STARVE_BLOCK_THRESH   = 10      # consecutive empty output blocks before warning
XRUN_LOG_COOLDOWN     = 5.0     # seconds between repeated xrun log lines


def log(*args):
    if PRINT_DEBUG:
        print("[AudioRouter]", *args)


# ----------------------------
# Helpers
# ----------------------------
def list_devices():
    devices  = sd.query_devices()
    hostapis = sd.query_hostapis()
    result   = []
    for idx, d in enumerate(devices):
        hostapi_name = hostapis[d['hostapi']]['name'] if 'hostapi' in d else 'Unknown'
        result.append({
            'id':     idx,
            'name':   d.get('name', f'Device {idx}'),
            'hostapi': hostapi_name,
            'ins':    int(d.get('max_input_channels', 0)),
            'outs':   int(d.get('max_output_channels', 0)),
        })
    return result


def device_display_str(dev):
    return f"[{dev['id']}] {dev['name']} — {dev['hostapi']} (in:{dev['ins']}, out:{dev['outs']})"


def adapt_channels(data: np.ndarray, out_channels: int) -> np.ndarray:
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    frames, in_ch = data.shape
    if in_ch == out_channels:
        return data
    if in_ch == 1 and out_channels == 2:
        return np.repeat(data, 2, axis=1)
    if in_ch == 2 and out_channels == 1:
        return np.mean(data, axis=1, keepdims=True)
    if in_ch > out_channels:
        return data[:, :out_channels]
    reps  = int(np.ceil(out_channels / in_ch))
    tiled = np.tile(data, (1, reps))
    return tiled[:, :out_channels]


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ----------------------------
# Thread-safe Event Log
# ----------------------------
class EventLog:
    MAX_ENTRIES = 2000

    def __init__(self):
        self._lock      = threading.Lock()
        self._entries   = []          # list of (timestamp_str, level, message)
        self._callbacks = []          # GUI callbacks; called outside the lock

    def add_callback(self, cb):
        with self._lock:
            self._callbacks.append(cb)

    def _emit(self, level: str, message: str):
        ts    = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = (ts, level, message)
        cbs   = []
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.MAX_ENTRIES:
                self._entries = self._entries[-self.MAX_ENTRIES:]
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(entry)
            except Exception:
                pass
        if PRINT_DEBUG:
            print(f"[{ts}] [{level}] {message}")

    def info(self, msg: str):  self._emit("INFO",  msg)
    def warn(self, msg: str):  self._emit("WARN",  msg)
    def error(self, msg: str): self._emit("ERROR", msg)

    def get_all(self):
        with self._lock:
            return list(self._entries)

    def clear(self):
        with self._lock:
            self._entries.clear()


# ----------------------------
# Thread-safe runtime state (no Tk access)
# ----------------------------
class RtState:
    def __init__(self):
        self._lock   = threading.Lock()
        self.inA_gain  = 1.0;  self.outA_gain = 1.0
        self.inB_gain  = 1.0;  self.outB_gain = 1.0
        self.mute_inA  = False; self.mute_outA = False
        self.mute_inB  = False; self.mute_outB = False
        self.mute_all  = False

    def update(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def gain_in_A(self) -> float:
        with self._lock:
            return 0.0 if (self.mute_all or self.mute_inA) else clamp01(self.inA_gain)

    def gain_out_A(self) -> float:
        with self._lock:
            return 0.0 if (self.mute_all or self.mute_outA) else clamp01(self.outA_gain)

    def gain_in_B(self) -> float:
        with self._lock:
            return 0.0 if (self.mute_all or self.mute_inB) else clamp01(self.inB_gain)

    def gain_out_B(self) -> float:
        with self._lock:
            return 0.0 if (self.mute_all or self.mute_outB) else clamp01(self.outB_gain)


# ----------------------------
# Audio Routing Engine
# ----------------------------
class AudioRoute:
    """
    Routes audio from one device's input to another's output via a queue.
    Emits diagnostic events to EventLog for silence, xruns, and queue starvation.
    """

    def __init__(self, input_dev_id: int, output_dev_id: int,
                 samplerate: int, blocksize: int,
                 get_in_gain, get_out_gain,
                 name: str = "A_to_B",
                 event_log: EventLog = None):
        self.input_dev_id  = int(input_dev_id)
        self.output_dev_id = int(output_dev_id)
        self.samplerate    = int(samplerate)
        self.blocksize     = int(blocksize)
        self.get_in_gain   = get_in_gain
        self.get_out_gain  = get_out_gain
        self.name          = name
        self.event_log     = event_log

        in_dev  = sd.query_devices(self.input_dev_id)
        out_dev = sd.query_devices(self.output_dev_id)
        self.in_channels  = max(1, min(2, int(in_dev['max_input_channels'])))
        self.out_channels = max(1, min(2, int(out_dev['max_output_channels'])))

        self.q           = queue.Queue(maxsize=ROUTE_QUEUE_SIZE)
        self._stop_flag  = threading.Event()
        self._streams_started = False

        # xrun counters
        self._xrun_in        = 0
        self._xrun_out       = 0
        self._last_xrun_log  = 0.0

        # silence detection
        self._silence_start  = None
        self._silence_warned = False

        # queue starvation detection
        self._starve_run     = 0
        self._last_starve_log = 0.0

        self._last_status_print = 0.0

        self.input_stream = sd.InputStream(
            device=self.input_dev_id,
            channels=self.in_channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._input_cb,
        )
        self.output_stream = sd.OutputStream(
            device=self.output_dev_id,
            channels=self.out_channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._output_cb,
        )

    # --- helpers ---

    def _elog_info(self, msg):
        if self.event_log: self.event_log.info(f"[{self.name}] {msg}")

    def _elog_warn(self, msg):
        if self.event_log: self.event_log.warn(f"[{self.name}] {msg}")

    def _elog_error(self, msg):
        if self.event_log: self.event_log.error(f"[{self.name}] {msg}")

    def _throttled_status_print(self, where, status):
        now = time.time()
        if now - self._last_status_print > 2.0:
            log(f"[{self.name}] {where} status: {status}")
            self._last_status_print = now

    # --- stream callbacks (audio thread) ---

    def _input_cb(self, indata, frames, time_info, status):
        if status:
            self._xrun_in += 1
            self._throttled_status_print("Input", status)
            now = time.time()
            if now - self._last_xrun_log > XRUN_LOG_COOLDOWN:
                self._elog_warn(
                    f"Input overflow — status: {status}  "
                    f"(total xruns in: {self._xrun_in})"
                )
                self._last_xrun_log = now

        if self._stop_flag.is_set():
            return

        # Silence detection — measured on raw signal, unaffected by gain/mute
        rms = float(np.sqrt(np.mean(indata ** 2)))
        now = time.time()
        if rms < SILENCE_THRESHOLD_RMS:
            if self._silence_start is None:
                self._silence_start  = now
                self._silence_warned = False
            elif not self._silence_warned and (now - self._silence_start) >= SILENCE_WARN_SECONDS:
                elapsed = now - self._silence_start
                self._elog_warn(
                    f"Input silence for {elapsed:.1f}s (RMS={rms:.6f}) "
                    f"— device may have stopped sending audio"
                )
                self._silence_warned = True
        else:
            if self._silence_warned:
                self._elog_info(f"Input signal restored (RMS={rms:.6f})")
            self._silence_start  = None
            self._silence_warned = False

        g_in = clamp01(self.get_in_gain())
        data = (indata.copy() * g_in).astype(np.float32)

        try:
            self.q.put_nowait(data)
        except queue.Full:
            try: self.q.get_nowait()
            except queue.Empty: pass
            try: self.q.put_nowait(data)
            except queue.Full: pass

    def _output_cb(self, outdata, frames, time_info, status):
        if status:
            self._xrun_out += 1
            self._throttled_status_print("Output", status)
            now = time.time()
            if now - self._last_xrun_log > XRUN_LOG_COOLDOWN:
                self._elog_warn(
                    f"Output underflow — status: {status}  "
                    f"(total xruns out: {self._xrun_out})"
                )
                self._last_xrun_log = now

        if self._stop_flag.is_set():
            outdata.fill(0)
            return

        try:
            data = self.q.get_nowait()
            self._starve_run = 0
        except queue.Empty:
            outdata.fill(0)
            self._starve_run += 1
            now = time.time()
            if self._starve_run > STARVE_BLOCK_THRESH and now - self._last_starve_log > XRUN_LOG_COOLDOWN:
                self._elog_warn(
                    f"Output queue starved — {self._starve_run} consecutive empty blocks; "
                    f"input feed may have stalled"
                )
                self._last_starve_log = now
            return

        try:
            data = adapt_channels(data, self.out_channels)
        except Exception:
            outdata.fill(0)
            return

        if data.shape[0] != frames:
            if data.shape[0] > frames:
                data = data[:frames, :]
            else:
                pad  = np.zeros((frames - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.concatenate([data, pad], axis=0)

        g_out    = clamp01(self.get_out_gain())
        outdata[:] = (data * g_out).astype(np.float32)

    # --- lifecycle ---

    def start(self):
        if self._streams_started:
            return
        self._stop_flag.clear()
        self._xrun_in = self._xrun_out = self._starve_run = 0
        self._silence_start = None;  self._silence_warned = False
        self.input_stream.start()
        self.output_stream.start()
        self._streams_started = True
        msg = (
            f"Started — in_dev={self.input_dev_id}, out_dev={self.output_dev_id}, "
            f"sr={self.samplerate}, bs={self.blocksize}, "
            f"in_ch={self.in_channels}, out_ch={self.out_channels}"
        )
        log(f"[{self.name}] {msg}")
        self._elog_info(msg)

    def stop(self):
        if not self._streams_started:
            return
        self._stop_flag.set()
        try:
            self.input_stream.abort()
            self.output_stream.abort()
        except Exception:
            try: self.input_stream.stop()
            except Exception: pass
            try: self.output_stream.stop()
            except Exception: pass
        finally:
            try: self.input_stream.close()
            except Exception: pass
            try: self.output_stream.close()
            except Exception: pass
            self._streams_started = False
            try:
                while True: self.q.get_nowait()
            except queue.Empty:
                pass
        summary = (
            f"Stopped — xruns in:{self._xrun_in}  out:{self._xrun_out}"
        )
        log(f"[{self.name}] {summary}")
        self._elog_info(summary)


# ----------------------------
# GUI Application
# ----------------------------
class AudioRouterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Aurora Audio Router (A↔B) v{VERSION}")
        self.geometry("860x860")

        self.rt        = RtState()
        self.event_log = EventLog()

        self.devices        = []
        self.display_to_id  = {}
        self.selected_A_var = tk.StringVar()
        self.selected_B_var = tk.StringVar()
        self.is_running     = False

        self.inA_gain_var  = tk.DoubleVar(value=0.25)
        self.outA_gain_var = tk.DoubleVar(value=0.25)
        self.inB_gain_var  = tk.DoubleVar(value=0.25)
        self.outB_gain_var = tk.DoubleVar(value=0.25)

        self.mute_inA_var  = tk.BooleanVar(value=False)
        self.mute_outA_var = tk.BooleanVar(value=False)
        self.mute_inB_var  = tk.BooleanVar(value=False)
        self.mute_outB_var = tk.BooleanVar(value=False)
        self.mute_all_var  = tk.BooleanVar(value=False)
        self._updating_mutes = False

        self.sample_rate_var = tk.IntVar(value=DEFAULT_SAMPLERATE)
        self.blocksize_var   = tk.IntVar(value=DEFAULT_BLOCKSIZE)
        self.status_var      = tk.StringVar(value="Idle")

        self.inA_pct_var  = tk.StringVar(value="25%")
        self.outA_pct_var = tk.StringVar(value="25%")
        self.inB_pct_var  = tk.StringVar(value="25%")
        self.outB_pct_var = tk.StringVar(value="25%")

        self.route_A_to_B = None
        self.route_B_to_A = None

        self._build_ui()
        self._wire_mute_traces()
        self._wire_rt_traces()
        self._wire_percent_traces()
        self._update_percent_labels()
        self._push_rt_state()

        self.event_log.add_callback(self._on_log_entry)
        self.event_log.info(f"Aurora Audio Router v{VERSION} started")

        self._refresh_devices()
        self.after(POLL_INTERVAL_MS, self._poll_devices)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Device selection
        dev_frame = ttk.LabelFrame(self, text="Interfaces")
        dev_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(dev_frame, text="WMT").grid(row=0, column=0, sticky='w', **pad)
        self.combo_A = ttk.Combobox(dev_frame, textvariable=self.selected_A_var,
                                    state="readonly", width=80)
        self.combo_A.grid(row=0, column=1, sticky='ew', **pad)
        ttk.Label(dev_frame, text="LAi").grid(row=1, column=0, sticky='w', **pad)
        self.combo_B = ttk.Combobox(dev_frame, textvariable=self.selected_B_var,
                                    state="readonly", width=80)
        self.combo_B.grid(row=1, column=1, sticky='ew', **pad)

        # Audio settings
        set_frame = ttk.LabelFrame(self, text="Audio Settings")
        set_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(set_frame, text="Sample rate (Hz):").grid(row=0, column=0, sticky='w', **pad)
        ttk.Entry(set_frame, textvariable=self.sample_rate_var, width=12).grid(row=0, column=1, sticky='w', **pad)
        ttk.Label(set_frame, text="Blocksize (frames):").grid(row=0, column=2, sticky='w', **pad)
        ttk.Entry(set_frame, textvariable=self.blocksize_var, width=12).grid(row=0, column=3, sticky='w', **pad)

        # Gains & mutes
        vol_frame = ttk.LabelFrame(self, text="Gains & Mutes")
        vol_frame.pack(fill='x', padx=10, pady=10)
        for col, hdr in enumerate(["Control", "Volume", "Mute", "Level (%)"]):
            ttk.Label(vol_frame, text=hdr).grid(row=0, column=col, sticky='w', **pad)
        ttk.Checkbutton(vol_frame, text="Mute All (inputs & outputs)",
                        variable=self.mute_all_var).grid(row=0, column=4, sticky='w', padx=20, pady=6)

        rows = [
            ("Headset Mic",     self.inA_gain_var,  self.mute_inA_var,  self.inA_pct_var),
            ("Headset Speaker", self.outA_gain_var,  self.mute_outA_var, self.outA_pct_var),
            ("LAi mic",         self.inB_gain_var,   self.mute_inB_var,  self.inB_pct_var),
            ("LAi speaker",     self.outB_gain_var,  self.mute_outB_var, self.outB_pct_var),
        ]
        for i, (label, gain_var, mute_var, pct_var) in enumerate(rows, start=1):
            ttk.Label(vol_frame, text=label).grid(row=i, column=0, sticky='w', **pad)
            ttk.Scale(vol_frame, from_=0.0, to=1.0, variable=gain_var,
                      orient='horizontal', length=300).grid(row=i, column=1, sticky='w', **pad)
            ttk.Checkbutton(vol_frame, text="Mute",
                            variable=mute_var).grid(row=i, column=2, sticky='w', **pad)
            ttk.Label(vol_frame, textvariable=pct_var,
                      width=6).grid(row=i, column=3, sticky='w', **pad)

        # Controls
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill='x', padx=10, pady=10)
        self.start_btn = ttk.Button(ctrl_frame, text="Start Routing", command=self.start_routing)
        self.start_btn.pack(side='left', padx=5)
        self.stop_btn  = ttk.Button(ctrl_frame, text="Stop", command=self.stop_routing, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        self.close_btn = ttk.Button(ctrl_frame, text="Close", command=self.on_close)
        self.close_btn.pack(side='left', padx=5)
        self.status_label = ttk.Label(ctrl_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(side='left', padx=20)

        # Notes
        info_frame = ttk.LabelFrame(self, text="Notes")
        info_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(info_frame, justify='left', text=(
            "• Routes Input A → Output B, and Input B → Output A.\n"
            "• Avoid selecting the same device for A and B.\n"
            "• Device lists update on hotplug.\n"
            "• Default: 16 kHz. Increase blocksize if you see ‘input overflow’."
        )).pack(fill='x', padx=10, pady=8)

        # ---- Event Log ----
        log_frame = ttk.LabelFrame(self, text="Event Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        log_btn_bar = ttk.Frame(log_frame)
        log_btn_bar.pack(fill='x', padx=5, pady=4)
        ttk.Button(log_btn_bar, text="Download Log",
                   command=self._download_log).pack(side='left', padx=4)
        ttk.Button(log_btn_bar, text="Clear Log",
                   command=self._clear_log).pack(side='left', padx=4)
        self._autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_btn_bar, text="Auto-scroll",
                        variable=self._autoscroll_var).pack(side='left', padx=8)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=14, state='disabled',
            font=("Courier", 9), wrap='none',
        )
        self.log_text.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        self.log_text.tag_configure("INFO",  foreground="black")
        self.log_text.tag_configure("WARN",  foreground="darkorange")
        self.log_text.tag_configure("ERROR", foreground="red")

    # ------------------------------------------------------------------ #
    # Event log UI helpers                                                 #
    # ------------------------------------------------------------------ #
    def _on_log_entry(self, entry):
        """Called from any thread; schedules the actual widget update on the main thread."""
        self.after(0, self._append_log_entry, entry)

    def _append_log_entry(self, entry):
        ts, level, msg = entry
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"[{ts}] [{level:5s}] {msg}\n", level)
        # Prune if the widget text gets very large
        line_count = int(self.log_text.index('end-1c').split('.')[0])
        if line_count > 1500:
            self.log_text.delete('1.0', '200.0')
        if self._autoscroll_var.get():
            self.log_text.see('end')
        self.log_text.config(state='disabled')

    def _download_log(self):
        entries = self.event_log.get_all()
        if not entries:
            messagebox.showinfo("Download Log", "The log is empty.")
            return
        default_name = f"aurora_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default_name,
            title="Save Event Log",
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"Aurora Audio Router v{VERSION} — Event Log\n")
                f.write(f"Exported: {datetime.datetime.now().isoformat()}\n")
                f.write("-" * 72 + "\n")
                for ts, level, msg in entries:
                    f.write(f"[{ts}] [{level:5s}] {msg}\n")
            messagebox.showinfo("Download Log", f"Log saved to:\n{path}")
            self.event_log.info(f"Log exported to {path}")
        except Exception as e:
            messagebox.showerror("Download Log", f"Failed to save log:\n{e}")

    def _clear_log(self):
        self.event_log.clear()
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.config(state='disabled')
        self.event_log.info("Log cleared by user")

    # ------------------------------------------------------------------ #
    # Slider percent labels                                               #
    # ------------------------------------------------------------------ #
    def _wire_percent_traces(self):
        for var in (self.inA_gain_var, self.outA_gain_var, self.inB_gain_var, self.outB_gain_var):
            var.trace_add('write', lambda *_: self._update_percent_labels())

    def _update_percent_labels(self):
        def pct(v):
            try:   return f"{int(round(float(v) * 100))}%"
            except: return "0%"
        self.inA_pct_var.set(pct(self.inA_gain_var.get()))
        self.outA_pct_var.set(pct(self.outA_gain_var.get()))
        self.inB_pct_var.set(pct(self.inB_gain_var.get()))
        self.outB_pct_var.set(pct(self.outB_gain_var.get()))

    # ------------------------------------------------------------------ #
    # Mute logic: Mute All <-> individual sync                            #
    # ------------------------------------------------------------------ #
    def _wire_mute_traces(self):
        self.mute_all_var.trace_add('write', self._on_mute_all_changed)
        for v in (self.mute_inA_var, self.mute_outA_var, self.mute_inB_var, self.mute_outB_var):
            v.trace_add('write', self._on_individual_mute_changed)

    def _on_mute_all_changed(self, *_):
        if self._updating_mutes: return
        self._updating_mutes = True
        try:
            if self.mute_all_var.get():
                for v in (self.mute_inA_var, self.mute_outA_var, self.mute_inB_var, self.mute_outB_var):
                    v.set(True)
        finally:
            self._updating_mutes = False
        self._push_rt_state()

    def _on_individual_mute_changed(self, *_):
        if self._updating_mutes: return
        if self.mute_all_var.get():
            all_on = all(v.get() for v in (self.mute_inA_var, self.mute_outA_var,
                                            self.mute_inB_var, self.mute_outB_var))
            if not all_on:
                self._updating_mutes = True
                try:   self.mute_all_var.set(False)
                finally: self._updating_mutes = False
        self._push_rt_state()

    # ------------------------------------------------------------------ #
    # RtState mirror                                                       #
    # ------------------------------------------------------------------ #
    def _wire_rt_traces(self):
        all_vars = (
            self.inA_gain_var, self.outA_gain_var, self.inB_gain_var, self.outB_gain_var,
            self.mute_inA_var, self.mute_outA_var, self.mute_inB_var, self.mute_outB_var,
            self.mute_all_var,
        )
        for v in all_vars:
            v.trace_add('write', lambda *_: self._push_rt_state())

    def _push_rt_state(self):
        self.rt.update(
            inA_gain=self.inA_gain_var.get(),   outA_gain=self.outA_gain_var.get(),
            inB_gain=self.inB_gain_var.get(),   outB_gain=self.outB_gain_var.get(),
            mute_inA=self.mute_inA_var.get(),   mute_outA=self.mute_outA_var.get(),
            mute_inB=self.mute_inB_var.get(),   mute_outB=self.mute_outB_var.get(),
            mute_all=self.mute_all_var.get(),
        )

    # ------------------------------------------------------------------ #
    # Device polling                                                       #
    # ------------------------------------------------------------------ #
    def _refresh_devices(self):
        try:
            devs = list_devices()
        except Exception as e:
            self.status_var.set(f"Error listing devices: {e}")
            self.event_log.error(f"Device enumeration failed: {e}")
            devs = []
        self.devices       = devs
        display_list       = [device_display_str(d) for d in devs]
        self.display_to_id = {device_display_str(d): d['id'] for d in devs}

        curA, curB = self.selected_A_var.get(), self.selected_B_var.get()
        self.combo_A['values'] = display_list
        self.combo_B['values'] = display_list

        self.selected_A_var.set(
            curA if curA in self.display_to_id else
            next((s for s in display_list if self._display_has_inputs(s)),
                 display_list[0] if display_list else "")
        )
        self.selected_B_var.set(
            curB if curB in self.display_to_id else
            next((s for s in display_list if self._display_has_outputs(s)),
                 display_list[0] if display_list else "")
        )

    def _display_has_inputs(self, s: str) -> bool:
        try:   return int(s.split("(in:")[1].split(",")[0].strip()) > 0
        except: return False

    def _display_has_outputs(self, s: str) -> bool:
        try:   return int(s.split("out:")[1].split(")")[0].strip()) > 0
        except: return False

    def _poll_devices(self):
        prev = set(self.combo_A['values'])
        self._refresh_devices()
        now  = set(self.combo_A['values'])
        if prev != now:
            for d in now - prev:
                self.event_log.info(f"Device connected: {d}")
            for d in prev - now:
                self.event_log.warn(f"Device disconnected: {d}")
        if self.is_running:
            valid_ids = {d['id'] for d in self.devices}
            a_id = self._selected_id(self.selected_A_var.get())
            b_id = self._selected_id(self.selected_B_var.get())
            if a_id not in valid_ids or b_id not in valid_ids:
                self.event_log.error("Selected device disappeared — stopping routing")
                self._safe_stop("Selected device disconnected. Routing stopped.")
        self.after(POLL_INTERVAL_MS, self._poll_devices)

    def _selected_id(self, display_str: str):
        return self.display_to_id.get(display_str, None)

    # ------------------------------------------------------------------ #
    # Route gain accessors (called by audio thread via RtState)           #
    # ------------------------------------------------------------------ #
    def _gain_AtB_in(self):  return self.rt.gain_in_A()
    def _gain_AtB_out(self): return self.rt.gain_out_B()
    def _gain_BtA_in(self):  return self.rt.gain_in_B()
    def _gain_BtA_out(self): return self.rt.gain_out_A()

    # ------------------------------------------------------------------ #
    # Start / Stop                                                         #
    # ------------------------------------------------------------------ #
    def start_routing(self):
        if self.is_running:
            return
        a_disp, b_disp = self.selected_A_var.get(), self.selected_B_var.get()
        if not a_disp or not b_disp:
            messagebox.showerror("Error", "Please select Interface A and Interface B.")
            return
        a_id, b_id = self._selected_id(a_disp), self._selected_id(b_disp)
        if a_id is None or b_id is None:
            messagebox.showerror("Error", "Invalid device selection.")
            return
        if a_id == b_id:
            messagebox.showerror("Unsafe Selection",
                "Interface A and B are the same device. Choose two different devices.")
            return
        devA, devB = sd.query_devices(a_id), sd.query_devices(b_id)
        for fail, msg in [
            (devA['max_input_channels']  <= 0, "Interface A has no input channels."),
            (devB['max_output_channels'] <= 0, "Interface B has no output channels."),
            (devB['max_input_channels']  <= 0, "Interface B has no input channels (needed for B→A)."),
            (devA['max_output_channels'] <= 0, "Interface A has no output channels (needed for B→A)."),
        ]:
            if fail:
                messagebox.showerror("Error", msg)
                return

        sr, bs = int(self.sample_rate_var.get()), int(self.blocksize_var.get())
        self.event_log.info(
            f"Starting routing — A: {a_disp} | B: {b_disp} | sr={sr} | bs={bs}"
        )
        try:
            self.route_A_to_B = AudioRoute(
                a_id, b_id, sr, bs,
                self._gain_AtB_in, self._gain_AtB_out,
                name="A→B", event_log=self.event_log,
            )
            self.route_B_to_A = AudioRoute(
                b_id, a_id, sr, bs,
                self._gain_BtA_in, self._gain_BtA_out,
                name="B→A", event_log=self.event_log,
            )
            self.route_A_to_B.start()
            self.route_B_to_A.start()
            self.is_running = True
            self.status_var.set(f"Running @ {sr} Hz, blocksize {bs}")
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
        except Exception as e:
            self.event_log.error(f"Failed to start routing: {e}\n{traceback.format_exc()}")
            self._safe_stop(f"Failed to start routing: {e}")

    def stop_routing(self):
        self._safe_stop("Stopped")

    def _safe_stop(self, status_msg: str):
        self.event_log.info(f"Stopping routing — {status_msg}")
        try:
            if self.route_A_to_B: self.route_A_to_B.stop()
            if self.route_B_to_A: self.route_B_to_A.stop()
        except Exception:
            self.event_log.error(f"Error during stop:\n{traceback.format_exc()}")
        finally:
            self.route_A_to_B = None
            self.route_B_to_A = None
            self.is_running   = False
            self.status_var.set(status_msg)
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')

    # ------------------------------------------------------------------ #
    # Close                                                                #
    # ------------------------------------------------------------------ #
    def on_close(self):
        if self.is_running:
            self._safe_stop("Stopped")
        try:   self.quit()
        except Exception: pass
        self.destroy()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    try:
        app = AudioRouterApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
