#!/usr/bin/env python3
VERSION = "1.1.0"

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import numpy as np
import sounddevice as sd
import sys
import traceback
import time

# ----------------------------
# Configuration defaults
# ----------------------------
DEFAULT_SAMPLERATE = 16000        # 16 kHz as requested
DEFAULT_BLOCKSIZE = 256           # Try 512/1024 if you still see overflows
POLL_INTERVAL_MS = 1000           # Device list refresh interval (ms)
ROUTE_QUEUE_SIZE = 32             # Per-route buffer queue depth
PRINT_DEBUG = True

def log(*args):
    if PRINT_DEBUG:
        print("[AudioRouter]", *args)

# ----------------------------
# Helpers
# ----------------------------
def list_devices():
    """Return a list of devices as dicts with id, name, hostapi, ins, outs."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    result = []
    for idx, d in enumerate(devices):
        hostapi_name = hostapis[d['hostapi']]['name'] if 'hostapi' in d else 'Unknown'
        result.append({
            'id': idx,
            'name': d.get('name', f'Device {idx}'),
            'hostapi': hostapi_name,
            'ins': int(d.get('max_input_channels', 0)),
            'outs': int(d.get('max_output_channels', 0)),
        })
    return result

def device_display_str(dev):
    return f"[{dev['id']}] {dev['name']} — {dev['hostapi']} (in:{dev['ins']}, out:{dev['outs']})"

def adapt_channels(data: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Convert frame array shape (frames, in_channels) to (frames, out_channels).
    Strategy:
      - Same channels: passthrough
      - 1 -> 2: duplicate
      - 2 -> 1: average
      - more->less: take first N
      - less->more: duplicate up to N
    """
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
    reps = int(np.ceil(out_channels / in_ch))
    tiled = np.tile(data, (1, reps))
    return tiled[:, :out_channels]

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

# ----------------------------
# Thread-safe runtime state (NO Tk access here)
# ----------------------------
class RtState:
    """Plain thread-safe mirror of all gains/mutes."""
    def __init__(self):
        self._lock = threading.Lock()
        # Raw controls (linear 0..1)
        self.inA_gain = 1.0
        self.outA_gain = 1.0
        self.inB_gain = 1.0
        self.outB_gain = 1.0
        # Mutes
        self.mute_inA = False
        self.mute_outA = False
        self.mute_inB = False
        self.mute_outB = False
        self.mute_all = False

    # --- Updaters called from GUI thread ---
    def update(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    # --- Readers used by audio thread (fast, no Tk) ---
    def gain_in_A(self) -> float:
        with self._lock:
            if self.mute_all or self.mute_inA:
                return 0.0
            return clamp01(self.inA_gain)

    def gain_out_A(self) -> float:
        with self._lock:
            if self.mute_all or self.mute_outA:
                return 0.0
            return clamp01(self.outA_gain)

    def gain_in_B(self) -> float:
        with self._lock:
            if self.mute_all or self.mute_inB:
                return 0.0
            return clamp01(self.inB_gain)

    def gain_out_B(self) -> float:
        with self._lock:
            if self.mute_all or self.mute_outB:
                return 0.0
            return clamp01(self.outB_gain)

# ----------------------------
# Audio Routing Engine
# ----------------------------
class AudioRoute:
    """
    Route audio from one device's input to another device's output using two separate streams
    connected by a queue. GUI state is read via RtState (thread-safe), not via Tkinter.
    """
    def __init__(self, input_dev_id: int, output_dev_id: int,
                 samplerate: int, blocksize: int,
                 get_in_gain, get_out_gain, name="A_to_B"):
        self.input_dev_id = int(input_dev_id)
        self.output_dev_id = int(output_dev_id)
        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self.get_in_gain = get_in_gain  # callables (no Tk inside)
        self.get_out_gain = get_out_gain
        self.name = name

        in_dev = sd.query_devices(self.input_dev_id)
        out_dev = sd.query_devices(self.output_dev_id)
        self.in_channels = max(1, min(2, int(in_dev['max_input_channels'])))
        self.out_channels = max(1, min(2, int(out_dev['max_output_channels'])))

        self.q = queue.Queue(maxsize=ROUTE_QUEUE_SIZE)
        self._stop_flag = threading.Event()
        self._streams_started = False

        # Let the backend choose stable latency; specifying 'low' can increase xruns on some hosts
        self.input_stream = sd.InputStream(
            device=self.input_dev_id,
            channels=self.in_channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._input_cb
        )
        self.output_stream = sd.OutputStream(
            device=self.output_dev_id,
            channels=self.out_channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._output_cb
        )
        self._last_status_print = 0.0

    def _maybe_log_status(self, where, status):
        # Throttle status prints to once per 2 seconds to avoid spamming
        now = time.time()
        if now - self._last_status_print > 2.0:
            log(f"[{self.name}] {where} status: {status}")
            self._last_status_print = now

    def _input_cb(self, indata, frames, time_info, status):
        if status:
            self._maybe_log_status("Input", status)
        if self._stop_flag.is_set():
            return

        # Apply input gain (including Mute All + input mute)
        g_in = clamp01(self.get_in_gain())
        data = (indata.copy() * g_in).astype(np.float32)

        # Non-blocking enqueue; drop oldest on overflow to keep latency bounded
        try:
            self.q.put_nowait(data)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(data)
            except queue.Full:
                pass

    def _output_cb(self, outdata, frames, time_info, status):
        if status:
            self._maybe_log_status("Output", status)
        if self._stop_flag.is_set():
            outdata.fill(0)
            return

        try:
            data = self.q.get_nowait()
        except queue.Empty:
            outdata.fill(0)
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
                pad = np.zeros((frames - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.concatenate([data, pad], axis=0)

        # Apply output gain (including Mute All + output mute)
        g_out = clamp01(self.get_out_gain())
        data = (data * g_out).astype(np.float32)
        outdata[:] = data

    def start(self):
        if self._streams_started:
            return
        self._stop_flag.clear()
        self.input_stream.start()
        self.output_stream.start()
        self._streams_started = True
        log(f"[{self.name}] Started: in_dev={self.input_dev_id}, out_dev={self.output_dev_id}, "
            f"sr={self.samplerate}, bs={self.blocksize}, in_ch={self.in_channels}, out_ch={self.out_channels}")

    def stop(self):
        """Abort immediately to avoid waiting for buffers; then close."""
        if not self._streams_started:
            return
        self._stop_flag.set()
        try:
            # Abort is immediate; safer on some backends than stop()
            self.input_stream.abort()
            self.output_stream.abort()
        except Exception:
            # Fallback if abort not supported
            try:
                self.input_stream.stop()
                self.output_stream.stop()
            except Exception:
                pass
        finally:
            try:
                self.input_stream.close()
            except Exception:
                pass
            try:
                self.output_stream.close()
            except Exception:
                pass
            self._streams_started = False

            # Drain queue safely
            try:
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass

        log(f"[{self.name}] Stopped")

# ----------------------------
# GUI Application
# ----------------------------
class AudioRouterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Raspberry Pi Audio Cross-Router (A↔B) v{VERSION}")
        self.geometry("840x620")

        # Thread-safe runtime state
        self.rt = RtState()

        # State
        self.devices = []
        self.display_to_id = {}
        self.selected_A_var = tk.StringVar()
        self.selected_B_var = tk.StringVar()
        self.is_running = False

        # Volume / mute (GUI variables) - DEFAULT 25%
        self.inA_gain_var = tk.DoubleVar(value=0.25)
        self.outA_gain_var = tk.DoubleVar(value=0.25)
        self.inB_gain_var = tk.DoubleVar(value=0.25)
        self.outB_gain_var = tk.DoubleVar(value=0.25)

        self.mute_inA_var = tk.BooleanVar(value=False)
        self.mute_outA_var = tk.BooleanVar(value=False)
        self.mute_inB_var = tk.BooleanVar(value=False)
        self.mute_outB_var = tk.BooleanVar(value=False)

        self.mute_all_var = tk.BooleanVar(value=False)
        self._updating_mutes = False  # prevents recursion when syncing checkboxes

        self.sample_rate_var = tk.IntVar(value=DEFAULT_SAMPLERATE)
        self.blocksize_var = tk.IntVar(value=DEFAULT_BLOCKSIZE)

        self.status_var = tk.StringVar(value="Idle")

        # Percent label StringVars
        self.inA_pct_var = tk.StringVar(value="25%")
        self.outA_pct_var = tk.StringVar(value="25%")
        self.inB_pct_var = tk.StringVar(value="25%")
        self.outB_pct_var = tk.StringVar(value="25%")

        # Audio routes
        self.route_A_to_B = None
        self.route_B_to_A = None

        # Build UI and wire state
        self._build_ui()
        self._wire_mute_traces()
        self._wire_rt_traces()
        self._wire_percent_traces()
        self._update_percent_labels()
        self._push_rt_state()

        # Devices
        self._refresh_devices()
        self.after(POLL_INTERVAL_MS, self._poll_devices)

        # Handle window "X" close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # --------- UI ----------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Device selection
        dev_frame = ttk.LabelFrame(self, text="Interfaces")
        dev_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(dev_frame, text="WMT").grid(row=0, column=0, sticky='w', **pad)
        self.combo_A = ttk.Combobox(dev_frame, textvariable=self.selected_A_var, state="readonly", width=80)
        self.combo_A.grid(row=0, column=1, sticky='ew', **pad)
        ttk.Label(dev_frame, text="LAi").grid(row=1, column=0, sticky='w', **pad)
        self.combo_B = ttk.Combobox(dev_frame, textvariable=self.selected_B_var, state="readonly", width=80)
        self.combo_B.grid(row=1, column=1, sticky='ew', **pad)

        # Settings
        set_frame = ttk.LabelFrame(self, text="Audio Settings")
        set_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(set_frame, text="Sample rate (Hz):").grid(row=0, column=0, sticky='w', **pad)
        ttk.Entry(set_frame, textvariable=self.sample_rate_var, width=12).grid(row=0, column=1, sticky='w', **pad)
        ttk.Label(set_frame, text="Blocksize (frames):").grid(row=0, column=2, sticky='w', **pad)
        ttk.Entry(set_frame, textvariable=self.blocksize_var, width=12).grid(row=0, column=3, sticky='w', **pad)

        # Gains & Mutes
        vol_frame = ttk.LabelFrame(self, text="Gains & Mutes")
        vol_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(vol_frame, text="Control").grid(row=0, column=0, sticky='w', **pad)
        ttk.Label(vol_frame, text="Volume").grid(row=0, column=1, sticky='w', **pad)
        ttk.Label(vol_frame, text="Mute").grid(row=0, column=2, sticky='w', **pad)
        # Level (%) header
        ttk.Label(vol_frame, text="Level (%)").grid(row=0, column=3, sticky='w', **pad)
        # Move Mute All to column 4 to keep column 3 for percentages
        ttk.Checkbutton(vol_frame, text="Mute All (inputs & outputs)",
                        variable=self.mute_all_var).grid(row=0, column=4, sticky='w', padx=20, pady=6)

        # Input A row
        ttk.Label(vol_frame, text="Headset Mic").grid(row=1, column=0, sticky='w', **pad)
        ttk.Scale(vol_frame, from_=0.0, to=1.0, variable=self.inA_gain_var,
                  orient='horizontal', length=300).grid(row=1, column=1, sticky='w', **pad)
        ttk.Checkbutton(vol_frame, text="Mute", variable=self.mute_inA_var).grid(row=1, column=2, sticky='w', **pad)
        ttk.Label(vol_frame, textvariable=self.inA_pct_var, width=6).grid(row=1, column=3, sticky='w', **pad)

        # Output A row
        ttk.Label(vol_frame, text="Headset Speaker").grid(row=2, column=0, sticky='w', **pad)
        ttk.Scale(vol_frame, from_=0.0, to=1.0, variable=self.outA_gain_var,
                  orient='horizontal', length=300).grid(row=2, column=1, sticky='w', **pad)
        ttk.Checkbutton(vol_frame, text="Mute", variable=self.mute_outA_var).grid(row=2, column=2, sticky='w', **pad)
        ttk.Label(vol_frame, textvariable=self.outA_pct_var, width=6).grid(row=2, column=3, sticky='w', **pad)

        # Input B row
        ttk.Label(vol_frame, text="LAi mic").grid(row=3, column=0, sticky='w', **pad)
        ttk.Scale(vol_frame, from_=0.0, to=1.0, variable=self.inB_gain_var,
                  orient='horizontal', length=300).grid(row=3, column=1, sticky='w', **pad)
        ttk.Checkbutton(vol_frame, text="Mute", variable=self.mute_inB_var).grid(row=3, column=2, sticky='w', **pad)
        ttk.Label(vol_frame, textvariable=self.inB_pct_var, width=6).grid(row=3, column=3, sticky='w', **pad)

        # Output B row
        ttk.Label(vol_frame, text="LAi speaker").grid(row=4, column=0, sticky='w', **pad)
        ttk.Scale(vol_frame, from_=0.0, to=1.0, variable=self.outB_gain_var,
                  orient='horizontal', length=300).grid(row=4, column=1, sticky='w', **pad)
        ttk.Checkbutton(vol_frame, text="Mute", variable=self.mute_outB_var).grid(row=4, column=2, sticky='w', **pad)
        ttk.Label(vol_frame, textvariable=self.outB_pct_var, width=6).grid(row=4, column=3, sticky='w', **pad)

        # Controls
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill='x', padx=10, pady=10)
        self.start_btn = ttk.Button(ctrl_frame, text="Start Routing", command=self.start_routing)
        self.start_btn.pack(side='left', padx=5)
        self.stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self.stop_routing, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        self.close_btn = ttk.Button(ctrl_frame, text="Close", command=self.on_close)
        self.close_btn.pack(side='left', padx=5)

        self.status_label = ttk.Label(ctrl_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(side='left', padx=20)

        info_frame = ttk.LabelFrame(self, text="Notes")
        info_frame.pack(fill='x', padx=10, pady=10)
        notes = (
            "• Routes Input A → Output B, and Input B → Output A.\n"
            "• Avoid selecting the same device for A and B.\n"
            "• Device lists update on hotplug.\n"
            "• Default: 16 kHz. Increase blocksize if you see 'input overflow'."
        )
        ttk.Label(info_frame, text=notes, justify='left').pack(fill='x', padx=10, pady=8)

    # --------- Percent label updates ----------
    def _wire_percent_traces(self):
        # Update percentage labels whenever any slider changes
        for var in (self.inA_gain_var, self.outA_gain_var, self.inB_gain_var, self.outB_gain_var):
            var.trace_add('write', lambda *_: self._update_percent_labels())

    def _update_percent_labels(self):
        def pct(v):  # 0..1 -> "NN%"
            try:
                return f"{int(round(float(v) * 100))}%"
            except Exception:
                return "0%"
        self.inA_pct_var.set(pct(self.inA_gain_var.get()))
        self.outA_pct_var.set(pct(self.outA_gain_var.get()))
        self.inB_pct_var.set(pct(self.inB_gain_var.get()))
        self.outB_pct_var.set(pct(self.outB_gain_var.get()))

    # --------- Mute logic: Mute All ↔ individual sync ----------
    def _wire_mute_traces(self):
        # Mute All controls all individuals
        self.mute_all_var.trace_add('write', self._on_mute_all_changed)
        # Any individual change can uncheck Mute All
        for var in (self.mute_inA_var, self.mute_outA_var, self.mute_inB_var, self.mute_outB_var):
            var.trace_add('write', self._on_individual_mute_changed)

    def _on_mute_all_changed(self, *args):
        if self._updating_mutes:
            return
        self._updating_mutes = True
        try:
            if self.mute_all_var.get():
                # Force all four individual mutes ON
                self.mute_inA_var.set(True)
                self.mute_outA_var.set(True)
                self.mute_inB_var.set(True)
                self.mute_outB_var.set(True)
            # If Mute All becomes False, we leave individuals as-is (per your spec)
        finally:
            self._updating_mutes = False
        # Push updated state to RtState
        self._push_rt_state()

    def _on_individual_mute_changed(self, *args):
        if self._updating_mutes:
            return
        # If any individual is OFF while Mute All is ON, uncheck Mute All
        if self.mute_all_var.get():
            if not (self.mute_inA_var.get() and self.mute_outA_var.get() and
                    self.mute_inB_var.get() and self.mute_outB_var.get()):
                self._updating_mutes = True
                try:
                    self.mute_all_var.set(False)
                finally:
                    self._updating_mutes = False
        # Push updated state to RtState
        self._push_rt_state()

    # --------- Keep RtState mirrored with GUI ----------
    def _wire_rt_traces(self):
        # Gains
        for var in (self.inA_gain_var, self.outA_gain_var, self.inB_gain_var, self.outB_gain_var):
            var.trace_add('write', lambda *_: self._push_rt_state())
        # Mutes (including Mute All)
        for var in (self.mute_inA_var, self.mute_outA_var, self.mute_inB_var, self.mute_outB_var, self.mute_all_var):
            var.trace_add('write', lambda *_: self._push_rt_state())

    def _push_rt_state(self):
        self.rt.update(
            inA_gain=self.inA_gain_var.get(),
            outA_gain=self.outA_gain_var.get(),
            inB_gain=self.inB_gain_var.get(),
            outB_gain=self.outB_gain_var.get(),
            mute_inA=self.mute_inA_var.get(),
            mute_outA=self.mute_outA_var.get(),
            mute_inB=self.mute_inB_var.get(),
            mute_outB=self.mute_outB_var.get(),
            mute_all=self.mute_all_var.get(),
        )

    # --------- Device polling ----------
    def _refresh_devices(self):
        try:
            devs = list_devices()
        except Exception as e:
            self.status_var.set(f"Error listing devices: {e}")
            devs = []
        self.devices = devs
        display_list = [device_display_str(d) for d in devs]
        self.display_to_id = {device_display_str(d): d['id'] for d in devs}

        curA = self.selected_A_var.get()
        curB = self.selected_B_var.get()
        self.combo_A['values'] = display_list
        self.combo_B['values'] = display_list

        if curA in self.display_to_id:
            self.selected_A_var.set(curA)
        else:
            first_in = next((s for s in display_list if self._display_has_inputs(s)), display_list[0] if display_list else "")
            self.selected_A_var.set(first_in)

        if curB in self.display_to_id:
            self.selected_B_var.set(curB)
        else:
            first_out = next((s for s in display_list if self._display_has_outputs(s)), display_list[0] if display_list else "")
            self.selected_B_var.set(first_out)

    def _display_has_inputs(self, display_str: str) -> bool:
        try:
            parts = display_str.split("(in:")[1]
            ins = int(parts.split(",")[0].strip())
            return ins > 0
        except Exception:
            return False

    def _display_has_outputs(self, display_str: str) -> bool:
        try:
            parts = display_str.split("out:")[1]
            outs = int(parts.split(")")[0].strip())
            return outs > 0
        except Exception:
            return False

    def _poll_devices(self):
        prev = set(self.combo_A['values'])
        self._refresh_devices()
        now = set(self.combo_A['values'])
        if prev != now:
            log("Device list updated")
        # If running and a selected device disappears, stop routing
        if self.is_running:
            valid_ids = {d['id'] for d in self.devices}
            a_id = self._selected_id(self.selected_A_var.get())
            b_id = self._selected_id(self.selected_B_var.get())
            if a_id not in valid_ids or b_id not in valid_ids:
                self._safe_stop("Selected device disconnected. Routing stopped.")
        self.after(POLL_INTERVAL_MS, self._poll_devices)

    def _selected_id(self, display_str: str):
        return self.display_to_id.get(display_str, None)

    # --------- Route gain accessors (audio thread uses RtState) ----------
    def _route_gain_A_to_B_in(self) -> float:
        return self.rt.gain_in_A()
    def _route_gain_A_to_B_out(self) -> float:
        return self.rt.gain_out_B()
    def _route_gain_B_to_A_in(self) -> float:
        return self.rt.gain_in_B()
    def _route_gain_B_to_A_out(self) -> float:
        return self.rt.gain_out_A()

    # --------- Start/Stop ----------
    def start_routing(self):
        if self.is_running:
            return

        a_disp = self.selected_A_var.get()
        b_disp = self.selected_B_var.get()
        if not a_disp or not b_disp:
            messagebox.showerror("Error", "Please select Interface A and Interface B.")
            return

        a_id = self._selected_id(a_disp)
        b_id = self._selected_id(b_disp)
        if a_id is None or b_id is None:
            messagebox.showerror("Error", "Invalid device selection.")
            return

        if a_id == b_id:
            messagebox.showerror("Unsafe Selection", "Interface A and B are the same. "
                                 "This can cause dangerous feedback. Choose two different devices.")
            return

        devA = sd.query_devices(a_id)
        devB = sd.query_devices(b_id)
        if devA['max_input_channels'] <= 0:
            messagebox.showerror("Error", "Selected Interface A has no input channels.")
            return
        if devB['max_output_channels'] <= 0:
            messagebox.showerror("Error", "Selected Interface B has no output channels.")
            return
        if devB['max_input_channels'] <= 0:
            messagebox.showerror("Error", "Selected Interface B has no input channels (required for B→A).")
            return
        if devA['max_output_channels'] <= 0:
            messagebox.showerror("Error", "Selected Interface A has no output channels (required for B→A).")
            return

        sr = int(self.sample_rate_var.get())
        bs = int(self.blocksize_var.get())

        try:
            self.route_A_to_B = AudioRoute(
                input_dev_id=a_id,
                output_dev_id=b_id,
                samplerate=sr,
                blocksize=bs,
                get_in_gain=self._route_gain_A_to_B_in,
                get_out_gain=self._route_gain_A_to_B_out,
                name="A_to_B"
            )
            self.route_B_to_A = AudioRoute(
                input_dev_id=b_id,
                output_dev_id=a_id,
                samplerate=sr,
                blocksize=bs,
                get_in_gain=self._route_gain_B_to_A_in,
                get_out_gain=self._route_gain_B_to_A_out,
                name="B_to_A"
            )

            self.route_A_to_B.start()
            self.route_B_to_A.start()

            self.is_running = True
            self.status_var.set(f"Running @ {sr} Hz, blocksize {bs}")
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
        except Exception as e:
            traceback.print_exc()
            self._safe_stop(f"Failed to start routing: {e}")

    def stop_routing(self):
        self._safe_stop("Stopped")

    def _safe_stop(self, status_msg: str):
        try:
            if self.route_A_to_B:
                self.route_A_to_B.stop()
            if self.route_B_to_A:
                self.route_B_to_A.stop()
        except Exception:
            traceback.print_exc()
        finally:
            self.route_A_to_B = None
            self.route_B_to_A = None
            self.is_running = False
            self.status_var.set(status_msg)
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')

    # --------- Close handling ----------
    def on_close(self):
        if self.is_running:
            self._safe_stop("Stopped")
        try:
            self.quit()
        except Exception:
            pass
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
