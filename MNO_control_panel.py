"""
mno_control_panel.py
================
Four-tab Tkinter GUI for MNO via waveforms_ads.py

Tabs:
  1. Photodiode Viewer    - live dual-channel scope (Ch0 / Ch1 vs time)
  2. Output Sweep Test    - triangle-wave output control with preview
  3. Diode Current Sweep  - Sweeps laser current and views photodiode response.
  4. B Field Sweep        - Sweeps B field and views photodiode response.

Acquisition architecture for tabs 3 & 4
-----------------------------------------
Each "cycle" of the output waveform is handled as follows:
  1. Configure ADC: sample rate, buffer (= n_pts), trigger = AnalogOut1/2,
     acqmodeSingle so it arms and waits.
  2. Configure DAC: funcTriangle, freq = scan_freq, amplitude = v_pp/2,
     offset = v_min + v_pp/2, phase = 270°, repeat = 1.
  3. Start ADC (armed, waiting for trigger).
  4. Start DAC (fires once → triggers ADC).
  5. Poll ADC until DwfStateDone, read n_pts samples.
  6. Stop / reset both instruments, accumulate into running average.
  7. Repeat for next cycle.

The output voltage axis for the plots is reconstructed from the known
triangle waveform shape (make_triangle_wave), so no separate voltage
channel is needed.

For Tab 3 the ADC sample rate is set by the user (fs_hz); the n_pts
buffer is captured across the whole cycle.  The x-axis is the
reconstructed output voltage (which sweeps min→peak→min), so points
are plotted in acquisition order but labelled by their reconstructed
output voltage.

For Tab 4 the ADC sample rate is set to n_pts * scan_freq so that
exactly one sample is taken per DAC step (tight coupling).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import csv
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------------------
# Hardware import / stub
# ---------------------------------------------------------------------------

try:
    from waveforms_ads import WaveFormsADS, DwfStateDone, acqmodeSingle, \
        trigsrcAnalogOut1, trigsrcAnalogOut2, trigsrcNone
    HW_AVAILABLE = True
except Exception:
    HW_AVAILABLE = False

def get_device():
    if HW_AVAILABLE:
        return WaveFormsADS()
    raise ModuleNotFoundError("waveforms_ads package not present")


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
BG        = "#1e1e2e"
FG        = "#cdd6f4"
ACCENT    = "#89b4fa"
BTN_RUN   = "#a6e3a1"
BTN_STOP  = "#f38ba8"
BTN_PAUSE = "#fab387"
BTN_CLEAR = "#89dceb"
BTN_SAVE  = "#cba6f7"
ENTRY_BG  = "#313244"
FRAME_BG  = "#181825"
PLOT_BG   = "#11111b"
CH0_COLOR = "#89b4fa"
CH1_COLOR = "#a6e3a1"
SUM_COLOR = "#fab387"
ROT_COLOR = "#f5c2e7"

FONT_LABEL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 10)
FONT_TITLE = ("Segoe UI", 10, "bold")

MAX_DISPLAY = 2000   # max points rendered on any plot (display undersampling)

ACQUISITION_TIMEOUT = 30.0   # seconds before giving up on a single cycle


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def make_triangle_wave(v_min, v_pp, n_points):
    """
    Reconstruct the voltage axis for n_points samples of a hardware
    funcTriangle wave (phase=270°): starts at v_min, rises to v_min+v_pp,
    falls back to v_min.
    """
    peak = v_min + v_pp
    half = n_points // 2
    up   = np.linspace(v_min, peak, half, endpoint=False)
    down = np.linspace(peak, v_min, n_points - half)
    return np.concatenate([up, down])


def downsample(arr, max_pts=MAX_DISPLAY):
    arr = np.asarray(arr)
    if len(arr) > max_pts:
        idx = np.round(np.linspace(0, len(arr) - 1, max_pts)).astype(int)
        return arr[idx]
    return arr


def rotation(c0, c1):
    """
    Element-wise (c0-c1)/(2*(c0+c1)).  Returns nan where denominator~0.
    Operates on numpy arrays; ignores div-by-zero silently.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 2.0 * (c0 + c1)
        r = np.where(np.abs(denom) > 1e-300, (c0 - c1) / denom, np.nan)
    return r


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------

def styled_button(parent, text, command, color=ACCENT, width=10):
    return tk.Button(
        parent, text=text, command=command,
        bg=color, fg="#1e1e2e", activebackground=color,
        font=FONT_TITLE, relief="flat", bd=0,
        padx=6, pady=3, width=width, cursor="hand2",
    )


def labeled_entry(parent, label, default, width=10):
    """Return (frame, StringVar).  Frame uses FRAME_BG."""
    frm = tk.Frame(parent, bg=FRAME_BG)
    tk.Label(frm, text=label, bg=FRAME_BG, fg=FG,
             font=FONT_LABEL).pack(side="left", padx=(0, 4))
    var = tk.StringVar(value=str(default))
    tk.Entry(frm, textvariable=var, width=width,
             bg=ENTRY_BG, fg=FG, insertbackground=FG,
             relief="flat", font=FONT_MONO).pack(side="left")
    return frm, var


def make_figure(n_rows, figsize=None):
    if figsize is None:
        figsize = (7, 2.4 * n_rows)
    fig  = Figure(facecolor=PLOT_BG, figsize=figsize)
    axes = []
    for i in range(n_rows):
        ax = fig.add_subplot(n_rows, 1, i + 1)
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#45475a")
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        axes.append(ax)
    fig.tight_layout(pad=1.4)
    return fig, axes


def embed_canvas(fig, parent):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def configure_adc_triggered(dev, trig_src, fs_hz, n_pts, rng, off):
    """
    Arm the ADC for a single triggered acquisition.
    trig_src: trigsrcAnalogOut1 or trigsrcAnalogOut2.
    """
    dev.analog_in_reset()
    dev.analog_in_set_sample_rate(fs_hz)
    dev.analog_in_set_buffer_size(n_pts)
    dev.analog_in_set_acquisition_mode(acqmodeSingle)
    for ch in (0, 1):
        dev.analog_in_channel_enable(ch)
        dev.analog_in_set_range(ch, rng)
        dev.analog_in_set_offset(ch, off)
    dev.analog_in_set_trigger_source(trig_src)
    dev.analog_in_set_trigger_position(0.5*n_pts/fs_hz) # put trigger at start of buffer
    dev.analog_in_configure(reconfigure=True, start=True)   # arms, waits


def wait_adc_done(dev, timeout=ACQUISITION_TIMEOUT):
    """Poll until ADC returns DwfStateDone.  Raises TimeoutError."""
    deadline = time.time() + timeout
    while True:
        state = dev.analog_in_status(read_data=True)
        if state == DwfStateDone:
            return
        if time.time() > deadline:
            raise TimeoutError("ADC acquisition timed out.")
        time.sleep(0.002)


def configure_dac_triangle(dev, ch, v_min, v_pp, scan_freq_hz,
                            n_repeats=1):
    """
    Configure the DAC to output exactly n_repeats cycles of a triangle
    wave (min→peak→min, phase=270°) then stop.
    amplitude = v_pp/2, offset = v_min + v_pp/2.
    """
    amplitude = v_pp / 2.0
    offset    = v_min + v_pp / 2.0
    dev.analog_out_reset(ch)
    dev.analog_out_set_triangle(
        ch,
        freq_hz    = scan_freq_hz,
        amplitude_v= amplitude,
        offset_v   = offset,
        phase_deg  = 270.0,
    )
    # Run for exactly n_repeats cycles then stop automatically
    # Should it be set repeats is n_repeats...? n_repeats=0 -> inf run
    dev.analog_out_set_run_time(ch, n_repeats / scan_freq_hz)
    dev.analog_out_set_repeat(ch, 1)   # repeat the run block once


# ---------------------------------------------------------------------------
# Tab 1 – Oscilloscope
# ---------------------------------------------------------------------------

class ScopeTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._running  = False
        self._thread   = None
        self._lock     = threading.Lock()
        self._t_data   = np.empty(0)
        self._ch0_data = np.empty(0)
        self._ch1_data = np.empty(0)
        self._t_start  = None
        self._build()

    def _build(self):
        # Controls bar
        ctrl = tk.Frame(self, bg=FRAME_BG, pady=6)
        ctrl.pack(fill="x", padx=8, pady=(8, 0))

        tk.Label(ctrl, text="Photodiode Viewer", bg=FRAME_BG, fg=ACCENT,
                 font=FONT_TITLE).grid(row=0, column=0, columnspan=8,
                                       sticky="w", padx=6)

        self._frm_fs,  self._var_fs  = labeled_entry(ctrl, "Sample Rate (Hz):", "100", 8)
        self._frm_rng, self._var_rng = labeled_entry(ctrl, "Voltage Range (V):", "10",  6)
        self._frm_off, self._var_off = labeled_entry(ctrl, "Voltage Offset (V):", "0",  6)
        self._frm_fs .grid(row=1, column=0, padx=8, pady=4, sticky="w")
        self._frm_rng.grid(row=1, column=1, padx=8, pady=4, sticky="w")
        self._frm_off.grid(row=1, column=2, padx=8, pady=4, sticky="w")

        bf = tk.Frame(ctrl, bg=FRAME_BG)
        bf.grid(row=1, column=3, padx=16, sticky="w")
        styled_button(bf, "▶  Run",   self._run,   BTN_RUN  ).pack(side="left", padx=3)
        styled_button(bf, "■  Stop",  self._stop,  BTN_STOP ).pack(side="left", padx=3)
        styled_button(bf, "✕  Clear", self._clear, BTN_CLEAR).pack(side="left", padx=3)

        # Plots
        pf = tk.Frame(self, bg=BG)
        pf.pack(fill="both", expand=True, padx=8, pady=8)
        self._fig, self._axes = make_figure(2, figsize=(8, 5))
        self._axes[0].set_ylabel("Ch0 (V)", color=CH0_COLOR)
        self._axes[1].set_ylabel("Ch1 (V)", color=CH1_COLOR)
        self._axes[1].set_xlabel("Time (s)")
        self._line0, = self._axes[0].plot([], [], color=CH0_COLOR, lw=1)
        self._line1, = self._axes[1].plot([], [], color=CH1_COLOR, lw=1)
        self._fig.tight_layout(pad=1.4)
        self._canvas = embed_canvas(self._fig, pf)

    # ── Controls ──────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return
        try:
            fs  = float(self._var_fs.get())
            rng = float(self._var_rng.get())
            off = float(self._var_off.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid numeric parameter.")
            return
        self._running = True
        self._t_start = time.time()
        self._thread  = threading.Thread(
            target=self._loop, args=(fs, rng, off), daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False

    def _clear(self):
        self._running = False
        with self._lock:
            self._t_data   = np.empty(0)
            self._ch0_data = np.empty(0)
            self._ch1_data = np.empty(0)
        self._line0.set_data([], [])
        self._line1.set_data([], [])
        try:
            rng = float(self._var_rng.get())
        except ValueError:
            rng = 10.0
        for ax in self._axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(-rng / 2, rng / 2)
        self._canvas.draw_idle()

    # ── Acquisition thread ────────────────────────────────────────────────

    def _loop(self, fs, rng, off):
        chunk = max(1, int(fs / 10))   # ~100 ms worth of samples per call
        dev   = get_device()
        try:
            dev.analog_in_reset()
            dev.analog_in_set_sample_rate(fs)
            dev.analog_in_set_buffer_size(chunk)
            dev.analog_in_set_acquisition_mode(acqmodeSingle)
            for ch in (0, 1):
                dev.analog_in_channel_enable(ch)
                dev.analog_in_set_range(ch, rng)
                dev.analog_in_set_offset(ch, off)
            dev.analog_in_set_trigger_source(trigsrcNone)
            dev.analog_in_configure(reconfigure=True, start=True)

            while self._running:
                t0 = time.time()
                wait_adc_done(dev, timeout=max(chunk / fs * 5, 2.0))
                c0 = dev.analog_in_get_data(0, chunk)
                c1 = dev.analog_in_get_data(1, chunk)
                t_chunk = np.linspace(
                    t0 - self._t_start - chunk / fs,
                    t0 - self._t_start,
                    chunk, endpoint=False)
                with self._lock:
                    self._t_data   = np.concatenate([self._t_data,   t_chunk])
                    self._ch0_data = np.concatenate([self._ch0_data, c0])
                    self._ch1_data = np.concatenate([self._ch1_data, c1])
                self._redraw()
                # Rearm for next chunk
                dev.analog_in_configure(reconfigure=False, start=True)
                elapsed = time.time() - t0
                time.sleep(max(0.0, chunk / fs - elapsed))
        except Exception as e:
            messagebox.showerror("Scope Error", str(e))
        finally:
            dev.close()
            self._running = False

    def _redraw(self):
        try:
            rng = float(self._var_rng.get())
        except ValueError:
            rng = 10.0
        with self._lock:
            t  = self._t_data.copy()
            c0 = self._ch0_data.copy()
            c1 = self._ch1_data.copy()
        if len(t) == 0:
            for ax in self._axes:
                ax.set_xlim(0, 1)
                ax.set_ylim(-rng / 2, rng / 2)
            self._canvas.draw_idle()
            return
        t_d  = downsample(t)
        c0_d = downsample(c0)
        c1_d = downsample(c1)
        self._line0.set_data(t_d, c0_d)
        self._line1.set_data(t_d, c1_d)
        for ax in self._axes:
            ax.set_xlim(t_d[0], max(t_d[-1], t_d[0] + 0.1))
            ax.set_ylim(-rng / 2, rng / 2)
        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Tab 2 – AWG Output (manual triangle sweep, preview + monitor)
# ---------------------------------------------------------------------------

class AWGTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._running    = False
        self._paused     = False
        self._thread     = None
        self._cur_v      = tk.StringVar(value="—")
        self._freq_str   = tk.StringVar(value="—")
        self._btn_start  = None   # set in _build; updated by pause/resume
        self._btn_pause  = None
        self._build()

    def _build(self):
        # ── Left: controls ────────────────────────────────────────────────
        left = tk.Frame(self, bg=FRAME_BG, width=300)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="Output Sweep Test", bg=FRAME_BG, fg=ACCENT,
                 font=FONT_TITLE).pack(anchor="w", padx=8, pady=(8, 4))

        # Channel selector
        rf = tk.Frame(left, bg=FRAME_BG)
        rf.pack(anchor="w", padx=8, pady=2)
        tk.Label(rf, text="Output Channel:", bg=FRAME_BG, fg=FG,
                 font=FONT_LABEL).pack(side="left")
        self._var_ch = tk.StringVar(value="0")
        ttk.Combobox(rf, textvariable=self._var_ch, values=["0", "1"],
                     width=4, state="readonly").pack(side="left", padx=4)

        # Parameter entries
        self._vars = {}
        for label, key, default in [
            ("Min Voltage (V):",       "v_min",   "0.0"),
            ("Peak-to-Peak (V):",      "v_pp",    "1.0"),
            ("Points / Scan:",         "n_pts",   "100"),
            ("Scan Freq (Hz):",        "freq",    "1.0"),
            ("Num Scans (0=∞):",       "n_scans", "0"),
        ]:
            frm, var = labeled_entry(left, label, default, 8)
            frm.pack(anchor="w", padx=8, pady=3)
            self._vars[key] = var

        # Derived frequency display
        fdf = tk.Frame(left, bg=FRAME_BG)
        fdf.pack(anchor="w", padx=8, pady=2)
        tk.Label(fdf, text="Scan Period:", bg=FRAME_BG, fg=FG,
                 font=FONT_LABEL).pack(side="left")
        tk.Label(fdf, textvariable=self._freq_str, bg=FRAME_BG,
                 fg=ACCENT, font=FONT_MONO).pack(side="left", padx=4)
        self._vars["freq"].trace_add("write", lambda *_: self._update_freq())

        # Buttons
        bf = tk.Frame(left, bg=FRAME_BG)
        bf.pack(anchor="w", padx=8, pady=8)
        self._btn_start = styled_button(bf, "▶  Start", self._start, BTN_RUN,   9)
        self._btn_start.pack(side="left", padx=2)
        self._btn_pause = styled_button(bf, "⏸  Pause", self._pause, BTN_PAUSE, 9)
        self._btn_pause.pack(side="left", padx=2)
        styled_button(bf, "■  Stop",  self._stop,  BTN_STOP,  9).pack(side="left", padx=2)

        # Voltage monitor
        mf = tk.Frame(left, bg=FRAME_BG)
        mf.pack(anchor="w", padx=8, pady=4)
        tk.Label(mf, text="Output Voltage:", bg=FRAME_BG, fg=FG,
                 font=FONT_LABEL).pack(side="left")
        tk.Label(mf, textvariable=self._cur_v, bg=FRAME_BG,
                 fg=BTN_RUN, font=("Consolas", 14, "bold")).pack(side="left", padx=6)

        for key in ("v_min", "v_pp", "n_pts"):
            self._vars[key].trace_add("write", lambda *_: self._update_preview())

        # ── Right: waveform preview ───────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        self._fig_p, self._axes_p = make_figure(1, figsize=(6, 3.5))
        self._axes_p[0].set_xlabel("Sample Index")
        self._axes_p[0].set_ylabel("Voltage (V)")
        self._axes_p[0].set_title("Waveform Preview", color=FG)
        self._line_p, = self._axes_p[0].plot([], [], color=ACCENT, lw=1.5)
        self._canvas_p = embed_canvas(self._fig_p, right)
        self._update_preview()
        self._update_freq()

    def _update_preview(self, *_):
        try:
            v_min = float(self._vars["v_min"].get())
            v_pp  = float(self._vars["v_pp"].get())
            n_pts = int(self._vars["n_pts"].get())
        except (ValueError, tk.TclError):
            return
        wave = make_triangle_wave(v_min, v_pp, n_pts)
        self._line_p.set_data(np.arange(len(wave)), wave)
        ax = self._axes_p[0]
        ax.set_xlim(0, max(len(wave) - 1, 1))
        pad = v_pp * 0.1 + 0.01
        ax.set_ylim(v_min - pad, v_min + v_pp + pad)
        self._canvas_p.draw_idle()

    def _update_freq(self, *_):
        try:
            freq = float(self._vars["freq"].get())
            self._freq_str.set(f"{1.0/freq:.4g} s  ({freq:.4g} Hz)")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self._freq_str.set("—")

    # ── Run logic ─────────────────────────────────────────────────────────

    def _start(self):
        # If paused, resume instead of starting a new thread
        if self._paused and self._running:
            self._paused = False
            self._btn_pause.config(text="⏸  Pause", bg=BTN_PAUSE)
            self._btn_start.config(text="▶  Start")
            return
        if self._running:
            return
        try:
            ch      = int(self._var_ch.get())
            v_min   = float(self._vars["v_min"].get())
            v_pp    = float(self._vars["v_pp"].get())
            n_pts   = int(self._vars["n_pts"].get())
            freq    = float(self._vars["freq"].get())
            n_scans = int(self._vars["n_scans"].get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid parameter.")
            return
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(
            target=self._loop,
            args=(ch, v_min, v_pp, n_pts, freq, n_scans),
            daemon=True)
        self._thread.start()

    def _pause(self):
        if not self._running:
            return
        self._paused = not self._paused
        if self._paused:
            self._btn_pause.config(text="▶  Resume", bg=BTN_RUN)
            self._btn_start.config(text="▶  Resume")
        else:
            self._btn_pause.config(text="⏸  Pause", bg=BTN_PAUSE)
            self._btn_start.config(text="▶  Start")

    def _stop(self):
        self._running = False
        self._paused  = False
        self._cur_v.set("—")
        if self._btn_pause:
            self._btn_pause.config(text="⏸  Pause", bg=BTN_PAUSE)
        if self._btn_start:
            self._btn_start.config(text="▶  Start")

    def _loop(self, ch, v_min, v_pp, n_pts, freq, n_scans):
        """
        Use the hardware triangle function directly.
        Step through the reconstructed waveform array for the voltage
        monitor only (one update per ~period/n_pts sleep).
        """
        wave     = make_triangle_wave(v_min, v_pp, n_pts)
        step_dt  = 1.0 / (freq * n_pts)   # time per point
        scan_dur = 1.0 / freq
        dev = get_device()
        try:
            configure_dac_triangle(dev, ch, v_min, v_pp,
                                   scan_freq_hz=freq,
                                   n_repeats=1)
            scan = 0
            while self._running:
                # Kind of janky solution, clean up/fix
                if n_scans > 0 and scan >= n_scans:
                    break
                while self._paused and self._running:
                    time.sleep(0.05)
                if not self._running:
                    break
                # Start one cycle
                dev.analog_out_start(ch)
                t_start = time.time()
                # Update voltage monitor in lock-step with reconstructed wave
                # Read out voltage directly instead? No, takes another channel
                for v in wave:
                    if not self._running or self._paused:
                        break
                    self._cur_v.set(f"{v:+.4f} V")
                    time.sleep(step_dt)
                # Wait for hardware cycle to finish if we exited loop early
                # Replace with hard stop/output zero...? Maybe idle set?
                remaining = scan_dur - (time.time() - t_start)
                if remaining > 0:
                    time.sleep(remaining)
                dev.analog_out_stop(ch)
                scan += 1
        except Exception as e:
            messagebox.showerror("AWG Error", str(e))
        finally:
            try:
                dev.analog_out_stop(ch)
                dev.close()
            except Exception:
                pass
            self._running = False
            self._cur_v.set("—")


# ---------------------------------------------------------------------------
# Shared mini AWG panel (used in Tabs 3 & 4)
# ---------------------------------------------------------------------------

class MiniAWGPanel(tk.LabelFrame):
    def __init__(self, parent, ch_label="Ch0"):
        super().__init__(parent, text=f"Output ({ch_label})",
                         bg=FRAME_BG, fg=ACCENT, font=FONT_TITLE)
        self._cur_v = tk.StringVar(value="—")
        self._build()

    def _build(self):
        self._fig, self._axes = make_figure(1, figsize=(3.2, 1.7))
        self._axes[0].set_xlabel("Index", fontsize=7)
        self._axes[0].set_ylabel("V",     fontsize=7)
        self._line, = self._axes[0].plot([], [], color=ACCENT, lw=1)
        self._canvas = embed_canvas(self._fig, self)

        mf = tk.Frame(self, bg=FRAME_BG)
        mf.pack(fill="x", padx=4, pady=2)
        tk.Label(mf, text="Output V:", bg=FRAME_BG, fg=FG,
                 font=FONT_LABEL).pack(side="left")
        tk.Label(mf, textvariable=self._cur_v, bg=FRAME_BG,
                 fg=BTN_RUN, font=("Consolas", 11, "bold")).pack(side="left", padx=4)

    def update_wave(self, wave):
        self._line.set_data(np.arange(len(wave)), wave)
        ax = self._axes[0]
        ax.set_xlim(0, max(len(wave) - 1, 1))
        if len(wave):
            span = wave.max() - wave.min()
            pad  = span * 0.1 + 0.01
            ax.set_ylim(wave.min() - pad, wave.max() + pad)
        self._canvas.draw_idle()

    def set_voltage(self, v):
        self._cur_v.set("—" if v is None else f"{v:+.4f} V")


# ---------------------------------------------------------------------------
# Tab 3 – Diode Current Sweep
#
# Architecture:
#   • ADC triggered by AnalogOut1 (trigsrcAnalogOut1)
#   • DAC Ch0: funcTriangle, one cycle per acquisition
#   • ADC captures n_pts samples across the full cycle at user fs_hz
#   • x-axis reconstructed from make_triangle_wave
#   • Per-cycle averaging of raw ADC arrays
# ---------------------------------------------------------------------------

class DiodeCurrentSweepTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._running    = False
        self._thread     = None
        self._lock       = threading.Lock()
        # Accumulated averages (numpy arrays, or None before first scan)
        self._avg_c0     = None
        self._avg_c1     = None
        self._vout_axis  = None   # reconstructed output voltage axis
        self._scan_count = 0
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build(self):
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=6)
        paned.pack(fill="both", expand=True)

        # Left panel
        left = tk.Frame(paned, bg=FRAME_BG, width=340)
        paned.add(left, minsize=290)

        tk.Label(left, text="Diode Current Sweep", bg=FRAME_BG,
                 fg=ACCENT, font=FONT_TITLE).pack(anchor="w", padx=8, pady=(8, 4))

        # Output params
        alf = tk.LabelFrame(left, text="Output (Ch0)", bg=FRAME_BG,
                            fg=ACCENT, font=FONT_TITLE)
        alf.pack(fill="x", padx=8, pady=4)
        self._vars_out = {}
        for label, key, default in [
            ("Min Voltage (V):",  "v_min",   "0.0"),
            ("Peak-to-Peak (V):", "v_pp",    "1.0"),
            ("Points / Scan:",    "n_pts",   "100"),
            ("Scan Freq (Hz):",   "freq",    "1.0"),
            ("Num Scans (0=∞):",  "n_scans", "0"),
        ]:
            frm, var = labeled_entry(alf, label, default, 8)
            frm.pack(anchor="w", padx=6, pady=2)
            self._vars_out[key] = var
            var.trace_add("write", lambda *_: self._update_mini())

        self._mini = MiniAWGPanel(left, "Ch0")
        self._mini.pack(fill="x", padx=8, pady=4)

        # Input params
        ilf = tk.LabelFrame(left, text="Input (Ch0 & Ch1)", bg=FRAME_BG,
                            fg=ACCENT, font=FONT_TITLE)
        ilf.pack(fill="x", padx=8, pady=4)
        self._vars_in = {}
        for label, key, default in [
            ("Sample Rate (Hz):",  "fs",  "1000"),
            ("Voltage Range (V):", "rng", "10"),
            ("Voltage Offset (V):","off", "0"),
        ]:
            frm, var = labeled_entry(ilf, label, default, 8)
            frm.pack(anchor="w", padx=6, pady=2)
            self._vars_in[key] = var

        # Scan counter label
        self._scan_label = tk.StringVar(value="Scans completed: 0")
        tk.Label(left, textvariable=self._scan_label, bg=FRAME_BG,
                 fg=FG, font=FONT_LABEL).pack(anchor="w", padx=8, pady=2)

        # Buttons
        bf = tk.Frame(left, bg=FRAME_BG)
        bf.pack(anchor="w", padx=8, pady=8)
        styled_button(bf, "▶  Start", self._start, BTN_RUN,  9).pack(side="left", padx=2)
        styled_button(bf, "■  Stop",  self._stop,  BTN_STOP, 9).pack(side="left", padx=2)
        styled_button(bf, "✕  Clear", self._clear, BTN_CLEAR,9).pack(side="left", padx=2)
        styled_button(bf, "Save Data", self._save,  BTN_SAVE, 9).pack(side="left", padx=2)

        # Right panel – plots
        right = tk.Frame(paned, bg=BG)
        paned.add(right, minsize=420)

        self._fig, self._axes = make_figure(2, figsize=(6, 5.5))
        self._axes[0].set_ylabel("Ch0 (V)", color=CH0_COLOR)
        self._axes[1].set_ylabel("Ch1 (V)", color=CH1_COLOR)
        for ax in self._axes:
            ax.set_xlabel("Output Voltage (V)")
        self._line0, = self._axes[0].plot([], [], color=CH0_COLOR, lw=1.2)
        self._line1, = self._axes[1].plot([], [], color=CH1_COLOR, lw=1.2)
        self._canvas = embed_canvas(self._fig, right)

        self._update_mini()

    def _update_mini(self, *_):
        try:
            wave = make_triangle_wave(
                float(self._vars_out["v_min"].get()),
                float(self._vars_out["v_pp"].get()),
                int(self._vars_out["n_pts"].get()),
            )
            self._mini.update_wave(wave)
        except (ValueError, tk.TclError):
            pass

    # ── Acquisition ───────────────────────────────────────────────────────

    def _parse_params(self):
        v_min   = float(self._vars_out["v_min"].get())
        v_pp    = float(self._vars_out["v_pp"].get())
        n_pts   = int(self._vars_out["n_pts"].get())
        freq    = float(self._vars_out["freq"].get())
        n_scans = int(self._vars_out["n_scans"].get())
        fs      = float(self._vars_in["fs"].get())
        rng     = float(self._vars_in["rng"].get())
        off     = float(self._vars_in["off"].get())
        return v_min, v_pp, n_pts, freq, n_scans, fs, rng, off

    def _start(self):
        if self._running:
            return
        try:
            params = self._parse_params()
        except ValueError:
            messagebox.showerror("Input Error", "Invalid parameter.")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, args=params, daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False

    def _clear(self):
        self._running = False
        with self._lock:
            self._avg_c0    = None
            self._avg_c1    = None
            self._vout_axis = None
            self._scan_count = 0
        self._scan_label.set("Scans completed: 0")
        self._mini.set_voltage(None)
        self._line0.set_data([], [])
        self._line1.set_data([], [])
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()
        self._canvas.draw_idle()

    def _loop(self, v_min, v_pp, n_pts, freq, n_scans, fs, rng, off):
        """
        Per-cycle: arm ADC (triggered by AnalogOut1), start DAC,
        wait for ADC done, read both channels, accumulate average.
        """
        vout_axis = make_triangle_wave(v_min, v_pp, n_pts)
        dev = get_device()
        scan = 0
        try:
            while self._running:
                if n_scans > 0 and scan >= n_scans:
                    break

                # 1. Arm ADC – waits for AnalogOut1 trigger
                configure_adc_triggered(
                    dev, trigsrcAnalogOut1, fs, n_pts, rng, off)

                # 2. Configure DAC for one cycle
                configure_dac_triangle(
                    dev, ch=0, v_min=v_min, v_pp=v_pp,
                    scan_freq_hz=freq, n_repeats=1)

                # 3. Fire DAC (triggers ADC)
                dev.analog_out_start(0)

                # 4. Update voltage monitor while hardware runs
                step_dt  = 1.0 / (freq * n_pts)
                t_fire   = time.time()
                for v in vout_axis:
                    if not self._running:
                        break
                    self._mini.set_voltage(v)
                    time.sleep(step_dt)

                # 5. Wait for ADC done
                wait_adc_done(dev)

                # 6. Read data
                c0 = dev.analog_in_get_data(0, n_pts)
                c1 = dev.analog_in_get_data(1, n_pts)

                dev.analog_out_stop(0)

                scan += 1
                # 7. Accumulate running average
                with self._lock:
                    self._scan_count += 1
                    n = self._scan_count
                    if n == 1:
                        self._avg_c0    = c0.copy()
                        self._avg_c1    = c1.copy()
                        self._vout_axis = vout_axis.copy()
                    else:
                        self._avg_c0 = (self._avg_c0 * (n - 1) + c0) / n
                        self._avg_c1 = (self._avg_c1 * (n - 1) + c1) / n

                self._scan_label.set(f"Scans completed: {self._scan_count}")
                self._redraw()

        except Exception as e:
            messagebox.showerror("Diode Current Sweep Error", str(e))
        finally:
            try:
                dev.analog_out_stop(0)
                dev.close()
            except Exception:
                pass
            self._mini.set_voltage(None)
            self._running = False

    def _redraw(self):
        with self._lock:
            vout = self._vout_axis
            c0   = self._avg_c0
            c1   = self._avg_c1
        if vout is None:
            self._canvas.draw_idle()
            return
        self._line0.set_data(downsample(vout), downsample(c0))
        self._line1.set_data(downsample(vout), downsample(c1))
        for ax in self._axes:
            ax.relim(); ax.autoscale_view()
        self._canvas.draw_idle()

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Sweep Data")
        if not path:
            return
        with self._lock:
            vout = self._vout_axis
            c0   = self._avg_c0
            c1   = self._avg_c1
            sc   = self._scan_count
        if vout is None:
            messagebox.showwarning("No Data", "No data to save.")
            return
        try:
            hdr = {k: self._vars_out[k].get() for k in self._vars_out}
            hdr.update({k: self._vars_in[k].get() for k in self._vars_in})
        except Exception:
            hdr = {}
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# Diode Current Scan"])
            for k, v in hdr.items():
                w.writerow([f"# {k}={v}"])
            w.writerow([f"# scans_completed={sc}"])
            w.writerow(["output_voltage_V", "ch0_V", "ch1_V"])
            for row in zip(vout, c0, c1):
                w.writerow([f"{x:.7g}" for x in row])
        messagebox.showinfo("Saved", f"Data saved to:\n{path}")


# ---------------------------------------------------------------------------
# Tab 4 – B Field Sweep
#
# Architecture:
#   • ADC triggered by AnalogOut2 (trigsrcAnalogOut2)
#   • DAC Ch1: funcTriangle, one cycle per acquisition
#   • ADC sample rate = n_pts * scan_freq  → exactly 1 sample per DAC step
#   • Four plots: Ch0, Ch1, Sum, Rotation vs output voltage
#   • Per-cycle averaging of Ch0 and Ch1; Sum/Rotation recomputed from averages
# ---------------------------------------------------------------------------

class BFieldSweepTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._running    = False
        self._thread     = None
        self._lock       = threading.Lock()
        self._avg_c0     = None
        self._avg_c1     = None
        self._vout_axis  = None
        self._scan_count = 0
        self._build()

    def _build(self):
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=6)
        paned.pack(fill="both", expand=True)

        # Left panel
        left = tk.Frame(paned, bg=FRAME_BG, width=340)
        paned.add(left, minsize=290)

        tk.Label(left, text="B Field Sweep", bg=FRAME_BG,
                 fg=ACCENT, font=FONT_TITLE).pack(anchor="w", padx=8, pady=(8, 4))

        alf = tk.LabelFrame(left, text="Output (Ch1)", bg=FRAME_BG,
                            fg=ACCENT, font=FONT_TITLE)
        alf.pack(fill="x", padx=8, pady=4)
        self._vars_out = {}
        for label, key, default in [
            ("Min Voltage (V):",  "v_min",   "0.0"),
            ("Peak-to-Peak (V):", "v_pp",    "1.0"),
            ("Points / Scan:",    "n_pts",   "100"),
            ("Scan Freq (Hz):",   "freq",    "1.0"),
            ("Num Scans (0=∞):",  "n_scans", "0"),
        ]:
            frm, var = labeled_entry(alf, label, default, 8)
            frm.pack(anchor="w", padx=6, pady=2)
            self._vars_out[key] = var
            var.trace_add("write", lambda *_: self._update_mini())

        self._mini = MiniAWGPanel(left, "Ch1")
        self._mini.pack(fill="x", padx=8, pady=4)

        # Note: ADC sample rate is derived (n_pts * scan_freq), not user-set
        ilf = tk.LabelFrame(left, text="Input (Ch0 & Ch1)", bg=FRAME_BG,
                            fg=ACCENT, font=FONT_TITLE)
        ilf.pack(fill="x", padx=8, pady=4)
        self._vars_in = {}
        for label, key, default in [
            ("Voltage Range (V):", "rng", "10"),
            ("Voltage Offset (V):","off", "0"),
        ]:
            frm, var = labeled_entry(ilf, label, default, 8)
            frm.pack(anchor="w", padx=6, pady=2)
            self._vars_in[key] = var

        # Derived ADC rate display
        self._fs_label = tk.StringVar(value="ADC rate: — Hz")
        tk.Label(ilf, textvariable=self._fs_label, bg=FRAME_BG,
                 fg=ACCENT, font=FONT_LABEL).pack(anchor="w", padx=6, pady=2)
        for key in ("n_pts", "freq"):
            self._vars_out[key].trace_add("write",
                lambda *_: self._update_fs_label())

        self._scan_label = tk.StringVar(value="Scans completed: 0")
        tk.Label(left, textvariable=self._scan_label, bg=FRAME_BG,
                 fg=FG, font=FONT_LABEL).pack(anchor="w", padx=8, pady=2)

        bf = tk.Frame(left, bg=FRAME_BG)
        bf.pack(anchor="w", padx=8, pady=8)
        styled_button(bf, "▶  Start", self._start, BTN_RUN,  9).pack(side="left", padx=2)
        styled_button(bf, "■  Stop",  self._stop,  BTN_STOP, 9).pack(side="left", padx=2)
        styled_button(bf, "✕  Clear", self._clear, BTN_CLEAR,9).pack(side="left", padx=2)
        styled_button(bf, "Save Data", self._save,  BTN_SAVE, 9).pack(side="left", padx=2)

        # Right panel – 4 stacked plots
        right = tk.Frame(paned, bg=BG)
        paned.add(right, minsize=420)

        self._fig, self._axes = make_figure(4, figsize=(6, 9.5))
        plot_cfg = [
            ("Ch0 (V)",  CH0_COLOR),
            ("Ch1 (V)",  CH1_COLOR),
            ("Ch0 + Ch1 (V)",  SUM_COLOR),
            ("Rotation Angle", ROT_COLOR),
        ]
        self._lines = []
        for i, (ylabel, color) in enumerate(plot_cfg):
            self._axes[i].set_ylabel(ylabel, color=color)
            line, = self._axes[i].plot([], [], color=color, lw=1.2)
            self._lines.append(line)
        self._axes[-1].set_xlabel("Output Voltage (V)")
        self._fig.tight_layout(pad=1.4)
        self._canvas = embed_canvas(self._fig, right)

        self._update_mini()
        self._update_fs_label()

    def _update_mini(self, *_):
        try:
            wave = make_triangle_wave(
                float(self._vars_out["v_min"].get()),
                float(self._vars_out["v_pp"].get()),
                int(self._vars_out["n_pts"].get()),
            )
            self._mini.update_wave(wave)
        except (ValueError, tk.TclError):
            pass

    def _update_fs_label(self, *_):
        try:
            n_pts = int(self._vars_out["n_pts"].get())
            freq  = float(self._vars_out["freq"].get())
            fs    = n_pts * freq
            self._fs_label.set(f"ADC rate: {fs:.4g} Hz  (1 sample / DAC step)")
        except (ValueError, tk.TclError, ZeroDivisionError):
            self._fs_label.set("ADC rate: —")

    # ── Acquisition ───────────────────────────────────────────────────────

    def _parse_params(self):
        v_min   = float(self._vars_out["v_min"].get())
        v_pp    = float(self._vars_out["v_pp"].get())
        n_pts   = int(self._vars_out["n_pts"].get())
        freq    = float(self._vars_out["freq"].get())
        n_scans = int(self._vars_out["n_scans"].get())
        rng     = float(self._vars_in["rng"].get())
        off     = float(self._vars_in["off"].get())
        return v_min, v_pp, n_pts, freq, n_scans, rng, off

    def _start(self):
        if self._running:
            return
        try:
            params = self._parse_params()
        except ValueError:
            messagebox.showerror("Input Error", "Invalid parameter.")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, args=params, daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False

    def _clear(self):
        self._running = False
        with self._lock:
            self._avg_c0    = None
            self._avg_c1    = None
            self._vout_axis = None
            self._scan_count = 0
        self._scan_label.set("Scans completed: 0")
        self._mini.set_voltage(None)
        for line in self._lines:
            line.set_data([], [])
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()
        self._canvas.draw_idle()

    def _loop(self, v_min, v_pp, n_pts, freq, n_scans, rng, off):
        """
        ADC sample rate = n_pts * freq  → one sample per DAC step.
        Triggered by AnalogOut2 (Ch1).
        """
        fs        = n_pts * freq          # tight coupling: 1 ADC sample per DAC point
        vout_axis = make_triangle_wave(v_min, v_pp, n_pts)
        dev  = get_device()
        scan = 0
        try:
            while self._running:
                if n_scans > 0 and scan >= n_scans:
                    break

                # 1. Arm ADC
                configure_adc_triggered(
                    dev, trigsrcAnalogOut2, fs, n_pts, rng, off)

                # 2. Configure DAC Ch1 for one cycle
                configure_dac_triangle(
                    dev, ch=1, v_min=v_min, v_pp=v_pp,
                    scan_freq_hz=freq, n_repeats=1)

                # 3. Fire DAC
                dev.analog_out_start(1)

                # 4. Update voltage monitor
                step_dt = 1.0 / (freq * n_pts)
                for v in vout_axis:
                    if not self._running:
                        break
                    self._mini.set_voltage(v)
                    time.sleep(step_dt)

                # 5. Wait for ADC done
                wait_adc_done(dev)

                # 6. Read both channels
                c0 = dev.analog_in_get_data(0, n_pts)
                c1 = dev.analog_in_get_data(1, n_pts)

                dev.analog_out_stop(1)

                scan += 1
                # 7. Running average of raw channels
                with self._lock:
                    self._scan_count += 1
                    n = self._scan_count
                    if n == 1:
                        self._avg_c0    = c0.copy()
                        self._avg_c1    = c1.copy()
                        self._vout_axis = vout_axis.copy()
                    else:
                        self._avg_c0 = (self._avg_c0 * (n - 1) + c0) / n
                        self._avg_c1 = (self._avg_c1 * (n - 1) + c1) / n

                self._scan_label.set(f"Scans completed: {self._scan_count}")
                self._redraw()

        except Exception as e:
            messagebox.showerror("B Field Scan Error", str(e))
        finally:
            try:
                dev.analog_out_stop(1)
                dev.close()
            except Exception:
                pass
            self._mini.set_voltage(None)
            self._running = False

    def _redraw(self):
        with self._lock:
            vout = self._vout_axis
            c0   = self._avg_c0
            c1   = self._avg_c1
        if vout is None:
            self._canvas.draw_idle()
            return
        # Derive sum and rotation from averaged channels
        s = c0 + c1
        r = rotation(c0, c1)

        arrs = [c0, c1, s, r]
        for i, (arr, line) in enumerate(zip(arrs, self._lines)):
            if i == 3:   # rotation: skip non-finite
                finite = np.isfinite(arr)
                if finite.any():
                    line.set_data(downsample(vout[finite]),
                                  downsample(arr[finite]))
                else:
                    line.set_data([], [])
            else:
                line.set_data(downsample(vout), downsample(arr))
            self._axes[i].relim()
            self._axes[i].autoscale_view()
        self._canvas.draw_idle()

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save B Field Scan Data")
        if not path:
            return
        with self._lock:
            vout = self._vout_axis
            c0   = self._avg_c0
            c1   = self._avg_c1
            sc   = self._scan_count
        if vout is None:
            messagebox.showwarning("No Data", "No data to save.")
            return
        s = c0 + c1
        r = rotation(c0, c1)
        try:
            hdr = {k: self._vars_out[k].get() for k in self._vars_out}
            hdr.update({k: self._vars_in[k].get() for k in self._vars_in})
            n_pts = int(self._vars_out["n_pts"].get())
            freq  = float(self._vars_out["freq"].get())
            hdr["adc_rate_hz"] = n_pts * freq
        except Exception:
            hdr = {}
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# B Field Scan"])
            for k, v in hdr.items():
                w.writerow([f"# {k}={v}"])
            w.writerow([f"# scans_completed={sc}"])
            w.writerow(["output_voltage_V", "ch0_V", "ch1_V", "sum_V", "rotation angle"])
            for row in zip(vout, c0, c1, s, r):
                w.writerow([
                    f"{x:.7g}" if np.isfinite(x) else "NaN" for x in row
                ])
        messagebox.showinfo("Saved", f"Data saved to:\n{path}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WaveForms ADS Control")
        self.geometry("1150x780")
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()

    def _build(self):
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TNotebook",     background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=FRAME_BG, foreground=FG,
                        font=FONT_TITLE, padding=[12, 5])
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#1e1e2e")])
        style.configure("TPanedwindow", background=BG)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        self._tabs = {
            "scope":  ScopeTab(nb),
            "sweep":  AWGTab(nb),
            "diode":  DiodeCurrentSweepTab(nb),
            "bfield": BFieldSweepTab(nb),
        }
        nb.add(self._tabs["scope"], text="  Photodiode Viewer  ")
        nb.add(self._tabs["sweep"],   text="  Sweep Test  ")
        nb.add(self._tabs["diode"],    text="  Diode Current Scan  ")
        nb.add(self._tabs["bfield"],   text="  B Field Scan  ")

    def _on_close(self):
        for tab in self._tabs.values():
            if hasattr(tab, "_running"):
                tab._running = False
        self.after(250, self.destroy)


if __name__ == "__main__":
    app = App()
    app.mainloop()