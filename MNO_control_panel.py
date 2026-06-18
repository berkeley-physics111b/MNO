import csv
import os
import sys
import time
import threading
import queue
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# WaveForms driver
# ---------------------------------------------------------------------------
try:
    from waveforms_ads import WaveFormsADS, DWFError
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False

class AcquisitionThread(threading.Thread):
    _CHUNK_MS = 40

    def __init__(self, device, sample_rate: float, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self._dev  = device
        self._rate = sample_rate
        self._n    = max(64, int(round(sample_rate * self._CHUNK_MS / 1000)))
        self._q    = out_queue
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from waveforms_ads import acqmodeSingle, trigsrcNone, DwfStateDone
        except ImportError:
            return

        dev = self._dev
        dev.analog_in_reset()
        dev.analog_in_set_sample_rate(self._rate)
        dev.analog_in_set_buffer_size(self._n)
        dev.analog_in_set_acquisition_mode(acqmodeSingle)
        dev.analog_in_channel_enable(0)
        dev.analog_in_channel_enable(1)
        dev.analog_in_set_trigger_source(trigsrcNone)

        while not self._stop.is_set():
            try:
                dev.analog_in_configure(reconfigure=True, start=True)
                deadline = time.time() + 5.0
                while not self._stop.is_set():
                    state = dev.analog_in_status(read_data=True)
                    if state == DwfStateDone or time.time() > deadline:
                        break
                    time.sleep(0.001)
                ch1 = dev.analog_in_get_data(0, self._n)
                ch2 = dev.analog_in_get_data(1, self._n)
                self._q.put((ch1, ch2))
            except Exception as exc:
                print(f"[AcqThread] {exc}", file=sys.stderr)
                time.sleep(0.1)