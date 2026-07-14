"""Govee VJ — capture this laptop's audio, visualize the spectrum, and sync
a Govee light to the beat.

Captures system audio via WASAPI loopback (whatever is playing on the default
output), runs an FFT into log-spaced frequency bands, detects beats from the
bass energy, and streams color to a Govee device over the LAN API. A tkinter
window shows a live spectrum analyzer, the detected beat, and the color being
sent to the light.

Modes:
    python vj.py                       # GUI visualizer + light  (default IP 10.0.0.98)
    python vj.py --ip 10.0.0.98        # explicit device IP
    python vj.py --no-light            # visualizer only, no device
    python vj.py --devices             # list WASAPI loopback devices and exit
    python vj.py --selftest 5          # headless: capture 5s, print bands/beats

Tuning knobs:
    --intensity 0.4                    # overall brightness scale (0.05-1.0)
    --color-template ocean             # palette: spectrum/rainbow/sunset/fire/
                                       #   ocean/neon/mono
    --beat-threshold 0.05              # min bass level to register a beat

Light mapping: dominant frequency -> hue (bass=red, treble=blue), overall
energy -> base glow, and each detected beat pops the brightness (synced to the
beat). Whole-device color only (LAN API has no per-segment control).
"""

import argparse
import colorsys
import json
import math
import socket
import threading
import time
from collections import deque

import numpy as np
import pyaudiowpatch as pyaudio

# ---- Audio / analysis config ------------------------------------------------
CHUNK = 1024            # frames per read (~23ms @ 44.1k) - low latency
N_BANDS = 48            # spectrum bars
FMIN = 30.0             # lowest band edge (Hz)
FMAX = 16000.0          # highest band edge (Hz)
BASS_LO, BASS_HI = 40.0, 120.0   # beat-detection band (kick drum fundamental)

# Beat detection
BEAT_HISTORY = 43       # ~1s of frames at 44.1k/1024
BEAT_SENS = 1.45        # instant bass must exceed local mean * this
BEAT_TAU = 0.12         # light beat-pop decay time constant (s)
# Expected tempo range (defaults: broad; override with --bpm-min/--bpm-max).
DEFAULT_BPM_MIN = 70.0
DEFAULT_BPM_MAX = 180.0

# Light
DEFAULT_IP = "10.0.0.98"
CONTROL_PORT = 4003
LIGHT_FPS = 25

# Color mapping: named templates map spectral centroid -> hue.
#   hue = (hue_start + centroid * hue_span) % 1.0
# centroid is 0 (bass-heavy) .. 1 (treble-heavy). A zero span pins one color.
COLOR_TEMPLATES = {
    "spectrum": (0.00, 0.66),  # bass=red .. treble=blue (classic default)
    "rainbow":  (0.00, 1.00),  # full hue wheel across the spectrum
    "sunset":   (0.95, 0.18),  # deep red -> orange -> gold
    "fire":     (0.00, 0.12),  # red -> orange, hot and tight
    "ocean":    (0.42, 0.28),  # teal -> blue -> violet
    "neon":     (0.78, 0.50),  # magenta -> pink -> cyan
    "mono":     (0.66, 0.00),  # fixed blue; only brightness reacts
}
DEFAULT_TEMPLATE = "spectrum"
DEFAULT_INTENSITY = 1.0      # overall brightness scale 0.05-1.0
DEFAULT_BEAT_THRESHOLD = 0.02  # min bass level (fraction of peak) to count a beat


# ---- Shared state between threads -------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.bands = np.zeros(N_BANDS, dtype=np.float32)  # display 0..1
        self.energy = 0.0            # overall 0..1
        self.centroid = 0.0          # spectral centroid 0..1
        self.last_beat = -10.0       # monotonic time of last beat
        self.beat_count = 0
        self.bpm = 0.0
        self.color = (0, 0, 0)       # last color sent to the light
        self.running = True
        # Live-tunable settings (edited from the UI, read by the threads).
        self.intensity = DEFAULT_INTENSITY
        self.template = DEFAULT_TEMPLATE
        self.beat_threshold = DEFAULT_BEAT_THRESHOLD

    def snapshot(self):
        with self.lock:
            return (self.bands.copy(), self.energy, self.centroid,
                    self.last_beat, self.beat_count, self.bpm, self.color)


# ---- Audio capture + analysis ----------------------------------------------
def find_loopback(p, name_hint=None):
    """Return the loopback device info matching the default output (or hint)."""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    target = name_hint or default_out["name"]
    best = None
    for lb in p.get_loopback_device_info_generator():
        if target and target in lb["name"]:
            return lb
        best = best or lb
    return best


class AudioAnalyzer(threading.Thread):
    def __init__(self, shared, name_hint=None,
                 bpm_min=DEFAULT_BPM_MIN, bpm_max=DEFAULT_BPM_MAX):
        super().__init__(daemon=True)
        self.shared = shared
        self.name_hint = name_hint
        self.p = None
        self.stream = None
        self.rate = 44100
        self.band_edges = None
        self.bass_hist = deque(maxlen=BEAT_HISTORY)
        self.beat_times = deque(maxlen=8)
        self.disp = np.zeros(N_BANDS, dtype=np.float32)
        self.peak = 1e-6
        # Tempo range -> debounce. Allow ~8% above bpm_max so a slightly early
        # detection isn't suppressed, but still block between-kick doubles.
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.min_interval = 60.0 / bpm_max * 0.92
        self.max_interval = 60.0 / bpm_min * 1.15

    def open(self):
        self.p = pyaudio.PyAudio()
        dev = find_loopback(self.p, self.name_hint)
        if dev is None:
            raise RuntimeError("No WASAPI loopback device found.")
        self.rate = int(dev["defaultSampleRate"])
        self.channels = dev["maxInputChannels"]
        self.stream = self.p.open(
            format=pyaudio.paInt16, channels=self.channels, rate=self.rate,
            input=True, input_device_index=dev["index"],
            frames_per_buffer=CHUNK)
        # Log-spaced band edges, clamped to Nyquist.
        fmax = min(FMAX, self.rate / 2 - 1)
        self.band_edges = np.geomspace(FMIN, fmax, N_BANDS + 1)
        self.freqs = np.fft.rfftfreq(CHUNK, 1.0 / self.rate)
        self.window = np.hanning(CHUNK).astype(np.float32)
        # Precompute bin index ranges per band.
        self.band_bins = []
        for i in range(N_BANDS):
            lo = np.searchsorted(self.freqs, self.band_edges[i], "left")
            hi = np.searchsorted(self.freqs, self.band_edges[i + 1], "left")
            if hi <= lo:
                hi = lo + 1
            self.band_bins.append((lo, hi))
        self.bass_lo = np.searchsorted(self.freqs, BASS_LO, "left")
        self.bass_hi = max(self.bass_lo + 1,
                           np.searchsorted(self.freqs, BASS_HI, "left"))
        return dev

    def analyze(self, mono):
        mag = np.abs(np.fft.rfft(mono * self.window))

        # Per-band magnitude (mean over the band's bins).
        raw = np.empty(N_BANDS, dtype=np.float32)
        for i, (lo, hi) in enumerate(self.band_bins):
            raw[i] = mag[lo:hi].mean()

        # Auto-gain: normalize by a slowly-decaying peak so bars fill nicely.
        self.peak = max(raw.max(), self.peak * 0.995, 1e-6)
        norm = np.clip(raw / self.peak, 0.0, 1.0) ** 0.5  # gamma for the eye

        # Attack fast, decay slow for pleasant motion.
        attack, decay = 0.6, 0.15
        rising = norm > self.disp
        self.disp = np.where(rising,
                             self.disp + (norm - self.disp) * attack,
                             self.disp + (norm - self.disp) * decay)

        energy = float(self.disp.mean())
        s = self.disp.sum()
        centroid = float((self.disp * np.arange(N_BANDS)).sum() / s / (N_BANDS - 1)) if s > 1e-6 else 0.0

        # Beat detection on raw bass energy (scale-invariant ratio to local mean).
        bass = float(mag[self.bass_lo:self.bass_hi].mean())
        now = time.monotonic()
        is_beat = False
        if len(self.bass_hist) == BEAT_HISTORY:
            local_mean = sum(self.bass_hist) / len(self.bass_hist)
            floor = self.peak * self.shared.beat_threshold
            if (bass > local_mean * BEAT_SENS and bass > floor and
                    now - self.shared.last_beat > self.min_interval):
                is_beat = True
        self.bass_hist.append(bass)

        with self.shared.lock:
            self.shared.bands = self.disp.copy()
            self.shared.energy = energy
            self.shared.centroid = centroid
            if is_beat:
                self.shared.last_beat = now
                self.shared.beat_count += 1
                self.beat_times.append(now)
                if len(self.beat_times) >= 4:
                    intervals = np.diff(self.beat_times)
                    # Keep only intervals plausible for the target tempo range
                    # so a single missed/extra kick doesn't skew the estimate.
                    good = intervals[(intervals >= self.min_interval) &
                                     (intervals <= self.max_interval)]
                    med = float(np.median(good)) if good.size else float(np.median(intervals))
                    if med > 0:
                        bpm = 60.0 / med
                        # Octave-correct into the expected range (fix half/double).
                        while bpm < self.bpm_min:
                            bpm *= 2.0
                        while bpm > self.bpm_max:
                            bpm /= 2.0
                        self.shared.bpm = bpm
        return is_beat

    def run(self):
        try:
            self.open()
        except Exception as e:  # noqa: BLE001
            print(f"[audio] failed to open loopback: {e}")
            with self.shared.lock:
                self.shared.running = False
            return
        while True:
            with self.shared.lock:
                if not self.shared.running:
                    break
            try:
                raw = self.stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                continue
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if self.channels > 1:
                data = data.reshape(-1, self.channels).mean(axis=1)
            if len(data) < CHUNK:
                data = np.pad(data, (0, CHUNK - len(data)))
            self.analyze(data[:CHUNK])
        self.cleanup()

    def cleanup(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            if self.p:
                self.p.terminate()


# ---- Light control ----------------------------------------------------------
def audio_to_color(centroid, energy, beat_env, intensity=1.0,
                   template=DEFAULT_TEMPLATE):
    """Map analysis -> RGB. energy=glow; beat=pop; template picks the palette.

    intensity (0..1) scales overall brightness — lower it if the light is too
    bright. template selects a COLOR_TEMPLATES palette mapping centroid -> hue.
    """
    hue_start, hue_span = COLOR_TEMPLATES.get(template,
                                              COLOR_TEMPLATES[DEFAULT_TEMPLATE])
    hue = (hue_start + centroid * hue_span) % 1.0
    base = 0.10 + 0.55 * energy
    value = min(1.0, max(base, beat_env)) * intensity
    sat = 1.0 - 0.15 * beat_env      # whiten slightly on a strong beat
    r, g, b = colorsys.hsv_to_rgb(hue, sat, value)
    return int(r * 255), int(g * 255), int(b * 255)


class LightController(threading.Thread):
    def __init__(self, shared, ip, fps=LIGHT_FPS):
        super().__init__(daemon=True)
        self.shared = shared
        self.ip = ip
        self.interval = 1.0 / fps
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)

    def _send(self, msg):
        try:
            self.sock.sendto(json.dumps(msg).encode(), (self.ip, CONTROL_PORT))
        except OSError:
            pass

    def run(self):
        self._send({"msg": {"cmd": "turn", "data": {"value": 1}}})
        self._send({"msg": {"cmd": "brightness", "data": {"value": 100}}})
        while True:
            with self.shared.lock:
                if not self.shared.running:
                    break
                energy = self.shared.energy
                centroid = self.shared.centroid
                last_beat = self.shared.last_beat
                intensity = self.shared.intensity
                template = self.shared.template
            beat_env = math.exp(-(time.monotonic() - last_beat) / BEAT_TAU)
            r, g, b = audio_to_color(centroid, energy, beat_env,
                                     intensity, template)
            self._send({"msg": {"cmd": "colorwc", "data": {
                "color": {"r": r, "g": g, "b": b}, "colorTemInKelvin": 0}}})
            with self.shared.lock:
                self.shared.color = (r, g, b)
            time.sleep(self.interval)
        # Rest on a solid dim red.
        self._send({"msg": {"cmd": "colorwc", "data": {
            "color": {"r": 150, "g": 0, "b": 0}, "colorTemInKelvin": 0}}})
        self.sock.close()


# ---- Modes ------------------------------------------------------------------
def list_devices():
    with pyaudio.PyAudio() as p:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        print(f"Default output: {default_out['name']}\nLoopback devices:")
        for lb in p.get_loopback_device_info_generator():
            print(f"  idx={lb['index']} ch={lb['maxInputChannels']} "
                  f"rate={int(lb['defaultSampleRate'])}  {lb['name']}")


def selftest(seconds, ip, use_light, bpm_min, bpm_max, intensity,
             template, beat_threshold):
    """Headless capture: print band summary + beats. Verifies the pipeline."""
    shared = SharedState()
    shared.intensity, shared.template, shared.beat_threshold = (
        intensity, template, beat_threshold)
    analyzer = AudioAnalyzer(shared, bpm_min=bpm_min, bpm_max=bpm_max)
    try:
        dev = analyzer.open()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot open audio: {e}")
        return 1
    print(f"Capturing from: {dev['name']}  ({analyzer.rate} Hz)")
    light = LightController(shared, ip) if use_light else None
    if light:
        light.start()

    end = time.monotonic() + seconds
    last_print = 0.0
    while time.monotonic() < end:
        raw = analyzer.stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if analyzer.channels > 1:
            data = data.reshape(-1, analyzer.channels).mean(axis=1)
        if len(data) < CHUNK:
            data = np.pad(data, (0, CHUNK - len(data)))
        beat = analyzer.analyze(data[:CHUNK])
        now = time.monotonic()
        if beat or now - last_print > 0.5:
            last_print = now
            bands, energy, centroid, _, bc, bpm, _ = shared.snapshot()
            # 16-char coarse spectrogram from the 48 bands.
            coarse = bands.reshape(16, -1).mean(axis=1)
            blocks = " .:-=+*#%@"
            spark = "".join(blocks[min(9, int(v * 9.999))] for v in coarse)
            tag = " <BEAT>" if beat else ""
            print(f"[{spark}] E={energy:.2f} C={centroid:.2f} "
                  f"beats={bc} bpm={bpm:.0f}{tag}")
    with shared.lock:
        shared.running = False
    print(f"\nDone. Total beats detected: {shared.beat_count}, "
          f"est BPM: {shared.bpm:.0f}")
    analyzer.cleanup()
    if light:
        light.join(timeout=1.0)
    return 0


def run_gui(ip, use_light, bpm_min, bpm_max, intensity, template, beat_threshold):
    import tkinter as tk

    shared = SharedState()
    shared.intensity, shared.template, shared.beat_threshold = (
        intensity, template, beat_threshold)
    analyzer = AudioAnalyzer(shared, bpm_min=bpm_min, bpm_max=bpm_max)
    analyzer.start()
    time.sleep(0.3)
    with shared.lock:
        if not shared.running:
            print("Audio failed to start; see message above.")
            return 1
    light = LightController(shared, ip) if use_light else None
    if light:
        light.start()

    W, H = 960, 520
    PAD = 16
    root = tk.Tk()
    root.title("Govee VJ — audio-reactive light")
    root.configure(bg="#0a0a0f")
    canvas = tk.Canvas(root, width=W, height=H, bg="#0a0a0f",
                       highlightthickness=0)
    canvas.pack()

    # ---- Live controls: palette / intensity / beat threshold ---------------
    BG, FG = "#0a0a0f", "#c8c8d4"
    ctl = tk.Frame(root, bg=BG)
    ctl.pack(fill="x", padx=PAD, pady=(0, 10))

    def label(text):
        tk.Label(ctl, text=text, bg=BG, fg="#8a8a98",
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 4))

    def set_template(name):
        with shared.lock:
            shared.template = name

    def set_intensity(v):
        with shared.lock:
            shared.intensity = float(v)

    def set_beat_threshold(v):
        with shared.lock:
            shared.beat_threshold = float(v)

    label("Palette")
    tvar = tk.StringVar(value=template)
    om = tk.OptionMenu(ctl, tvar, *sorted(COLOR_TEMPLATES),
                       command=set_template)
    om.configure(bg="#1a1a24", fg=FG, activebackground="#2a2a38",
                 activeforeground=FG, highlightthickness=0, bd=0,
                 font=("Segoe UI", 10), width=9)
    om["menu"].configure(bg="#1a1a24", fg=FG, activebackground="#2a2a38")
    om.pack(side="left", padx=(0, 18))

    scale_kw = dict(orient="horizontal", bg=BG, fg=FG, troughcolor="#1a1a24",
                    highlightthickness=0, bd=0, length=150,
                    font=("Segoe UI", 8), sliderrelief="flat")
    label("Intensity")
    iscale = tk.Scale(ctl, from_=0.05, to=1.0, resolution=0.05,
                      command=set_intensity, **scale_kw)
    iscale.set(intensity)
    iscale.pack(side="left", padx=(0, 18))
    label("Beat thresh")
    bscale = tk.Scale(ctl, from_=0.0, to=0.20, resolution=0.005,
                      command=set_beat_threshold, **scale_kw)
    bscale.set(beat_threshold)
    bscale.pack(side="left")

    def on_close():
        with shared.lock:
            shared.running = False
        root.after(200, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)

    def draw():
        with shared.lock:
            if not shared.running:
                root.destroy()
                return
        bands, energy, centroid, last_beat, bc, bpm, color = shared.snapshot()
        beat_env = math.exp(-(time.monotonic() - last_beat) / BEAT_TAU)
        canvas.delete("all")

        # Beat flash: subtle full-canvas tint on a strong beat.
        if beat_env > 0.05:
            cr, cg, cb = color
            canvas.create_rectangle(
                0, 0, W, H, fill=f"#{int(cr*0.15*beat_env):02x}"
                f"{int(cg*0.15*beat_env):02x}{int(cb*0.15*beat_env):02x}",
                outline="")

        # Spectrum bars.
        n = len(bands)
        bw = (W - 2 * PAD) / n
        base_y = H - 60
        for i, v in enumerate(bands):
            x0 = PAD + i * bw
            x1 = x0 + bw * 0.8
            bh = v * (base_y - PAD)
            hue = i / n * 0.66
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.5 + 0.5 * v)
            col = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            canvas.create_rectangle(x0, base_y - bh, x1, base_y,
                                    fill=col, outline="")

        # Color swatch (what the light shows).
        cr, cg, cb = color
        sw = f"#{cr:02x}{cg:02x}{cb:02x}"
        canvas.create_rectangle(W - 150, H - 46, W - 20, H - 12,
                                fill=sw, outline="#333")
        canvas.create_text(W - 155, H - 29, anchor="e", fill="#aaa",
                           text="light", font=("Segoe UI", 10))

        # Beat dot.
        rad = 10 + 22 * beat_env
        canvas.create_oval(PAD, H - 52, PAD + 2 * rad, H - 52 + 2 * rad,
                           fill=sw, outline="")
        # HUD text.
        canvas.create_text(
            PAD, 18, anchor="w", fill="#e8e8f0", font=("Segoe UI", 13, "bold"),
            text=f"Govee VJ   beats:{bc}   ~{bpm:.0f} BPM   "
                 f"energy:{energy:.2f}   {'LIGHT '+ip if use_light else 'no light'}")
        canvas.create_text(
            PAD + 70, H - 40, anchor="w", fill="#888", font=("Segoe UI", 10),
            text="bass                    mids                    treble")

        root.after(16, draw)  # ~60 fps UI

    draw()
    root.mainloop()

    with shared.lock:
        shared.running = False
    analyzer.join(timeout=1.0)
    if light:
        light.join(timeout=1.0)
    return 0


def main():
    p = argparse.ArgumentParser(description="Govee audio-reactive VJ.")
    p.add_argument("--ip", default=DEFAULT_IP, help="Govee device IP.")
    p.add_argument("--no-light", action="store_true", help="Visualizer only.")
    p.add_argument("--intensity", type=float, default=DEFAULT_INTENSITY,
                   help="Light brightness scale 0.05-1.0 (lower = dimmer).")
    p.add_argument("--color-template", default=DEFAULT_TEMPLATE,
                   choices=sorted(COLOR_TEMPLATES),
                   help="Color palette mapping spectrum -> hue.")
    p.add_argument("--beat-threshold", type=float, default=DEFAULT_BEAT_THRESHOLD,
                   help="Min bass level (fraction of peak, 0-1) a beat must "
                        "exceed to fire a light effect; raise to ignore quiet "
                        "passages.")
    p.add_argument("--bpm-min", type=float, default=DEFAULT_BPM_MIN,
                   help="Lowest expected tempo (e.g. 120 for prog house).")
    p.add_argument("--bpm-max", type=float, default=DEFAULT_BPM_MAX,
                   help="Highest expected tempo (e.g. 140 for prog house).")
    p.add_argument("--devices", action="store_true", help="List loopback devices.")
    p.add_argument("--selftest", nargs="?", type=float, const=5.0,
                   metavar="SECONDS", help="Headless capture test.")
    args = p.parse_args()

    if args.devices:
        list_devices()
        return 0
    use_light = not args.no_light
    intensity = max(0.05, min(1.0, args.intensity))
    beat_threshold = max(0.0, min(1.0, args.beat_threshold))
    template = args.color_template
    if args.selftest is not None:
        return selftest(args.selftest, args.ip, use_light,
                        args.bpm_min, args.bpm_max, intensity,
                        template, beat_threshold)
    return run_gui(args.ip, use_light, args.bpm_min, args.bpm_max, intensity,
                   template, beat_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
