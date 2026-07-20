"""Light Studio — design and animate virtual lights in a UI, then sync them
to real Govee devices over the LAN API.

The stage holds virtual lights (bulbs, strips, and bar pairs like the Govee
Flow Plus TV light bars). Each light runs its own
animated effect (solid / breathe / rainbow / chase / wave / strobe / groovy)
with a base color, speed, and brightness. Drag lights into place on the dark
stage; they render with a soft glow at ~30 fps.

The window is resizable and the stage grows with it; F11 toggles full-screen
(Esc exits).

Audio: the spectrum / vu / pulse / beat effects react to whatever is playing on the
default output (WASAPI loopback via vj.py's analyzer — enable "React to sound"
in the AUDIO section, or just pick an audio effect and capture auto-starts).
While capture is on, the stage bottom shows a live oscilloscope waveform and
the full 48-band spectrum (toggle with the Visualizer checkbox).

Sync: scan the LAN for Govee devices (or type an IP), assign a device to a
virtual light, and flip Sync on — every mapped light streams its animated
color to its device in real time. The LAN API is whole-device color only, so
a strip syncs the average of its segment colors.

Usage:
    python light_studio.py                 # open the studio
    python light_studio.py --selftest      # headless effect-engine test
    python light_studio.py --sync-fps 20   # device update rate (max 25)
    python light_studio.py --iface 10.0.0.5  # multicast interface for scan

Layouts persist to studio_layout.json via Save / Load.
"""

import argparse
import colorsys
import json
import math
import os
import threading
import time

import govee_lan

try:
    import vj as vj_audio            # audio capture + FFT (needs pyaudiowpatch)
except Exception:                    # noqa: BLE001 — audio is optional
    vj_audio = None

LAYOUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "studio_layout.json")

# ---- Stage / UI config -------------------------------------------------------
STAGE_W, STAGE_H = 940, 620
PANEL_W = 300
TICK_MS = 33               # ~30 fps animation
BG = "#0a0a0f"             # stage background (matches vj.py)
PANEL_BG = "#12121a"
CTL_BG = "#1a1a24"
FG = "#c8c8d4"
DIM = "#8a8a98"
ACCENT = "#5ad1ff"
FONT = ("Segoe UI", 10)

BULB_R = 24                # bulb core radius (px)
SEG_W, SEG_H = 16, 26      # strip segment size (px)
BAR_W, BAR_SEG_H = 16, 14  # bar-pair segment size (px)
BAR_GAP = 150              # gap between the two bars (the "TV" in between)

MAX_SYNC_FPS = 25          # protect devices from flooding


# ---- Effect engine -----------------------------------------------------------
# An effect maps (t, u, h, s, v) -> (h, s, v).
#   t: light-local time (already scaled by the light's speed, plus its phase)
#   u: position along the light, 0..1 (0 for a bulb)
#   h, s, v: the light's base color in HSV (0..1 each)
#
# Audio effects additionally read AUDIO, a module-level snapshot of the live
# capture (updated once per tick by the app; all zeros when capture is off).

N_COARSE_BANDS = 16
N_FULL_BANDS = vj_audio.N_BANDS if vj_audio else 48

AUDIO = {"on": False, "energy": 0.0, "centroid": 0.0, "beat": 0.0,
         "beats": 0,                                # total beat count
         "bands": [0.0] * N_COARSE_BANDS,          # coarse, for effects
         "spectrum": [0.0] * N_FULL_BANDS,          # full, for the visualizer
         "bpm": 0.0}

AUDIO_EFFECTS = ("spectrum", "vu", "pulse", "beat")

def _wrap_dist(a, b):
    d = abs(a - b)
    return min(d, 1.0 - d)


def eff_solid(t, u, h, s, v):
    return h, s, v


def eff_breathe(t, u, h, s, v):
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t / 3.0)
    return h, s, v * (0.30 + 0.70 * pulse)


def eff_rainbow(t, u, h, s, v):
    return (h + 0.12 * t + 0.75 * u) % 1.0, max(s, 0.85), v


def eff_chase(t, u, h, s, v):
    pos = (0.45 * t) % 1.0
    d = _wrap_dist(u, pos)
    return h, s, v * max(0.05, math.exp(-((d / 0.09) ** 2)))


def eff_wave(t, u, h, s, v):
    y = 0.5 + 0.5 * math.sin(2 * math.pi * (1.6 * u - 0.5 * t))
    return h, s, v * (0.12 + 0.88 * y)


def eff_strobe(t, u, h, s, v):
    return h, s, (v if (4.0 * t) % 1.0 < 0.5 else v * 0.02)


def eff_groovy(t, u, h, s, v):
    """Port of govee_effects.groovy_red_frame, centered on the base hue."""
    wob = (18.0 * math.sin(2 * math.pi * t / 6.0)
           + 6.0 * math.sin(2 * math.pi * t / 2.3)) / 360.0
    pulse = (0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 3.5))
             + 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 1.7)))
    return (h + wob) % 1.0, s, v * (0.32 + 0.68 * pulse)


def _audio_band(u):
    bands = AUDIO["bands"]
    return bands[min(len(bands) - 1, int(u * len(bands)))]


def eff_spectrum(t, u, h, s, v):
    """Audio: segments show the live spectrum (bass at u=0, treble at u=1)."""
    lvl = _audio_band(u)
    return (h + 0.20 * u) % 1.0, s, v * (0.04 + 0.96 * lvl)


def eff_vu(t, u, h, s, v):
    """Audio: VU meter — the light fills up with overall energy."""
    lit = 1.0 if u <= AUDIO["energy"] else 0.0
    return h, s, v * (0.04 + 0.96 * max(lit, 0.5 * AUDIO["beat"]))


def eff_pulse(t, u, h, s, v):
    """Audio: glow with energy, pop on beats, hue leans with the spectrum."""
    e, beat, cen = AUDIO["energy"], AUDIO["beat"], AUDIO["centroid"]
    hue = (h + 0.15 * (cen - 0.5)) % 1.0
    return hue, s * (1.0 - 0.25 * beat), v * min(1.0, 0.12 + 0.60 * e + 0.90 * beat)


def eff_beat(t, u, h, s, v):
    """Audio: dark until a beat hits, then flash out from the center.

    Each kick rotates the hue so consecutive beats pop in different colors;
    outer segments decay faster, so the flash radiates outward.
    """
    beat, count = AUDIO["beat"], AUDIO["beats"]
    hue = (h + 0.13 * count) % 1.0
    env = beat ** (1.0 + 2.5 * abs(u - 0.5))
    return hue, s, v * (0.03 + 0.97 * env)


EFFECTS = {
    "solid": eff_solid,
    "breathe": eff_breathe,
    "rainbow": eff_rainbow,
    "chase": eff_chase,
    "wave": eff_wave,
    "strobe": eff_strobe,
    "groovy": eff_groovy,
    "spectrum": eff_spectrum,
    "vu": eff_vu,
    "pulse": eff_pulse,
    "beat": eff_beat,
}


# ---- Virtual light model -----------------------------------------------------
class Light:
    _counter = 0

    def __init__(self, kind, x, y, name=None, segments=12):
        Light._counter += 1
        self.kind = kind                     # 'bulb' | 'strip' | 'bars'
        self.x, self.y = x, y                # center on the stage
        self.name = name or f"{kind.title()} {Light._counter}"
        self.segments = segments if kind in ("strip", "bars") else 1
        self.effect = "breathe"
        self.color = (255, 60, 40)           # base RGB
        self.speed = 1.0
        self.brightness = 1.0
        self.phase = (Light._counter * 1.7) % 10.0  # desync identical lights
        self.device_ip = None                # mapped Govee device (or None)
        self.colors = [(0, 0, 0)] * self.segments   # current animated colors

    def animate(self, t):
        """Compute per-segment colors at studio time t. Returns the list."""
        h0, s0, v0 = colorsys.rgb_to_hsv(*(c / 255.0 for c in self.color))
        fn = EFFECTS.get(self.effect, eff_solid)
        lt = t * self.speed + self.phase
        n = self.segments
        out = []
        for i in range(n):
            u = i / (n - 1) if n > 1 else 0.0
            h, s, v = fn(lt, u, h0, s0, v0)
            v = max(0.0, min(1.0, v)) * self.brightness
            r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), v)
            out.append((int(r * 255), int(g * 255), int(b * 255)))
        self.colors = out
        return out

    def avg_color(self):
        n = len(self.colors)
        r = sum(c[0] for c in self.colors) // n
        g = sum(c[1] for c in self.colors) // n
        b = sum(c[2] for c in self.colors) // n
        return r, g, b

    def set_segments(self, n):
        if self.kind not in ("strip", "bars"):
            return
        n = max(2, min(40, int(n)))
        if self.kind == "bars":
            n -= n % 2               # bars split segments evenly in two
        self.segments = n
        self.colors = [(0, 0, 0)] * self.segments

    def bbox(self):
        """Stage bounding box (x0, y0, x1, y1) for hit-testing/clamping."""
        if self.kind == "bulb":
            r = BULB_R + 8
            return self.x - r, self.y - r, self.x + r, self.y + r
        if self.kind == "bars":
            w = BAR_GAP / 2 + BAR_W / 2 + 6
            h = (self.segments // 2) * BAR_SEG_H / 2 + 6
            return self.x - w, self.y - h, self.x + w, self.y + h
        w = self.segments * SEG_W / 2 + 6
        h = SEG_H / 2 + 6
        return self.x - w, self.y - h, self.x + w, self.y + h

    def to_dict(self):
        return {"kind": self.kind, "x": self.x, "y": self.y,
                "name": self.name, "segments": self.segments,
                "effect": self.effect, "color": list(self.color),
                "speed": self.speed, "brightness": self.brightness,
                "device_ip": self.device_ip}

    @classmethod
    def from_dict(cls, d):
        light = cls(d["kind"], d["x"], d["y"], d.get("name"),
                    d.get("segments", 12))
        light.effect = d.get("effect", "breathe")
        light.color = tuple(d.get("color", (255, 60, 40)))
        light.speed = float(d.get("speed", 1.0))
        light.brightness = float(d.get("brightness", 1.0))
        light.device_ip = d.get("device_ip")
        return light


# ---- Audio bridge --------------------------------------------------------------
if vj_audio is not None:
    class _WaveAnalyzer(vj_audio.AudioAnalyzer):
        """AudioAnalyzer that also keeps the latest raw waveform chunk."""

        def __init__(self, shared):
            super().__init__(shared)
            self.wave = None

        def analyze(self, mono):
            self.wave = mono.copy()   # reference swap; UI reads it lock-free
            return super().analyze(mono)


class AudioBridge:
    """Runs vj.py's WASAPI-loopback AudioAnalyzer and exposes its features.

    The analyzer thread captures whatever plays on the default output; the UI
    thread polls features() once per tick into the AUDIO globals.
    """

    def __init__(self):
        self.shared = None
        self.analyzer = None

    def start(self):
        if vj_audio is None:
            return False
        self.shared = vj_audio.SharedState()
        self.analyzer = _WaveAnalyzer(self.shared)
        self.analyzer.start()
        return True

    def started(self):
        return self.analyzer is not None

    def active(self):
        """True while the capture thread is running (open() can fail late)."""
        if self.shared is None:
            return False
        with self.shared.lock:
            return self.shared.running

    def stop(self):
        if self.shared:
            with self.shared.lock:
                self.shared.running = False
        self.shared = None
        self.analyzer = None

    def features(self):
        bands, energy, centroid, last_beat, cnt, bpm, _col = \
            self.shared.snapshot()
        beat = math.exp(-(time.monotonic() - last_beat) / vj_audio.BEAT_TAU)
        coarse = bands.reshape(N_COARSE_BANDS, -1).mean(axis=1)
        return {"energy": energy, "centroid": centroid,
                "beat": min(1.0, beat), "beats": cnt,
                "bands": [float(x) for x in coarse],
                "spectrum": [float(x) for x in bands], "bpm": bpm}

    def wave(self):
        """Latest raw waveform chunk (float32 -1..1) or None."""
        return self.analyzer.wave if self.analyzer else None


# ---- Govee sync engine ---------------------------------------------------------
class SyncEngine(threading.Thread):
    """Streams the studio's animated colors to mapped Govee devices.

    The UI thread pushes {ip: (r, g, b)} via set_targets(); this thread sends
    UDP color commands at a capped fps. New IPs get a power-on + brightness
    command once per enable.
    """

    def __init__(self, fps):
        super().__init__(daemon=True)
        self.interval = 1.0 / max(1, min(MAX_SYNC_FPS, fps))
        self.lock = threading.Lock()
        self.targets = {}
        self.enabled = False
        self.running = True
        self.sent = 0
        self._powered = set()

    def set_targets(self, mapping):
        with self.lock:
            self.targets = dict(mapping)

    def set_enabled(self, on):
        with self.lock:
            self.enabled = on
            if on:
                self._powered.clear()

    def stop(self):
        with self.lock:
            self.running = False

    def run(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                enabled = self.enabled
                targets = dict(self.targets)
                powered = set(self._powered)
            if enabled:
                for ip, (r, g, b) in targets.items():
                    try:
                        if ip not in powered:
                            govee_lan.set_power(ip, True)
                            govee_lan.set_brightness(ip, 100)
                            with self.lock:
                                self._powered.add(ip)
                        govee_lan.set_color(ip, r, g, b)
                        self.sent += 1
                    except OSError:
                        pass
            time.sleep(self.interval)


# ---- Studio app ---------------------------------------------------------------
class StudioApp:
    def __init__(self, root, sync_fps=20, iface=None):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.iface = iface
        root.title("Light Studio — design, animate, sync")
        root.configure(bg=BG)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
        root.minsize(700, 420)

        self.lights = []
        self.selected = None
        self.stage_w, self.stage_h = STAGE_W, STAGE_H  # live canvas size
        self.playing = True
        self.t = 0.0
        self._last_tick = time.monotonic()
        self._drag = None            # (light, dx, dy) while dragging
        self._loading_panel = False  # guard: programmatic widget updates
        self.devices = []            # [(label, ip)] from LAN scan
        self._scan_results = None    # thread handoff
        self._scanning = False

        self.sync = SyncEngine(sync_fps)
        self.sync.start()
        self.audio = AudioBridge()

        self._build_ui()
        self._tick()

    # ---- UI construction ----
    def _build_ui(self):
        tk = self.tk
        self.canvas = tk.Canvas(self.root, width=STAGE_W, height=STAGE_H,
                                bg=BG, highlightthickness=0, cursor="hand2")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>", self._on_stage_resize)

        panel = tk.Frame(self.root, width=PANEL_W, bg=PANEL_BG)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        self.panel = panel

        def section(text, pady=(12, 4)):
            tk.Label(panel, text=text, bg=PANEL_BG, fg=DIM,
                     font=("Segoe UI", 9, "bold")).pack(
                anchor="w", padx=12, pady=pady)

        def button(parent, text, cmd, **kw):
            b = tk.Button(parent, text=text, command=cmd, bg=CTL_BG, fg=FG,
                          activebackground="#2a2a38", activeforeground=FG,
                          bd=0, font=FONT, padx=10, pady=3, **kw)
            return b

        # -- Stage controls --
        section("STAGE", pady=(14, 4))
        row = tk.Frame(panel, bg=PANEL_BG)
        row.pack(fill="x", padx=12)
        button(row, "+ Bulb", self.add_bulb).pack(side="left", padx=(0, 6))
        button(row, "+ Strip", self.add_strip).pack(side="left", padx=(0, 6))
        button(row, "+ Bars", self.add_bars).pack(side="left", padx=(0, 6))
        button(row, "Delete", self.delete_selected).pack(side="left")

        row2 = tk.Frame(panel, bg=PANEL_BG)
        row2.pack(fill="x", padx=12, pady=(6, 0))
        self.play_btn = button(row2, "Pause", self.toggle_play)
        self.play_btn.pack(side="left", padx=(0, 6))
        button(row2, "Save", self.save_layout).pack(side="left", padx=(0, 6))
        button(row2, "Load", self.load_layout).pack(side="left")

        # -- Selected light properties --
        section("SELECTED LIGHT")
        self.sel_label = tk.Label(panel, text="(click a light)", bg=PANEL_BG,
                                  fg=FG, font=("Segoe UI", 11, "bold"))
        self.sel_label.pack(anchor="w", padx=12)

        grid = tk.Frame(panel, bg=PANEL_BG)
        grid.pack(fill="x", padx=12, pady=(6, 0))

        def prop_label(r, text):
            tk.Label(grid, text=text, bg=PANEL_BG, fg=DIM, font=FONT).grid(
                row=r, column=0, sticky="w", pady=2)

        prop_label(0, "Effect")
        self.effect_var = tk.StringVar(value="breathe")
        om = tk.OptionMenu(grid, self.effect_var, *EFFECTS,
                           command=self._on_effect)
        om.configure(bg=CTL_BG, fg=FG, activebackground="#2a2a38",
                     activeforeground=FG, highlightthickness=0, bd=0,
                     font=FONT, width=10)
        om["menu"].configure(bg=CTL_BG, fg=FG, activebackground="#2a2a38")
        om.grid(row=0, column=1, sticky="w", padx=(8, 0))

        prop_label(1, "Color")
        self.color_btn = tk.Button(grid, text="  pick  ", bg="#ff3c28",
                                   fg="#000", bd=0, font=FONT,
                                   command=self._on_pick_color)
        self.color_btn.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=2)

        scale_kw = dict(orient="horizontal", bg=PANEL_BG, fg=FG,
                        troughcolor=CTL_BG, highlightthickness=0, bd=0,
                        length=150, font=("Segoe UI", 8), sliderrelief="flat")
        prop_label(2, "Speed")
        self.speed_scale = tk.Scale(grid, from_=0.1, to=3.0, resolution=0.05,
                                    command=self._on_speed, **scale_kw)
        self.speed_scale.set(1.0)
        self.speed_scale.grid(row=2, column=1, sticky="w", padx=(8, 0))

        prop_label(3, "Brightness")
        self.bright_scale = tk.Scale(grid, from_=0.05, to=1.0, resolution=0.05,
                                     command=self._on_brightness, **scale_kw)
        self.bright_scale.set(1.0)
        self.bright_scale.grid(row=3, column=1, sticky="w", padx=(8, 0))

        prop_label(4, "Segments")
        self.seg_scale = tk.Scale(grid, from_=2, to=40, resolution=1,
                                  command=self._on_segments, **scale_kw)
        self.seg_scale.set(12)
        self.seg_scale.grid(row=4, column=1, sticky="w", padx=(8, 0))

        # -- Audio --
        section("AUDIO")
        arow = tk.Frame(panel, bg=PANEL_BG)
        arow.pack(fill="x", padx=12)
        self.audio_var = tk.BooleanVar(value=False)
        acb = tk.Checkbutton(arow, text="React to sound output",
                             variable=self.audio_var,
                             command=self._on_audio_toggle, bg=PANEL_BG,
                             fg=FG, selectcolor=CTL_BG,
                             activebackground=PANEL_BG, activeforeground=FG,
                             font=FONT)
        acb.pack(side="left")
        self.viz_var = tk.BooleanVar(value=True)
        vcb = tk.Checkbutton(arow, text="Visualizer", variable=self.viz_var,
                             bg=PANEL_BG, fg=FG, selectcolor=CTL_BG,
                             activebackground=PANEL_BG, activeforeground=FG,
                             font=FONT)
        vcb.pack(side="left", padx=(8, 0))
        self.audio_label = tk.Label(panel, text="(off)", bg=PANEL_BG, fg=DIM,
                                    font=("Segoe UI", 9))
        self.audio_label.pack(anchor="w", padx=12)

        # -- Govee sync --
        section("GOVEE SYNC")
        srow = tk.Frame(panel, bg=PANEL_BG)
        srow.pack(fill="x", padx=12)
        self.scan_btn = button(srow, "Scan LAN", self.scan_devices)
        self.scan_btn.pack(side="left", padx=(0, 8))
        self.sync_var = tk.BooleanVar(value=False)
        cb = tk.Checkbutton(srow, text="Sync ON", variable=self.sync_var,
                            command=self._on_sync_toggle, bg=PANEL_BG, fg=FG,
                            selectcolor=CTL_BG, activebackground=PANEL_BG,
                            activeforeground=FG, font=FONT)
        cb.pack(side="left")

        tk.Label(panel, text="Device for selected light:", bg=PANEL_BG,
                 fg=DIM, font=FONT).pack(anchor="w", padx=12, pady=(8, 0))
        self.device_var = tk.StringVar(value="(none)")
        self.device_menu = tk.OptionMenu(panel, self.device_var, "(none)",
                                         command=self._on_device_pick)
        self.device_menu.configure(bg=CTL_BG, fg=FG,
                                   activebackground="#2a2a38",
                                   activeforeground=FG, highlightthickness=0,
                                   bd=0, font=FONT, width=24)
        self.device_menu["menu"].configure(bg=CTL_BG, fg=FG,
                                           activebackground="#2a2a38")
        self.device_menu.pack(anchor="w", padx=12, pady=(2, 0))

        irow = tk.Frame(panel, bg=PANEL_BG)
        irow.pack(fill="x", padx=12, pady=(6, 0))
        self.ip_entry = tk.Entry(irow, bg=CTL_BG, fg=FG, bd=0,
                                 insertbackground=FG, font=FONT, width=14)
        self.ip_entry.pack(side="left", ipady=3)
        button(irow, "Assign IP", self._on_assign_ip).pack(side="left",
                                                           padx=(8, 0))

        self.status = tk.Label(panel, text="", bg=PANEL_BG, fg=DIM,
                               font=("Segoe UI", 9), justify="left",
                               wraplength=PANEL_W - 24)
        self.status.pack(anchor="w", padx=12, pady=(12, 0))
        self._set_status("Add a light, pick an effect, then scan "
                         "and assign a Govee device to sync. "
                         "F11 = full-screen.")

    def _set_status(self, text):
        self.status.configure(text=text)

    # ---- Window / stage geometry ----
    def _toggle_fullscreen(self, _ev=None):
        full = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not full)

    def _on_stage_resize(self, ev):
        self.stage_w, self.stage_h = ev.width, ev.height
        # Keep lights reachable when the stage shrinks.
        for light in self.lights:
            light.x = min(light.x, self.stage_w - 20)
            light.y = min(light.y, self.stage_h - 20)

    # ---- Stage actions ----
    def add_bulb(self):
        light = Light("bulb", self.stage_w // 2, self.stage_h // 2)
        self.lights.append(light)
        self.select(light)

    def add_strip(self):
        light = Light("strip", self.stage_w // 2, self.stage_h // 2)
        self.lights.append(light)
        self.select(light)

    def add_bars(self):
        """Bar pair like the Govee Flow Plus (TV light bars)."""
        light = Light("bars", self.stage_w // 2, self.stage_h // 2)
        self.lights.append(light)
        self.select(light)

    def delete_selected(self):
        if self.selected in self.lights:
            self.lights.remove(self.selected)
        self.select(None)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")

    def select(self, light):
        self.selected = light
        self._refresh_panel()

    def _refresh_panel(self):
        """Load the selected light's values into the property widgets."""
        self._loading_panel = True
        try:
            light = self.selected
            if light is None:
                self.sel_label.configure(text="(click a light)")
                self.device_var.set("(none)")
                return
            self.sel_label.configure(text=light.name)
            self.effect_var.set(light.effect)
            r, g, b = light.color
            self.color_btn.configure(bg=f"#{r:02x}{g:02x}{b:02x}")
            self.speed_scale.set(light.speed)
            self.bright_scale.set(light.brightness)
            if light.kind in ("strip", "bars"):
                self.seg_scale.configure(state="normal", fg=FG)
                self.seg_scale.set(light.segments)
            else:
                self.seg_scale.configure(state="disabled", fg=DIM)
            self.device_var.set(light.device_ip or "(none)")
        finally:
            self._loading_panel = False

    # ---- Property callbacks ----
    def _on_effect(self, name):
        if self._loading_panel or not self.selected:
            return
        self.selected.effect = name
        # Audio effects are dead without capture — start it automatically.
        if name in AUDIO_EFFECTS and not self.audio.started():
            self.audio_var.set(True)
            self._on_audio_toggle()

    def _on_pick_color(self):
        if not self.selected:
            return
        from tkinter import colorchooser
        rgb, _hex = colorchooser.askcolor(color=self.color_btn["bg"],
                                          title="Base color")
        if rgb:
            self.selected.color = tuple(int(c) for c in rgb)
            self._refresh_panel()

    def _on_speed(self, v):
        if self._loading_panel or not self.selected:
            return
        self.selected.speed = float(v)

    def _on_brightness(self, v):
        if self._loading_panel or not self.selected:
            return
        self.selected.brightness = float(v)

    def _on_segments(self, v):
        if self._loading_panel or not self.selected:
            return
        self.selected.set_segments(int(float(v)))

    # ---- Audio ----
    def _on_audio_toggle(self):
        if self.audio_var.get():
            if self.audio.started():
                return
            if not self.audio.start():
                self.audio_var.set(False)
                self._set_status("Audio needs vj.py + pyaudiowpatch "
                                 "(pip install pyaudiowpatch numpy).")
                return
            self._set_status("Audio ON — capturing the default output. "
                             "Try the spectrum / vu / pulse effects.")
        else:
            self.audio.stop()
            self.audio_label.configure(text="(off)")
            self._set_status("Audio off.")

    def _update_audio(self):
        """Refresh the AUDIO globals from the capture thread (or zero them)."""
        if self.audio.started() and not self.audio.active():
            # Analyzer thread died — usually no loopback device.
            self.audio.stop()
            self.audio_var.set(False)
            self.audio_label.configure(text="(off)")
            self._set_status("Audio capture failed — no WASAPI loopback "
                             "device? Run 'python vj.py --devices' to check.")
        if self.audio.active():
            AUDIO.update(self.audio.features())
            AUDIO["on"] = True
            self.audio_label.configure(
                text=f"energy {AUDIO['energy']:.2f}   "
                     f"~{AUDIO['bpm']:.0f} bpm")
        elif AUDIO["on"]:
            AUDIO.update({"on": False, "energy": 0.0, "centroid": 0.0,
                          "beat": 0.0, "beats": 0,
                          "bands": [0.0] * N_COARSE_BANDS,
                          "spectrum": [0.0] * N_FULL_BANDS, "bpm": 0.0})

    # ---- Sync / devices ----
    def scan_devices(self):
        if self._scanning:
            return
        self._scanning = True
        self.scan_btn.configure(text="Scanning...", state="disabled")
        self._set_status("Scanning LAN for Govee devices (3s)...")

        def worker():
            try:
                found = govee_lan.scan(timeout=3.0, iface=self.iface)
            except OSError as e:
                found = e
            self._scan_results = found

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(200, self._poll_scan)

    def _poll_scan(self):
        if self._scan_results is None:
            self.root.after(200, self._poll_scan)
            return
        results, self._scan_results = self._scan_results, None
        self._scanning = False
        self.scan_btn.configure(text="Scan LAN", state="normal")
        if isinstance(results, OSError):
            self._set_status(f"Scan failed: {results}")
            return
        self.devices = [(f"{d.get('sku', '?')} {d.get('ip', '?')}",
                         d.get("ip")) for d in results if d.get("ip")]
        self._rebuild_device_menu()
        if self.devices:
            self._set_status(f"Found {len(self.devices)} device(s). Select a "
                             "light, then pick its device below.")
        else:
            self._set_status("No devices found. Is LAN Control enabled in the "
                             "Govee Home app? You can also type an IP below.")

    def _rebuild_device_menu(self):
        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        options = ["(none)"] + [label for label, _ip in self.devices]
        for opt in options:
            menu.add_command(
                label=opt,
                command=lambda o=opt: self._on_device_pick(o))

    def _on_device_pick(self, label):
        self.device_var.set(label)
        if self._loading_panel or not self.selected:
            return
        if label == "(none)":
            self.selected.device_ip = None
        else:
            for lbl, ip in self.devices:
                if lbl == label:
                    self.selected.device_ip = ip
                    break
            else:
                self.selected.device_ip = label  # a raw IP shown as-is
        self._refresh_panel()

    def _on_assign_ip(self):
        ip = self.ip_entry.get().strip()
        if not self.selected or not ip:
            return
        self.selected.device_ip = ip
        self._refresh_panel()
        self._set_status(f"{self.selected.name} → {ip}")

    def _on_sync_toggle(self):
        on = self.sync_var.get()
        self.sync.set_enabled(on)
        if on:
            mapped = sum(1 for l in self.lights if l.device_ip)
            self._set_status(f"Sync ON — streaming {mapped} light(s). "
                             "Strips/bars send their average color.")
        else:
            self._set_status("Sync off.")

    # ---- Mouse interaction ----
    def _light_at(self, x, y):
        for light in reversed(self.lights):
            x0, y0, x1, y1 = light.bbox()
            if x0 <= x <= x1 and y0 <= y <= y1:
                return light
        return None

    def _on_click(self, ev):
        light = self._light_at(ev.x, ev.y)
        self.select(light)
        if light:
            self._drag = (light, ev.x - light.x, ev.y - light.y)

    def _on_drag(self, ev):
        if not self._drag:
            return
        light, dx, dy = self._drag
        light.x = max(20, min(self.stage_w - 20, ev.x - dx))
        light.y = max(20, min(self.stage_h - 20, ev.y - dy))

    def _on_release(self, _ev):
        self._drag = None

    # ---- Rendering ----
    @staticmethod
    def _blend(color, k, bg=(10, 10, 15)):
        """Blend color toward the stage background by factor k (0..1)."""
        r = int(bg[0] + (color[0] - bg[0]) * k)
        g = int(bg[1] + (color[1] - bg[1]) * k)
        b = int(bg[2] + (color[2] - bg[2]) * k)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_light(self, light):
        c = self.canvas
        if light.kind == "bulb":
            col = light.colors[0]
            x, y = light.x, light.y
            for rr, k in ((BULB_R * 2.3, 0.10), (BULB_R * 1.75, 0.22),
                          (BULB_R * 1.3, 0.45)):
                c.create_oval(x - rr, y - rr, x + rr, y + rr,
                              fill=self._blend(col, k), outline="")
            c.create_oval(x - BULB_R, y - BULB_R, x + BULB_R, y + BULB_R,
                          fill=self._blend(col, 1.0), outline="")
            label_y = y + BULB_R * 2.3 + 10
        elif light.kind == "bars":
            nper = light.segments // 2
            bh = nper * BAR_SEG_H
            y0 = light.y - bh / 2
            # Faint outline of the TV the bars flank.
            tv = BAR_GAP / 2 - BAR_W
            c.create_rectangle(light.x - tv, y0, light.x + tv, y0 + bh,
                               outline="#232330")
            for side, cols in ((-1, light.colors[:nper]),
                               (+1, light.colors[nper:])):
                bx0 = light.x + side * BAR_GAP / 2 - BAR_W / 2
                avg_r = sum(cc[0] for cc in cols) // nper
                avg_g = sum(cc[1] for cc in cols) // nper
                avg_b = sum(cc[2] for cc in cols) // nper
                c.create_rectangle(bx0 - 8, y0 - 8, bx0 + BAR_W + 8,
                                   y0 + bh + 8,
                                   fill=self._blend((avg_r, avg_g, avg_b),
                                                    0.12), outline="")
                for i, col in enumerate(cols):  # bottom → top
                    sy = y0 + bh - (i + 1) * BAR_SEG_H
                    c.create_rectangle(bx0, sy + 1, bx0 + BAR_W,
                                       sy + BAR_SEG_H - 1,
                                       fill=self._blend(col, 1.0), outline="")
            label_y = y0 + bh + 16
        else:
            n = light.segments
            w = n * SEG_W
            x0 = light.x - w / 2
            y0 = light.y - SEG_H / 2
            avg = light.avg_color()
            c.create_rectangle(x0 - 8, y0 - 8, x0 + w + 8, y0 + SEG_H + 8,
                               fill=self._blend(avg, 0.12), outline="")
            for i, col in enumerate(light.colors):
                sx = x0 + i * SEG_W
                c.create_rectangle(sx + 1, y0, sx + SEG_W - 1, y0 + SEG_H,
                                   fill=self._blend(col, 1.0), outline="")
            label_y = y0 + SEG_H + 18

        tag = f"{light.name}"
        if light.device_ip:
            tag += f"  → {light.device_ip}"
        c.create_text(light.x, label_y, text=tag, fill=DIM,
                      font=("Segoe UI", 9))
        if light is self.selected:
            x0, y0, x1, y1 = light.bbox()
            c.create_rectangle(x0, y0, x1, y1, outline=ACCENT, dash=(3, 3))

    def _draw_visualizer(self):
        """Waveform + full spectrum along the bottom of the stage."""
        c = self.canvas
        spec = AUDIO["spectrum"]
        n = len(spec)
        base = self.stage_h - 26     # bottom of the spectrum bars
        vh = 80                      # max bar height
        bw = (self.stage_w - 24) / n
        for i, v in enumerate(spec):
            x0 = 12 + i * bw
            hue = i / n * 0.66       # red (bass) .. blue (treble), as in vj.py
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.30 + 0.70 * v)
            c.create_rectangle(x0, base - v * vh, x0 + bw * 0.8, base,
                               fill=f"#{int(r * 255):02x}{int(g * 255):02x}"
                                    f"{int(b * 255):02x}", outline="")
        c.create_text(14, self.stage_h - 12, anchor="w", fill=DIM,
                      font=("Segoe UI", 8), text="bass")
        c.create_text(self.stage_w - 14, self.stage_h - 12, anchor="e",
                      fill=DIM, font=("Segoe UI", 8), text="treble")

        wave = self.audio.wave()
        if wave is None or not len(wave):
            return
        mid = base - vh - 34         # oscilloscope centerline
        peak = max(0.05, float(max(abs(wave.max()), abs(wave.min()))))
        gain = 28.0 / peak           # auto-gain into ±28 px
        step = max(1, len(wave) // 240)
        pts = []
        for j in range(0, len(wave), step):
            x = 12 + j / len(wave) * (self.stage_w - 24)
            pts += [x, mid - float(wave[j]) * gain]
        c.create_line(*pts, fill=ACCENT, width=1)

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        if not self.lights:
            c.create_text(self.stage_w / 2, self.stage_h / 2,
                          text="Add a bulb or strip to get started",
                          fill=DIM, font=("Segoe UI", 13))
        if AUDIO["on"] and self.viz_var.get():
            self._draw_visualizer()
        for light in self.lights:
            self._draw_light(light)
        state = "▶ playing" if self.playing else "❚❚ paused"
        sync = "sync ON" if self.sync_var.get() else "sync off"
        audio = f"♪ ~{AUDIO['bpm']:.0f} bpm" if AUDIO["on"] else "♪ off"
        c.create_text(12, 14, anchor="w", fill="#e8e8f0",
                      font=("Segoe UI", 11, "bold"),
                      text=f"Light Studio   {state}   {sync}   {audio}   "
                           f"{len(self.lights)} light(s)")

    # ---- Main loop ----
    def _tick(self):
        now = time.monotonic()
        dt = min(0.1, now - self._last_tick)
        self._last_tick = now
        if self.playing:
            self.t += dt
        self._update_audio()
        for light in self.lights:
            light.animate(self.t)
        # Push mapped colors to the sync engine (last write wins per IP).
        self.sync.set_targets({l.device_ip: l.avg_color()
                               for l in self.lights if l.device_ip})
        self._redraw()
        self.root.after(TICK_MS, self._tick)

    # ---- Persistence ----
    def save_layout(self):
        data = {"lights": [l.to_dict() for l in self.lights]}
        with open(LAYOUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._set_status(f"Saved {len(self.lights)} light(s) to "
                         f"{os.path.basename(LAYOUT_FILE)}")

    def load_layout(self):
        if not os.path.exists(LAYOUT_FILE):
            self._set_status("No saved layout yet — press Save first.")
            return
        with open(LAYOUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        self.lights = [Light.from_dict(d) for d in data.get("lights", [])]
        self.select(None)
        self._set_status(f"Loaded {len(self.lights)} light(s).")

    def shutdown(self):
        self.audio.stop()
        self.sync.set_enabled(False)
        self.sync.stop()


# ---- Modes --------------------------------------------------------------------
def selftest():
    """Headless: run every effect on a bulb and a strip, check output ranges."""
    print("Effect engine selftest:")
    ok = True
    # Fake audio features so spectrum/vu/pulse produce non-trivial output.
    AUDIO.update({"on": True, "energy": 0.6, "centroid": 0.5, "beat": 1.0,
                  "beats": 5,
                  "bands": [i / (N_COARSE_BANDS - 1)
                            for i in range(N_COARSE_BANDS)], "bpm": 120.0})
    for name in EFFECTS:
        for kind, segs in (("bulb", 1), ("strip", 8), ("bars", 8)):
            light = Light(kind, 0, 0, segments=segs)
            light.effect = name
            samples = []
            for step in range(6):
                cols = light.animate(step * 0.35)
                for rgb in cols:
                    if not all(0 <= c <= 255 for c in rgb):
                        print(f"  {name}/{kind}: out of range {rgb}")
                        ok = False
                samples.append(cols[0])
            swatch = " ".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in samples)
            print(f"  {name:<8} {kind:<5} {swatch}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def run_gui(sync_fps, iface):
    import tkinter as tk
    root = tk.Tk()
    app = StudioApp(root, sync_fps=sync_fps, iface=iface)

    # Tk windows launched from a terminal can open behind it without focus;
    # briefly pin on top so the studio is actually visible.
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()
    root.after(500, lambda: root.attributes("-topmost", False))

    def on_close():
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Light Studio — animate virtual lights, sync to Govee.")
    p.add_argument("--sync-fps", type=int, default=20,
                   help=f"Device update rate when syncing (max {MAX_SYNC_FPS}).")
    p.add_argument("--iface", help="Local IP of the interface for LAN scan "
                   "(fixes VPN/multi-adapter machines).")
    p.add_argument("--selftest", action="store_true",
                   help="Headless effect-engine test, no GUI.")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    return run_gui(args.sync_fps, args.iface)


if __name__ == "__main__":
    raise SystemExit(main())