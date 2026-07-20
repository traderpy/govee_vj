"""Show Studio — choreograph a light show to a song, then play it back on
virtual lights and real Govee devices.

Load an MP3 (or WAV/FLAC/OGG); the song is decoded and analyzed offline into a
waveform, a 48-band spectrogram, beat onsets and a BPM estimate, all drawn on a
zoomable timeline. Every light on the stage gets its own cue track under the
timeline: drag on an empty part of a track to paint a cue, then give the cue an
effect, color, speed and brightness. During playback each light runs whichever
cue is under the playhead (and is dark between cues), so scrubbing anywhere in
the song shows exactly what the show looks like at that moment.

All effects from light_studio are available inside cues — including the
audio-reactive ones (spectrum / vu / pulse / beat), which are driven by the
song's precomputed analysis, so they work identically when scrubbing, in
playback, and on every run.

Timeline controls:
    click ruler / waveform   seek (drag to scrub)
    drag on empty track      create a cue
    drag a cue               move it (drag its edges to resize)
    Ctrl + mouse wheel       zoom around the cursor
    mouse wheel              pan
    Space                    play / pause        Home  rewind
    Left / Right             nudge 1 s (Shift = 0.1 s)
    Delete                   delete selected cue (or light)

Cues snap to detected beats while "Snap to beats" is on.

Sync works like light_studio: scan the LAN, assign a device to a light, flip
Sync ON — during playback every mapped light streams its color to its device
(whole-device color; strips/bars send their average).

Usage:
    python show_studio.py                     # open the studio
    python show_studio.py --audio song.mp3    # open with a song loaded
    python show_studio.py --selftest          # headless decode/analysis test
    python show_studio.py --sync-fps 20 --iface 10.0.0.5
"""

import argparse
import bisect
import colorsys
import json
import math
import os
import threading
import time

import numpy as np

import light_studio as ls          # Light model, effect engine, SyncEngine

try:
    import miniaudio               # MP3/WAV/FLAC/OGG decode + playback
except Exception:                  # noqa: BLE001 — report nicely in the UI
    miniaudio = None

SR = 44100                         # everything is resampled to this on load
FFT_N = 2048
HOP = 1024                         # analysis frame hop (~43 fps at 44.1 kHz)
N_BANDS = ls.N_FULL_BANDS          # spectrogram bands (matches the effects)
BEAT_TAU = 0.22                    # beat envelope decay (s)

SHOW_EXT = ".show.json"
HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS_DIR = os.path.join(HERE, "tracks")   # drop songs here for the browser
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg")

# ---- Timeline geometry -------------------------------------------------------
GUTTER = 112                       # left gutter with track names (px)
RULER_H = 18
WAVE_H = 44
SPEC_H = 48
LANE_H = 26
TL_MIN_SPAN = 1.0                  # max zoom-in: 1 second across the view
EDGE_PX = 6                        # cue edge-grab tolerance
SNAP_S = 0.12                      # snap radius to a beat (s)

BG, PANEL_BG, CTL_BG = ls.BG, ls.PANEL_BG, ls.CTL_BG
FG, DIM, ACCENT, FONT = ls.FG, ls.DIM, ls.ACCENT, ls.FONT


# ---- Audio clip --------------------------------------------------------------
class AudioClip:
    """A decoded song: int16 stereo PCM for playback, float mono for analysis."""

    def __init__(self, path):
        if miniaudio is None:
            raise RuntimeError("miniaudio is not installed "
                               "(pip install miniaudio)")
        dec = miniaudio.decode_file(
            path, output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=2, sample_rate=SR)
        self.path = path
        self.pcm = np.asarray(dec.samples, dtype=np.int16).reshape(-1, 2)
        self.mono = self.pcm.astype(np.float32).mean(axis=1) / 32768.0
        self.sr = SR
        self.frames = len(self.mono)
        self.duration = self.frames / SR


# ---- Offline analysis --------------------------------------------------------
class Analysis:
    """Precomputed features for a clip: spectrogram, energy, beats, bpm.

    features_at(t) returns a dict shaped like light_studio.AUDIO, so the
    audio-reactive effects run off the song itself — deterministically, at any
    playhead position.
    """

    def __init__(self, clip):
        mono, sr = clip.mono, clip.sr
        if len(mono) < FFT_N:
            mono = np.pad(mono, (0, FFT_N - len(mono)))
        window = np.hanning(FFT_N).astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(mono, FFT_N)[::HOP]
        n = len(frames)
        edges = np.geomspace(50.0, 16000.0, N_BANDS + 1)
        bins = np.clip((edges / sr * FFT_N).astype(int), 1, FFT_N // 2)

        bands = np.zeros((n, N_BANDS), dtype=np.float32)
        rms = np.zeros(n, dtype=np.float32)
        for a in range(0, n, 2000):                 # chunked: bounds memory
            b = min(n, a + 2000)
            chunk = frames[a:b] * window
            mag = np.abs(np.fft.rfft(chunk, axis=1)).astype(np.float32)
            for i in range(N_BANDS):
                lo, hi = bins[i], max(bins[i] + 1, bins[i + 1])
                bands[a:b, i] = mag[:, lo:hi].mean(axis=1)
            rms[a:b] = np.sqrt((chunk ** 2).mean(axis=1))

        # Per-band auto-gain, then a soft knee into 0..1.
        ref = np.maximum(np.percentile(bands, 95, axis=0), 1e-6)
        self.bands = np.clip((bands / ref) ** 0.6, 0.0, 1.0)
        self.energy = np.clip(rms / max(np.percentile(rms, 95), 1e-6), 0.0, 1.0)
        wsum = self.bands.sum(axis=1)
        idx = np.arange(N_BANDS, dtype=np.float32)
        self.centroid = np.where(
            wsum > 1e-6, (self.bands * idx).sum(axis=1) /
            np.maximum(wsum, 1e-6) / (N_BANDS - 1), 0.5).astype(np.float32)

        self.n_frames = n
        self.hop_t = HOP / sr
        self.onsets = self._detect_onsets()
        self.bpm = self._estimate_bpm()

    def _detect_onsets(self):
        """Spectral-flux onsets with an adaptive threshold. Returns times."""
        flux = np.maximum(0.0, np.diff(self.bands, axis=0)).sum(axis=1)
        flux = np.concatenate([[0.0], flux])
        k = np.ones(43) / 43.0                      # ~1 s window
        mean = np.convolve(flux, k, mode="same")
        sq = np.convolve(flux ** 2, k, mode="same")
        std = np.sqrt(np.maximum(0.0, sq - mean ** 2))
        thr = mean + 1.3 * std + 1e-3
        gap = max(1, int(0.15 / self.hop_t))
        out, last = [], -gap
        for i in range(1, len(flux) - 1):
            if (flux[i] > thr[i] and flux[i] >= flux[i - 1]
                    and flux[i] > flux[i + 1] and i - last >= gap):
                out.append(i * self.hop_t)
                last = i
        return out

    def _estimate_bpm(self):
        iois = [b - a for a, b in zip(self.onsets, self.onsets[1:])
                if 0.25 <= b - a <= 2.0]
        if len(iois) < 3:
            return 0.0
        bpm = 60.0 / float(np.median(iois))
        while bpm < 70.0:
            bpm *= 2.0
        while bpm > 180.0:
            bpm /= 2.0
        return bpm

    def features_at(self, t):
        i = min(self.n_frames - 1, max(0, int(t / self.hop_t)))
        n_beats = bisect.bisect_right(self.onsets, t)
        beat = 0.0
        if n_beats:
            beat = math.exp(-max(0.0, t - self.onsets[n_beats - 1]) / BEAT_TAU)
        spectrum = self.bands[i]
        coarse = spectrum.reshape(ls.N_COARSE_BANDS, -1).mean(axis=1)
        return {"on": True, "energy": float(self.energy[i]),
                "centroid": float(self.centroid[i]),
                "beat": min(1.0, beat), "beats": n_beats,
                "bands": [float(x) for x in coarse],
                "spectrum": [float(x) for x in spectrum],
                "bpm": self.bpm}

    def peaks(self, t0, t1, cols):
        """Waveform min/max envelope over [t0, t1] in `cols` columns."""
        raise NotImplementedError   # bound to a clip below


def wave_peaks(mono, sr, t0, t1, cols):
    """Min/max envelope of mono[t0:t1] in `cols` columns (for the timeline)."""
    lo = np.zeros(cols, dtype=np.float32)
    hi = np.zeros(cols, dtype=np.float32)
    bounds = np.linspace(t0, t1, cols + 1) * sr
    bounds = np.clip(bounds.astype(np.int64), 0, len(mono))
    for i in range(cols):
        a, b = bounds[i], bounds[i + 1]
        if b > a:
            seg = mono[a:b]
            lo[i], hi[i] = seg.min(), seg.max()
    return lo, hi


# ---- Playback ----------------------------------------------------------------
class Player:
    """Streams a clip to the default output; tracks position sample-accurately.

    The device runs continuously; when paused the generator yields silence, so
    play/pause/seek are just flag/pointer updates (safe under the GIL).
    """

    def __init__(self, clip):
        self.clip = clip
        self.frame = 0
        self.playing = False
        self.device = None
        self.error = None
        if miniaudio is None:
            self.error = "miniaudio not installed"
            return
        try:
            self.device = miniaudio.PlaybackDevice(
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=2, sample_rate=clip.sr, buffersize_msec=100)
            gen = self._stream()
            next(gen)
            self.device.start(gen)
        except Exception as e:       # noqa: BLE001 — no output device etc.
            self.error = str(e)
            self.device = None

    def _stream(self):
        required = yield b""
        while True:
            if self.playing:
                a = self.frame
                b = min(a + required, self.clip.frames)
                chunk = self.clip.pcm[a:b].reshape(-1)
                self.frame = b
                if b >= self.clip.frames:
                    self.playing = False
                if len(chunk) < required * 2:
                    chunk = np.concatenate(
                        [chunk, np.zeros(required * 2 - len(chunk),
                                         dtype=np.int16)])
                required = yield chunk
            else:
                required = yield np.zeros(required * 2, dtype=np.int16)

    def pos(self):
        return self.frame / self.clip.sr

    def seek(self, t):
        self.frame = int(max(0.0, min(self.clip.duration, t)) * self.clip.sr)

    def close(self):
        self.playing = False
        if self.device:
            try:
                self.device.close()
            except Exception:        # noqa: BLE001
                pass
            self.device = None


# ---- Cues --------------------------------------------------------------------
class Cue:
    """A choreography step: effect + look, active over [start, end)."""

    def __init__(self, start, end, effect="solid", color=(255, 60, 40),
                 speed=1.0, brightness=1.0):
        self.start, self.end = float(start), float(end)
        self.effect = effect
        self.color = tuple(color)
        self.speed = float(speed)
        self.brightness = float(brightness)

    @property
    def dur(self):
        return self.end - self.start

    def to_dict(self):
        return {"start": self.start, "end": self.end, "effect": self.effect,
                "color": list(self.color), "speed": self.speed,
                "brightness": self.brightness}

    @classmethod
    def from_dict(cls, d):
        return cls(d["start"], d["end"], d.get("effect", "solid"),
                   d.get("color", (255, 60, 40)), d.get("speed", 1.0),
                   d.get("brightness", 1.0))


def active_cue(cues, t):
    for cue in cues:
        if cue.start <= t < cue.end:
            return cue
    return None


def apply_show(lights, t):
    """Drive every light from its cue track at song time t (ls.AUDIO is
    assumed to already hold the analysis features for t)."""
    for light in lights:
        cue = active_cue(light.cues, t)
        if cue is None:
            light.colors = [(0, 0, 0)] * light.segments
            continue
        light.effect = cue.effect
        light.color = cue.color
        light.speed = cue.speed
        light.brightness = cue.brightness
        light.phase = 0.0            # cues restart their effect at cue.start
        light.animate(t - cue.start)


# ---- Studio app ---------------------------------------------------------------
class ShowApp:
    def __init__(self, root, sync_fps=20, iface=None, audio_path=None):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.iface = iface
        root.title("Show Studio — choreograph lights to music")
        root.configure(bg=BG)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.minsize(900, 560)
        root.bind("<F11>", lambda _e: root.attributes(
            "-fullscreen", not root.attributes("-fullscreen")))
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
        root.bind("<Key>", self._on_key)

        self.clip = None
        self.analysis = None
        self.player = None
        self.t = 0.0                 # playhead when no player exists
        self.duration = 60.0         # editable timeline even before audio
        self.view_t0, self.view_t1 = 0.0, 60.0

        self.lights = []             # ls.Light, each with a .cues list
        self.selected = None         # selected light
        self.sel_cue = None          # selected cue (belongs to self.selected)
        self.cue_defaults = dict(effect="solid", color=(255, 60, 40),
                                 speed=1.0, brightness=1.0)
        self.snap_var = None         # built in _build_ui

        self.stage_w, self.stage_h = 600, 360
        self._drag = None            # stage drag: (light, dx, dy)
        self._tl_action = None       # timeline drag state
        self._strip_cache = None     # (t0, t1, width) of cached strip art
        self._wave_pts = None
        self._spec_img = None
        self._beat_xs = []
        self._loading_panel = False
        self.devices = []
        self._scan_results = None
        self._scanning = False

        self.sync = ls.SyncEngine(sync_fps)
        self.sync.start()

        self._build_ui()
        if audio_path:
            self.root.after(50, lambda: self.load_audio(audio_path))
        self._last_tick = time.monotonic()
        self._tick()

    # ---- UI construction ----
    def _build_ui(self):
        tk = self.tk
        main = tk.Frame(self.root, bg=BG)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(main, bg=BG, highlightthickness=0,
                                cursor="hand2")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_stage_click)
        self.canvas.bind("<B1-Motion>", self._on_stage_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(
            self, "_drag", None))
        self.canvas.bind("<Configure>", self._on_stage_resize)

        self.tl = tk.Canvas(main, bg="#0d0d13", highlightthickness=0,
                            height=self._tl_height())
        self.tl.grid(row=1, column=0, sticky="ew")
        self.tl.bind("<Button-1>", self._tl_press)
        self.tl.bind("<B1-Motion>", self._tl_motion)
        self.tl.bind("<ButtonRelease-1>", self._tl_release)
        self.tl.bind("<MouseWheel>", self._tl_wheel)
        self.tl.bind("<Configure>", lambda _e: self._invalidate_strip())

        panel = tk.Frame(self.root, width=300, bg=PANEL_BG)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)

        def section(text, pady=(10, 3)):
            tk.Label(panel, text=text, bg=PANEL_BG, fg=DIM,
                     font=("Segoe UI", 9, "bold")).pack(
                anchor="w", padx=12, pady=pady)

        def button(parent, text, cmd, **kw):
            return tk.Button(parent, text=text, command=cmd, bg=CTL_BG, fg=FG,
                             activebackground="#2a2a38", activeforeground=FG,
                             bd=0, font=FONT, padx=8, pady=3, **kw)

        # -- Show / audio --
        section("SHOW", pady=(12, 3))
        row = tk.Frame(panel, bg=PANEL_BG)
        row.pack(fill="x", padx=12)
        button(row, "Load Audio", self._pick_audio).pack(side="left",
                                                         padx=(0, 6))
        button(row, "Save", self.save_show).pack(side="left", padx=(0, 6))
        button(row, "Load", self.load_show_dialog).pack(side="left")
        self.song_label = tk.Label(panel, text="(no audio loaded)",
                                   bg=PANEL_BG, fg=DIM, font=("Segoe UI", 9),
                                   wraplength=276, justify="left")
        self.song_label.pack(anchor="w", padx=12, pady=(4, 0))

        # -- Track browser (./tracks folder) --
        thead = tk.Frame(panel, bg=PANEL_BG)
        thead.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(thead, text="TRACKS", bg=PANEL_BG, fg=DIM,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        tk.Button(thead, text="↻", command=self._refresh_tracks, bg=PANEL_BG,
                  fg=DIM, activebackground=PANEL_BG, activeforeground=FG,
                  bd=0, font=("Segoe UI", 9)).pack(side="left", padx=(6, 0))
        tframe = tk.Frame(panel, bg=PANEL_BG)
        tframe.pack(fill="x", padx=12)
        tsb = tk.Scrollbar(tframe, orient="vertical")
        self.track_list = tk.Listbox(
            tframe, height=5, bg=CTL_BG, fg=FG, bd=0, highlightthickness=0,
            selectbackground="#2a4a5e", selectforeground="#e8e8f0",
            activestyle="none", font=("Segoe UI", 9),
            yscrollcommand=tsb.set)
        tsb.configure(command=self.track_list.yview)
        self.track_list.pack(side="left", fill="x", expand=True)
        tsb.pack(side="left", fill="y")
        self.track_list.bind("<Double-Button-1>", self._on_track_open)
        self._track_paths = []
        self._refresh_tracks()

        # -- Transport --
        section("TRANSPORT")
        trow = tk.Frame(panel, bg=PANEL_BG)
        trow.pack(fill="x", padx=12)
        self.play_btn = button(trow, "▶ Play", self.toggle_play, width=7)
        self.play_btn.pack(side="left", padx=(0, 6))
        button(trow, "⏮", lambda: self.seek(0.0)).pack(side="left",
                                                        padx=(0, 6))
        self.time_label = tk.Label(trow, text="0:00.0 / 0:00.0", bg=PANEL_BG,
                                   fg=FG, font=("Consolas", 10))
        self.time_label.pack(side="left")
        self.snap_var = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text="Snap to beats", variable=self.snap_var,
                       bg=PANEL_BG, fg=FG, selectcolor=CTL_BG,
                       activebackground=PANEL_BG, activeforeground=FG,
                       font=FONT).pack(anchor="w", padx=12)

        # -- Lights --
        section("LIGHTS")
        lrow = tk.Frame(panel, bg=PANEL_BG)
        lrow.pack(fill="x", padx=12)
        button(lrow, "+ Bulb", lambda: self.add_light("bulb")).pack(
            side="left", padx=(0, 5))
        button(lrow, "+ Strip", lambda: self.add_light("strip")).pack(
            side="left", padx=(0, 5))
        button(lrow, "+ Bars", lambda: self.add_light("bars")).pack(
            side="left", padx=(0, 5))
        button(lrow, "Del", self.delete_light).pack(side="left")

        lgrid = tk.Frame(panel, bg=PANEL_BG)
        lgrid.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(lgrid, text="Segments", bg=PANEL_BG, fg=DIM,
                 font=FONT).grid(row=0, column=0, sticky="w")
        scale_kw = dict(orient="horizontal", bg=PANEL_BG, fg=FG,
                        troughcolor=CTL_BG, highlightthickness=0, bd=0,
                        length=150, font=("Segoe UI", 8), sliderrelief="flat")
        self.seg_scale = tk.Scale(lgrid, from_=2, to=40, resolution=1,
                                  command=self._on_segments, **scale_kw)
        self.seg_scale.set(12)
        self.seg_scale.grid(row=0, column=1, sticky="w", padx=(8, 0))

        # -- Cue editing --
        section("SELECTED CUE")
        crow = tk.Frame(panel, bg=PANEL_BG)
        crow.pack(fill="x", padx=12)
        button(crow, "+ Cue @ playhead", self.add_cue).pack(side="left",
                                                            padx=(0, 6))
        button(crow, "Del Cue", self.delete_cue).pack(side="left")

        cgrid = tk.Frame(panel, bg=PANEL_BG)
        cgrid.pack(fill="x", padx=12, pady=(4, 0))

        def cue_label(r, text):
            tk.Label(cgrid, text=text, bg=PANEL_BG, fg=DIM, font=FONT).grid(
                row=r, column=0, sticky="w", pady=1)

        cue_label(0, "Effect")
        self.effect_var = tk.StringVar(value="solid")
        om = tk.OptionMenu(cgrid, self.effect_var, *ls.EFFECTS,
                           command=self._on_cue_effect)
        om.configure(bg=CTL_BG, fg=FG, activebackground="#2a2a38",
                     activeforeground=FG, highlightthickness=0, bd=0,
                     font=FONT, width=10)
        om["menu"].configure(bg=CTL_BG, fg=FG, activebackground="#2a2a38")
        om.grid(row=0, column=1, sticky="w", padx=(8, 0))

        cue_label(1, "Color")
        self.color_btn = tk.Button(cgrid, text="  pick  ", bg="#ff3c28",
                                   fg="#000", bd=0, font=FONT,
                                   command=self._on_cue_color)
        self.color_btn.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=1)

        cue_label(2, "Speed")
        self.speed_scale = tk.Scale(cgrid, from_=0.1, to=3.0, resolution=0.05,
                                    command=self._on_cue_speed, **scale_kw)
        self.speed_scale.set(1.0)
        self.speed_scale.grid(row=2, column=1, sticky="w", padx=(8, 0))

        cue_label(3, "Brightness")
        self.bright_scale = tk.Scale(cgrid, from_=0.05, to=1.0,
                                     resolution=0.05,
                                     command=self._on_cue_bright, **scale_kw)
        self.bright_scale.set(1.0)
        self.bright_scale.grid(row=3, column=1, sticky="w", padx=(8, 0))

        # -- Govee sync --
        section("GOVEE SYNC")
        srow = tk.Frame(panel, bg=PANEL_BG)
        srow.pack(fill="x", padx=12)
        self.scan_btn = button(srow, "Scan LAN", self.scan_devices)
        self.scan_btn.pack(side="left", padx=(0, 8))
        self.sync_var = tk.BooleanVar(value=False)
        tk.Checkbutton(srow, text="Sync ON", variable=self.sync_var,
                       command=self._on_sync_toggle, bg=PANEL_BG, fg=FG,
                       selectcolor=CTL_BG, activebackground=PANEL_BG,
                       activeforeground=FG, font=FONT).pack(side="left")
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
        irow.pack(fill="x", padx=12, pady=(4, 0))
        self.ip_entry = tk.Entry(irow, bg=CTL_BG, fg=FG, bd=0,
                                 insertbackground=FG, font=FONT, width=14)
        self.ip_entry.pack(side="left", ipady=3)
        button(irow, "Assign IP", self._on_assign_ip).pack(side="left",
                                                           padx=(8, 0))

        self.status = tk.Label(panel, text="", bg=PANEL_BG, fg=DIM,
                               font=("Segoe UI", 9), justify="left",
                               wraplength=276)
        self.status.pack(anchor="w", padx=12, pady=(10, 0))
        self._set_status("Load an MP3, add lights, then drag on a light's "
                         "track to paint cues. Space = play.")

    def _set_status(self, text):
        self.status.configure(text=text)

    # ---- Playhead / transport ----
    def playhead(self):
        return self.player.pos() if self.player else self.t

    def seek(self, t):
        t = max(0.0, min(self.duration, t))
        if self.player:
            self.player.seek(t)
        self.t = t

    def toggle_play(self):
        if not self.player:
            self._set_status("Load an audio file first.")
            return
        if self.player.error:
            self._set_status(f"No audio output: {self.player.error}")
            return
        if not self.player.playing and self.player.pos() >= self.clip.duration:
            self.player.seek(0.0)
        self.player.playing = not self.player.playing
        self.play_btn.configure(
            text="❚❚ Pause" if self.player.playing else "▶ Play")

    def _on_key(self, ev):
        if isinstance(self.root.focus_get(), self.tk.Entry):
            return
        if ev.keysym == "space":
            self.toggle_play()
        elif ev.keysym == "Home":
            self.seek(0.0)
        elif ev.keysym in ("Left", "Right"):
            step = 0.1 if ev.state & 0x1 else 1.0
            self.seek(self.playhead() +
                      (step if ev.keysym == "Right" else -step))
        elif ev.keysym == "Delete":
            if self.sel_cue is not None:
                self.delete_cue()
            elif self.selected is not None:
                self.delete_light()

    # ---- Track browser ----
    def _refresh_tracks(self):
        os.makedirs(TRACKS_DIR, exist_ok=True)
        try:
            names = sorted((n for n in os.listdir(TRACKS_DIR)
                            if n.lower().endswith(AUDIO_EXTS)),
                           key=str.lower)
        except OSError:
            names = []
        self._track_paths = [os.path.join(TRACKS_DIR, n) for n in names]
        self.track_list.delete(0, "end")
        if names:
            for n in names:
                self.track_list.insert("end", " " + n)
        else:
            self.track_list.insert("end", " (drop songs into ./tracks)")
        self._mark_loaded_track()

    def _mark_loaded_track(self):
        """Select the currently loaded song in the track list."""
        if not self.clip:
            return
        cur = os.path.normcase(os.path.abspath(self.clip.path))
        for i, p in enumerate(self._track_paths):
            if os.path.normcase(os.path.abspath(p)) == cur:
                self.track_list.selection_clear(0, "end")
                self.track_list.selection_set(i)
                self.track_list.see(i)
                break

    def _on_track_open(self, _ev):
        sel = self.track_list.curselection()
        if sel and sel[0] < len(self._track_paths):
            self.load_audio(self._track_paths[sel[0]])

    # ---- Audio loading ----
    def _pick_audio(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Open audio",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg"),
                       ("All files", "*.*")])
        if path:
            self.load_audio(path)

    def load_audio(self, path):
        if miniaudio is None:
            self._set_status("miniaudio missing — pip install miniaudio")
            return
        self._set_status(f"Decoding {os.path.basename(path)}...")
        self.root.update_idletasks()
        try:
            clip = AudioClip(path)
        except Exception as e:       # noqa: BLE001 — bad file, codec, etc.
            self._set_status(f"Could not load audio: {e}")
            return
        if self.player:
            self.player.close()
        self.clip = clip
        self._set_status("Analyzing (spectrum + beats)...")
        self.root.update_idletasks()
        self.analysis = Analysis(clip)
        self.player = Player(clip)
        self.duration = clip.duration
        self.view_t0, self.view_t1 = 0.0, clip.duration
        self._invalidate_strip()
        self.seek(0.0)
        self.play_btn.configure(text="▶ Play")
        name = os.path.basename(path)
        bpm = self.analysis.bpm
        beats = len(self.analysis.onsets)
        self.song_label.configure(
            text=f"{name} — {fmt_time(clip.duration)}, "
                 f"~{bpm:.0f} bpm, {beats} beats")
        msg = "Loaded. Drag on a track to paint a cue."
        if self.player.error:
            msg += f" (playback unavailable: {self.player.error})"
        self._set_status(msg)
        self._mark_loaded_track()

    # ---- Lights ----
    def add_light(self, kind):
        light = ls.Light(kind, self.stage_w // 2, self.stage_h // 2)
        light.cues = []
        self.lights.append(light)
        self.select(light)
        self._layout_timeline()

    def delete_light(self):
        if self.selected in self.lights:
            self.lights.remove(self.selected)
        self.select(None)
        self._layout_timeline()

    def select(self, light, cue=None):
        self.selected = light
        self.sel_cue = cue
        self._refresh_panel()

    def _refresh_panel(self):
        self._loading_panel = True
        try:
            light = self.selected
            if light and light.kind in ("strip", "bars"):
                self.seg_scale.configure(state="normal", fg=FG)
                self.seg_scale.set(light.segments)
            else:
                self.seg_scale.configure(state="disabled", fg=DIM)
            self.device_var.set(light.device_ip if light and light.device_ip
                                else "(none)")
            cue = self.sel_cue
            src = cue if cue else self.cue_defaults
            get = (lambda k: getattr(src, k)) if cue else src.__getitem__
            self.effect_var.set(get("effect"))
            r, g, b = get("color")
            self.color_btn.configure(bg=f"#{r:02x}{g:02x}{b:02x}")
            self.speed_scale.set(get("speed"))
            self.bright_scale.set(get("brightness"))
        finally:
            self._loading_panel = False

    def _on_segments(self, v):
        if self._loading_panel or not self.selected:
            return
        self.selected.set_segments(int(float(v)))

    # ---- Cue editing ----
    def _snap(self, t):
        if not (self.snap_var.get() and self.analysis
                and self.analysis.onsets):
            return t
        i = bisect.bisect_left(self.analysis.onsets, t)
        best = t
        for j in (i - 1, i):
            if 0 <= j < len(self.analysis.onsets):
                o = self.analysis.onsets[j]
                if abs(o - t) < min(SNAP_S, abs(best - t) + 1e-9):
                    best = o
        return best

    def _cue_from_defaults(self, start, end):
        d = self.cue_defaults
        return Cue(start, end, d["effect"], d["color"], d["speed"],
                   d["brightness"])

    def _insert_cue(self, light, cue):
        light.cues.append(cue)
        light.cues.sort(key=lambda c: c.start)

    def _neighbors(self, light, cue):
        """(prev_end, next_start) bounds for moving/resizing this cue."""
        lo, hi = 0.0, self.duration
        for other in light.cues:
            if other is cue:
                continue
            if other.end <= cue.start + 1e-9:
                lo = max(lo, other.end)
            elif other.start >= cue.end - 1e-9:
                hi = min(hi, other.start)
        return lo, hi

    def add_cue(self):
        if not self.selected:
            self._set_status("Select a light first (click it on the stage).")
            return
        light = self.selected
        start = self._snap(self.playhead())
        hi = self.duration
        for other in sorted(light.cues, key=lambda c: c.start):
            if other.start > start + 1e-9:
                hi = min(hi, other.start)
                break
            if other.start <= start < other.end:
                self._set_status("A cue already covers the playhead here.")
                return
        end = min(start + 4.0, hi)
        if end - start < 0.05:
            self._set_status("No room for a cue at the playhead.")
            return
        cue = self._cue_from_defaults(start, end)
        self._insert_cue(light, cue)
        self.select(light, cue)

    def delete_cue(self):
        if self.selected and self.sel_cue in self.selected.cues:
            self.selected.cues.remove(self.sel_cue)
        self.select(self.selected)

    def _remember_defaults(self, cue):
        self.cue_defaults = dict(effect=cue.effect, color=cue.color,
                                 speed=cue.speed, brightness=cue.brightness)

    def _on_cue_effect(self, name):
        if self._loading_panel:
            return
        if self.sel_cue:
            self.sel_cue.effect = name
            self._remember_defaults(self.sel_cue)
        else:
            self.cue_defaults["effect"] = name

    def _on_cue_color(self):
        from tkinter import colorchooser
        rgb, _hex = colorchooser.askcolor(color=self.color_btn["bg"],
                                          title="Cue color")
        if not rgb:
            return
        color = tuple(int(c) for c in rgb)
        if self.sel_cue:
            self.sel_cue.color = color
            self._remember_defaults(self.sel_cue)
        else:
            self.cue_defaults["color"] = color
        self._refresh_panel()

    def _on_cue_speed(self, v):
        if self._loading_panel:
            return
        if self.sel_cue:
            self.sel_cue.speed = float(v)
            self._remember_defaults(self.sel_cue)
        else:
            self.cue_defaults["speed"] = float(v)

    def _on_cue_bright(self, v):
        if self._loading_panel:
            return
        if self.sel_cue:
            self.sel_cue.brightness = float(v)
            self._remember_defaults(self.sel_cue)
        else:
            self.cue_defaults["brightness"] = float(v)

    # ---- Timeline geometry ----
    def _tl_height(self):
        return (RULER_H + WAVE_H + SPEC_H + 6
                + max(1, len(self.lights)) * LANE_H + 6)

    def _layout_timeline(self):
        self.tl.configure(height=self._tl_height())

    def _lanes_y0(self):
        return RULER_H + WAVE_H + SPEC_H + 6

    def _tl_width(self):
        return max(50, self.tl.winfo_width() - GUTTER)

    def t2x(self, t):
        span = max(1e-6, self.view_t1 - self.view_t0)
        return GUTTER + (t - self.view_t0) / span * self._tl_width()

    def x2t(self, x):
        span = self.view_t1 - self.view_t0
        return self.view_t0 + (x - GUTTER) / self._tl_width() * span

    def _lane_at(self, y):
        i = int((y - self._lanes_y0()) // LANE_H)
        return i if 0 <= i < len(self.lights) else None

    # ---- Timeline interaction ----
    def _tl_press(self, ev):
        lane = self._lane_at(ev.y)
        if lane is None:
            if RULER_H + WAVE_H + SPEC_H >= ev.y >= 0 and ev.x >= GUTTER:
                self._tl_action = ("seek",)
                self.seek(self.x2t(ev.x))
            return
        light = self.lights[lane]
        if ev.x < GUTTER:            # gutter click: just select the light
            self.select(light)
            return
        t = self.x2t(ev.x)
        for cue in light.cues:
            x0, x1 = self.t2x(cue.start), self.t2x(cue.end)
            if x0 - EDGE_PX <= ev.x <= x1 + EDGE_PX:
                self.select(light, cue)
                if abs(ev.x - x0) <= EDGE_PX:
                    self._tl_action = ("resize_l", light, cue)
                elif abs(ev.x - x1) <= EDGE_PX:
                    self._tl_action = ("resize_r", light, cue)
                else:
                    self._tl_action = ("move", light, cue, t - cue.start)
                return
        self.select(light)
        self._tl_action = ("create", light, self._snap(t), self._snap(t))

    def _tl_motion(self, ev):
        act = self._tl_action
        if not act:
            return
        t = self.x2t(ev.x)
        if act[0] == "seek":
            self.seek(t)
        elif act[0] == "create":
            self._tl_action = (*act[:3], self._snap(
                max(0.0, min(self.duration, t))))
        elif act[0] == "move":
            _, light, cue, grab = act
            lo, hi = self._neighbors(light, cue)
            start = self._snap(t - grab)
            start = max(lo, min(hi - cue.dur, start))
            cue.end = start + cue.dur
            cue.start = start
        elif act[0] == "resize_l":
            _, light, cue = act
            lo, _hi = self._neighbors(light, cue)
            cue.start = max(lo, min(cue.end - 0.05, self._snap(t)))
        elif act[0] == "resize_r":
            _, light, cue = act
            _lo, hi = self._neighbors(light, cue)
            cue.end = min(hi, max(cue.start + 0.05, self._snap(t)))

    def _tl_release(self, _ev):
        act, self._tl_action = self._tl_action, None
        if act and act[0] == "create":
            _, light, a, b = act
            a, b = min(a, b), max(a, b)
            b = min(b, self.duration)
            a = max(0.0, a)
            for other in light.cues:  # reject overlaps
                if other.start < b and a < other.end:
                    return
            if b - a >= 0.05:
                cue = self._cue_from_defaults(a, b)
                self._insert_cue(light, cue)
                self.select(light, cue)

    def _tl_wheel(self, ev):
        if self.duration <= 0:
            return
        steps = ev.delta / 120.0
        span = self.view_t1 - self.view_t0
        if ev.state & 0x4:           # Ctrl: zoom around the cursor
            t = self.x2t(ev.x)
            factor = 1.25 ** steps
            new_span = max(TL_MIN_SPAN, min(self.duration, span / factor))
            frac = (t - self.view_t0) / span
            self.view_t0 = t - frac * new_span
            self.view_t1 = self.view_t0 + new_span
        else:                        # wheel: pan
            shift = -steps * span * 0.12
            self.view_t0 += shift
            self.view_t1 += shift
        self._clamp_view()
        self._invalidate_strip()

    def _clamp_view(self):
        span = self.view_t1 - self.view_t0
        span = max(TL_MIN_SPAN, min(self.duration if self.duration > 0
                                    else span, span))
        self.view_t0 = max(0.0, min(self.view_t0,
                                    max(0.0, self.duration - span)))
        self.view_t1 = self.view_t0 + span

    # ---- Timeline art (waveform + spectrogram, cached per view) ----
    def _invalidate_strip(self):
        self._strip_cache = None

    def _rebuild_strip(self):
        cols = int(self._tl_width())
        key = (round(self.view_t0, 4), round(self.view_t1, 4), cols)
        if self._strip_cache == key or cols < 10:
            return
        self._strip_cache = key
        self._wave_pts = None
        self._spec_img = None
        self._beat_xs = []
        if not self.clip:
            return
        t0, t1 = self.view_t0, self.view_t1
        lo, hi = wave_peaks(self.clip.mono, self.clip.sr, t0, t1, cols)
        y_mid = RULER_H + WAVE_H / 2
        amp = (WAVE_H / 2 - 3)
        pts = []
        for i in range(cols):
            pts += [GUTTER + i, y_mid - float(hi[i]) * amp]
        for i in range(cols - 1, -1, -1):
            pts += [GUTTER + i, y_mid - float(lo[i]) * amp]
        self._wave_pts = pts

        if self.analysis:
            an = self.analysis
            # Color LUT: 32 intensity levels per band row.
            tkimg = self.tk.PhotoImage(width=cols, height=N_BANDS)
            frame_idx = np.clip(
                ((t0 + (np.arange(cols) + 0.5) / cols * (t1 - t0))
                 / an.hop_t).astype(int), 0, an.n_frames - 1)
            grid = an.bands[frame_idx]           # (cols, N_BANDS)
            q = np.clip((grid * 31).astype(int), 0, 31)
            lut = []
            for band in range(N_BANDS):
                hue = band / N_BANDS * 0.66
                row = []
                for level in range(32):
                    r, g, b = colorsys.hsv_to_rgb(
                        hue, 0.85, 0.10 + 0.90 * (level / 31) ** 0.8)
                    row.append(f"#{int(r*255):02x}{int(g*255):02x}"
                               f"{int(b*255):02x}")
                lut.append(row)
            rows = []
            for y in range(N_BANDS):             # top row = treble
                band = N_BANDS - 1 - y
                lrow = lut[band]
                rows.append("{" + " ".join(lrow[q[x, band]]
                                           for x in range(cols)) + "}")
            tkimg.put(" ".join(rows), to=(0, 0))
            self._spec_img = tkimg
            self._beat_xs = [self.t2x(o) for o in an.onsets
                             if t0 <= o <= t1]

    # ---- Timeline drawing ----
    def _ruler_step(self):
        span = self.view_t1 - self.view_t0
        px_per_s = self._tl_width() / max(1e-6, span)
        for step in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
            if step * px_per_s >= 60:
                return step
        return 600

    def _redraw_timeline(self):
        c = self.tl
        c.delete("all")
        w = c.winfo_width()
        lanes_y0 = self._lanes_y0()
        strip_y1 = RULER_H + WAVE_H + SPEC_H

        # Waveform + spectrogram
        if self._spec_img is not None:
            c.create_image(GUTTER, RULER_H + WAVE_H, anchor="nw",
                           image=self._spec_img)
        if self._wave_pts:
            c.create_polygon(*self._wave_pts, fill="#33556e", outline="")
        for x in self._beat_xs:
            c.create_line(x, RULER_H, x, RULER_H + 6, fill="#5ad1ff", width=1)
            c.create_line(x, RULER_H + 6, x, strip_y1, fill="#1d3a4d",
                          width=1)
        if not self.clip:
            c.create_text(GUTTER + self._tl_width() / 2,
                          RULER_H + (WAVE_H + SPEC_H) / 2,
                          text="Load Audio to see the song here",
                          fill=DIM, font=("Segoe UI", 11))

        # Ruler
        step = self._ruler_step()
        t = math.floor(self.view_t0 / step) * step
        while t <= self.view_t1 + 1e-9:
            x = self.t2x(t)
            if x >= GUTTER - 1:
                c.create_line(x, 0, x, RULER_H, fill="#2a2a38")
                label = fmt_time(t) if step >= 1 else f"{t:.1f}"
                c.create_text(x + 3, 3, anchor="nw", text=label,
                              fill=DIM, font=("Segoe UI", 7))
            t += step

        # Lanes + cues
        for i, light in enumerate(self.lights):
            y0 = lanes_y0 + i * LANE_H
            if light is self.selected:
                c.create_rectangle(0, y0, w, y0 + LANE_H, fill="#14141d",
                                   outline="")
            c.create_line(GUTTER, y0 + LANE_H, w, y0 + LANE_H,
                          fill="#1a1a24")
            name = light.name if len(light.name) <= 14 else \
                light.name[:13] + "…"
            dot = "● " if light.device_ip else ""
            c.create_text(8, y0 + LANE_H / 2, anchor="w", text=dot + name,
                          fill=FG if light is self.selected else DIM,
                          font=("Segoe UI", 9))
            for cue in light.cues:
                x0 = max(GUTTER, self.t2x(cue.start))
                x1 = min(w, self.t2x(cue.end))
                if x1 <= GUTTER or x0 >= w:
                    continue
                sel = cue is self.sel_cue
                c.create_rectangle(
                    x0, y0 + 3, x1, y0 + LANE_H - 3,
                    fill=blend(cue.color, 0.45), outline=ACCENT if sel
                    else blend(cue.color, 0.8), width=2 if sel else 1)
                if x1 - x0 > 44:
                    c.create_text((x0 + x1) / 2, y0 + LANE_H / 2,
                                  text=cue.effect, fill="#e8e8f0",
                                  font=("Segoe UI", 8))
        # In-progress cue creation preview
        if self._tl_action and self._tl_action[0] == "create":
            _, light, a, b = self._tl_action
            if abs(b - a) > 0.01 and light in self.lights:
                i = self.lights.index(light)
                y0 = lanes_y0 + i * LANE_H
                c.create_rectangle(self.t2x(min(a, b)), y0 + 3,
                                   self.t2x(max(a, b)), y0 + LANE_H - 3,
                                   outline=ACCENT, dash=(3, 2))

        # Gutter mask + playhead
        c.create_rectangle(0, 0, GUTTER, RULER_H + WAVE_H + SPEC_H,
                           fill="#0d0d13", outline="")
        c.create_text(8, RULER_H + 8, anchor="w", text="song",
                      fill=DIM, font=("Segoe UI", 8))
        x = self.t2x(self.playhead())
        if x >= GUTTER:
            c.create_line(x, 0, x, self._tl_height(), fill=ACCENT, width=1)
            c.create_polygon(x - 5, 0, x + 5, 0, x, 8, fill=ACCENT,
                             outline="")

    # ---- Stage ----
    def _on_stage_resize(self, ev):
        self.stage_w, self.stage_h = ev.width, ev.height
        for light in self.lights:
            light.x = min(light.x, self.stage_w - 20)
            light.y = min(light.y, self.stage_h - 20)

    def _light_at(self, x, y):
        for light in reversed(self.lights):
            x0, y0, x1, y1 = light.bbox()
            if x0 <= x <= x1 and y0 <= y <= y1:
                return light
        return None

    def _on_stage_click(self, ev):
        light = self._light_at(ev.x, ev.y)
        self.select(light)
        if light:
            self._drag = (light, ev.x - light.x, ev.y - light.y)

    def _on_stage_drag(self, ev):
        if not self._drag:
            return
        light, dx, dy = self._drag
        light.x = max(20, min(self.stage_w - 20, ev.x - dx))
        light.y = max(20, min(self.stage_h - 20, ev.y - dy))

    def _draw_light(self, light):
        c = self.canvas
        outline = "#20202c"          # keeps dark lights visible on the stage
        if light.kind == "bulb":
            col = light.colors[0]
            x, y = light.x, light.y
            for rr, k in ((ls.BULB_R * 2.3, 0.10), (ls.BULB_R * 1.75, 0.22),
                          (ls.BULB_R * 1.3, 0.45)):
                if col != (0, 0, 0):
                    c.create_oval(x - rr, y - rr, x + rr, y + rr,
                                  fill=blend(col, k), outline="")
            c.create_oval(x - ls.BULB_R, y - ls.BULB_R, x + ls.BULB_R,
                          y + ls.BULB_R, fill=blend(col, 1.0),
                          outline=outline)
            label_y = y + ls.BULB_R * 2.3 + 10
        elif light.kind == "bars":
            nper = light.segments // 2
            bh = nper * ls.BAR_SEG_H
            y0 = light.y - bh / 2
            tv = ls.BAR_GAP / 2 - ls.BAR_W
            c.create_rectangle(light.x - tv, y0, light.x + tv, y0 + bh,
                               outline="#232330")
            for side, cols in ((-1, light.colors[:nper]),
                               (+1, light.colors[nper:])):
                bx0 = light.x + side * ls.BAR_GAP / 2 - ls.BAR_W / 2
                avg = tuple(sum(cc[i] for cc in cols) // nper
                            for i in range(3))
                if avg != (0, 0, 0):
                    c.create_rectangle(bx0 - 8, y0 - 8, bx0 + ls.BAR_W + 8,
                                       y0 + bh + 8, fill=blend(avg, 0.12),
                                       outline="")
                for i, col in enumerate(cols):
                    sy = y0 + bh - (i + 1) * ls.BAR_SEG_H
                    c.create_rectangle(bx0, sy + 1, bx0 + ls.BAR_W,
                                       sy + ls.BAR_SEG_H - 1,
                                       fill=blend(col, 1.0), outline=outline)
            label_y = y0 + bh + 16
        else:
            n = light.segments
            w = n * ls.SEG_W
            x0 = light.x - w / 2
            y0 = light.y - ls.SEG_H / 2
            avg = light.avg_color()
            if avg != (0, 0, 0):
                c.create_rectangle(x0 - 8, y0 - 8, x0 + w + 8,
                                   y0 + ls.SEG_H + 8, fill=blend(avg, 0.12),
                                   outline="")
            for i, col in enumerate(light.colors):
                sx = x0 + i * ls.SEG_W
                c.create_rectangle(sx + 1, y0, sx + ls.SEG_W - 1,
                                   y0 + ls.SEG_H, fill=blend(col, 1.0),
                                   outline=outline)
            label_y = y0 + ls.SEG_H + 18

        tag = light.name
        if light.device_ip:
            tag += f"  → {light.device_ip}"
        c.create_text(light.x, label_y, text=tag, fill=DIM,
                      font=("Segoe UI", 9))
        if light is self.selected:
            x0, y0, x1, y1 = light.bbox()
            c.create_rectangle(x0, y0, x1, y1, outline=ACCENT, dash=(3, 3))

    def _redraw_stage(self):
        c = self.canvas
        c.delete("all")
        if not self.lights:
            c.create_text(self.stage_w / 2, self.stage_h / 2,
                          text="Add a light, then paint cues on its track "
                               "below", fill=DIM, font=("Segoe UI", 13))
        for light in self.lights:
            self._draw_light(light)
        playing = self.player.playing if self.player else False
        state = "▶ playing" if playing else "❚❚ paused"
        sync = "sync ON" if self.sync_var.get() else "sync off"
        bpm = f"~{self.analysis.bpm:.0f} bpm" if self.analysis else "no audio"
        c.create_text(12, 14, anchor="w", fill="#e8e8f0",
                      font=("Segoe UI", 11, "bold"),
                      text=f"Show Studio   {state}   {sync}   ♪ {bpm}   "
                           f"{len(self.lights)} light(s)")

    # ---- Govee sync (same flow as light_studio) ----
    def scan_devices(self):
        if self._scanning:
            return
        self._scanning = True
        self.scan_btn.configure(text="Scanning...", state="disabled")
        self._set_status("Scanning LAN for Govee devices (3s)...")

        def worker():
            try:
                found = ls.govee_lan.scan(timeout=3.0, iface=self.iface)
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
        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        for opt in ["(none)"] + [label for label, _ip in self.devices]:
            menu.add_command(label=opt,
                             command=lambda o=opt: self._on_device_pick(o))
        if self.devices:
            self._set_status(f"Found {len(self.devices)} device(s). Select a "
                             "light, then pick its device.")
        else:
            self._set_status("No devices found. Is LAN Control enabled in "
                             "the Govee Home app?")

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
                self.selected.device_ip = label

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
            self._set_status(f"Sync ON — {mapped} mapped light(s) stream "
                             "during playback/scrubbing.")
        else:
            self._set_status("Sync off.")

    # ---- Persistence ----
    def _default_show_path(self):
        if self.clip:
            stem = os.path.splitext(os.path.basename(self.clip.path))[0]
            return stem + SHOW_EXT
        return "my_show" + SHOW_EXT

    def save_show(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Save show", initialdir=HERE,
            initialfile=self._default_show_path(),
            defaultextension=".json",
            filetypes=[("Show files", "*" + SHOW_EXT), ("JSON", "*.json")])
        if not path:
            return
        data = {"audio": self.clip.path if self.clip else None,
                "lights": []}
        for light in self.lights:
            d = light.to_dict()
            d["cues"] = [cue.to_dict() for cue in light.cues]
            data["lights"].append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._set_status(f"Saved show to {os.path.basename(path)}")

    def load_show_dialog(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Load show", initialdir=HERE,
            filetypes=[("Show files", "*" + SHOW_EXT), ("JSON", "*.json"),
                       ("All files", "*.*")])
        if path:
            self.load_show(path)

    def load_show(self, path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            self._set_status(f"Could not load show: {e}")
            return
        self.lights = []
        for d in data.get("lights", []):
            light = ls.Light.from_dict(d)
            light.cues = [Cue.from_dict(cd) for cd in d.get("cues", [])]
            light.cues.sort(key=lambda c: c.start)
            self.lights.append(light)
        self.select(None)
        self._layout_timeline()
        audio = data.get("audio")
        if audio and os.path.exists(audio):
            self.load_audio(audio)
        else:
            ends = [c.end for l in self.lights for c in l.cues]
            self.duration = max(ends + [60.0])
            self.view_t0, self.view_t1 = 0.0, self.duration
            self._invalidate_strip()
            if audio:
                self._set_status(f"Show loaded, but audio file not found: "
                                 f"{audio}")
            else:
                self._set_status("Show loaded (no audio attached).")

    # ---- Main loop ----
    def _tick(self):
        t = self.playhead()
        self.t = t

        # Feed the song's precomputed features to the audio-reactive effects.
        if self.analysis:
            ls.AUDIO.update(self.analysis.features_at(t))
        elif ls.AUDIO["on"]:
            ls.AUDIO.update({"on": False, "energy": 0.0, "centroid": 0.0,
                             "beat": 0.0, "beats": 0,
                             "bands": [0.0] * ls.N_COARSE_BANDS,
                             "spectrum": [0.0] * ls.N_FULL_BANDS, "bpm": 0.0})

        apply_show(self.lights, t)
        self.sync.set_targets({l.device_ip: l.avg_color()
                               for l in self.lights if l.device_ip})

        # Keep the playhead in view while playing.
        playing = self.player.playing if self.player else False
        if playing and t > self.view_t1 - 0.02 * (self.view_t1 -
                                                  self.view_t0):
            span = self.view_t1 - self.view_t0
            self.view_t0 = min(max(0.0, self.duration - span),
                               t - 0.1 * span)
            self.view_t1 = self.view_t0 + span
            self._invalidate_strip()
        if self.player and not playing and \
                self.play_btn["text"].startswith("❚❚"):
            self.play_btn.configure(text="▶ Play")   # song reached the end

        self._rebuild_strip()
        self._redraw_stage()
        self._redraw_timeline()
        self.time_label.configure(
            text=f"{fmt_time(t)} / {fmt_time(self.duration)}")
        self.root.after(ls.TICK_MS, self._tick)

    def shutdown(self):
        if self.player:
            self.player.close()
        self.sync.set_enabled(False)
        self.sync.stop()


# ---- Helpers -----------------------------------------------------------------
def blend(color, k, bg=(10, 10, 15)):
    """Blend color toward the stage background by factor k (0..1)."""
    r = int(bg[0] + (color[0] - bg[0]) * k)
    g = int(bg[1] + (color[1] - bg[1]) * k)
    b = int(bg[2] + (color[2] - bg[2]) * k)
    return f"#{r:02x}{g:02x}{b:02x}"


def fmt_time(t):
    m = int(t // 60)
    return f"{m}:{t % 60:04.1f}"


# ---- Modes --------------------------------------------------------------------
def _make_test_wav(path, seconds=8.0):
    """Synthesize a little techno loop: kick every 0.5 s + tones."""
    import wave
    n = int(seconds * SR)
    t = np.arange(n) / SR
    kick_env = np.zeros(n, dtype=np.float32)
    for beat in np.arange(0.0, seconds, 0.5):
        i = int(beat * SR)
        j = min(n, i + int(0.12 * SR))
        kick_env[i:j] = np.exp(-np.arange(j - i) / (0.02 * SR))
    audio = (0.8 * kick_env * np.sin(2 * np.pi * 55 * t)
             + 0.15 * np.sin(2 * np.pi * 440 * t)
             + 0.08 * np.sin(2 * np.pi * 3000 * t))
    pcm = (np.clip(audio, -1, 1) * 32767 * 0.9).astype(np.int16)
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def selftest():
    """Headless: decode a synthesized song, analyze it, run the cue engine."""
    import tempfile
    print("Show Studio selftest:")
    if miniaudio is None:
        print("  FAILED: miniaudio not installed")
        return 1
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "test.wav")
        _make_test_wav(wav)
        clip = AudioClip(wav)
        print(f"  decode: {clip.duration:.2f}s @ {clip.sr} Hz, "
              f"{clip.frames} frames")
        if not 7.5 < clip.duration < 8.5:
            print("  FAILED: unexpected duration")
            ok = False
        an = Analysis(clip)
        print(f"  analysis: {an.n_frames} frames, {len(an.onsets)} onsets, "
              f"~{an.bpm:.0f} bpm")
        if len(an.onsets) < 8:
            print("  FAILED: expected >= 8 beat onsets (16 kicks synthed)")
            ok = False
        if not 60 <= an.bpm <= 200:
            print(f"  FAILED: bpm {an.bpm:.0f} out of range")
            ok = False
        feats = an.features_at(1.05)
        if not (0 <= feats["energy"] <= 1 and len(feats["spectrum"])
                == N_BANDS and feats["beats"] >= 1):
            print("  FAILED: bad features_at output")
            ok = False

        # Cue engine: two cues on a strip, gap after.
        light = ls.Light("strip", 0, 0, segments=8)
        light.cues = [Cue(0.0, 2.0, "solid", (255, 0, 0)),
                      Cue(2.0, 4.0, "spectrum", (0, 128, 255))]
        ls.AUDIO.update(an.features_at(3.0))
        for t, want_on in ((1.0, True), (3.0, True), (5.0, False)):
            apply_show([light], t)
            lit = any(c != (0, 0, 0) for c in light.colors)
            rng = all(0 <= v <= 255 for c in light.colors for v in c)
            print(f"  cues @ t={t}: lit={lit} in-range={rng}")
            if lit != want_on or not rng:
                print("  FAILED: cue engine wrong state")
                ok = False
        if active_cue(light.cues, 2.0).effect != "spectrum":
            print("  FAILED: cue boundary should belong to the later cue")
            ok = False

        # Playback device (may legitimately fail on machines w/o audio out).
        player = Player(clip)
        print(f"  playback device: "
              f"{'OK' if not player.error else player.error}")
        player.close()
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def run_gui(sync_fps, iface, audio_path, smoke_ms=0):
    import tkinter as tk
    root = tk.Tk()
    app = ShowApp(root, sync_fps=sync_fps, iface=iface,
                  audio_path=audio_path)
    root.geometry("1280x760")
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()
    root.after(500, lambda: root.attributes("-topmost", False))

    def on_close():
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    if smoke_ms:
        root.after(smoke_ms, on_close)
    root.mainloop()
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Show Studio — choreograph lights to a song.")
    p.add_argument("--audio", help="Audio file to load on startup "
                   "(mp3/wav/flac/ogg).")
    p.add_argument("--sync-fps", type=int, default=20,
                   help="Device update rate when syncing "
                        f"(max {ls.MAX_SYNC_FPS}).")
    p.add_argument("--iface", help="Local IP of the interface for LAN scan.")
    p.add_argument("--selftest", action="store_true",
                   help="Headless decode/analysis/cue-engine test, no GUI.")
    p.add_argument("--smoke", type=int, default=0, metavar="MS",
                   help="Open the GUI and auto-close after MS milliseconds "
                        "(for testing).")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    return run_gui(args.sync_fps, args.iface, args.audio, args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
