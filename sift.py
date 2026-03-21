#!/usr/bin/env python3
"""
Sift — Tinder for Your Music Library

Chorus detection chain (first hit wins):
  1. LRCLIB  — free crowdsourced synced-lyrics API, no key required
  2. librosa  — RMS energy sliding-window fallback

Deps:
  pip install librosa soundfile mutagen requests send2trash --break-system-packages
"""

import os
import re
import json
import math
import tempfile
import shutil
import atexit
import threading
import datetime
from collections import Counter
from contextlib import contextmanager

from send2trash import send2trash

import gi
import librosa
import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile
from mutagen.id3 import ID3
from mutagen.mp3 import MP3

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gst, Gtk, Adw, GLib, Gio, Gdk

Gst.init(None)


# ── Constants ─────────────────────────────────────────────────────────────────

CLIP_DIR  = os.path.join(tempfile.gettempdir(), "sift_clips")
CLIP_SECS = 30
SR        = 22050

LIKED_NAME   = "sift_liked.txt"
TRASH_NAME   = "sift_trash.txt"
STATE_NAME   = "sift_state.json"
STATS_NAME   = "sift_stats.json"

_CONFIG_DIR  = os.path.join(GLib.get_user_config_dir(), "sift")
APP_CFG_FILE = os.path.join(_CONFIG_DIR, "config.json")

_DEFAULT_WORKSPACE = os.path.join(
    GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_MUSIC)
    or os.path.expanduser("~/Music"),
    "sift-workspace",
)

atexit.register(lambda: shutil.rmtree(CLIP_DIR, ignore_errors=True))


# ── App config helpers ────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(APP_CFG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(data: dict) -> None:
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(APP_CFG_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Workspace helpers ─────────────────────────────────────────────────────────

def workspace_paths(workspace: str) -> tuple[str, str, str, str]:
    """Return (liked_file, trash_file, state_file, stats_file) for a workspace."""
    return (
        os.path.join(workspace, LIKED_NAME),
        os.path.join(workspace, TRASH_NAME),
        os.path.join(workspace, STATE_NAME),
        os.path.join(workspace, STATS_NAME),
    )

def ensure_workspace(workspace: str) -> None:
    os.makedirs(workspace, exist_ok=True)


# ── Stats persistence ─────────────────────────────────────────────────────────

_EMPTY_STATS = {
    "library": {
        "judged": 0, "kept": 0, "trashed": 0, "skipped": 0,
        "artists": {}, "genres": {},
    },
    "new_music": {
        "judged": 0, "kept": 0, "trashed": 0, "skipped": 0,
        "artists": {}, "genres": {},
    },
    "deleted": [],   # list of {path, size, deleted_at}
}

def load_stats(fname: str) -> dict:
    """Load stats from workspace. Returns a fresh empty stats dict if missing."""
    try:
        with open(fname) as f:
            data = json.load(f)
        # Ensure all keys exist (forward compat)
        for mode in ("library", "new_music"):
            data.setdefault(mode, dict(_EMPTY_STATS[mode]))
            for k in ("judged", "kept", "trashed", "skipped", "artists", "genres"):
                data[mode].setdefault(k, {} if k in ("artists", "genres") else 0)
        data.setdefault("deleted", [])
        return data
    except Exception:
        return json.loads(json.dumps(_EMPTY_STATS))

def save_stats(fname: str, stats: dict) -> None:
    tmp = fname + ".tmp"
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp, fname)


# ── LRC / lyrics helpers ──────────────────────────────────────────────────────

_LRC_RE = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")

def _parse_lrc(text: str) -> list[tuple[float, str]]:
    out = []
    for line in text.splitlines():
        m = _LRC_RE.match(line.strip())
        if m:
            t = float(m.group(1)) * 60 + float(m.group(2))
            out.append((t, m.group(3).strip()))
    return out

def _chorus_from_lrc(lines: list[tuple[float, str]]) -> float | None:
    if not lines:
        return None
    counts     = Counter(txt.lower().strip() for _, txt in lines)
    repeats    = [ts for ts, txt in lines
                  if counts[txt.lower().strip()] >= 2 and len(txt.strip()) > 8]
    candidates = [t for t in repeats if t > 20.0]
    return min(candidates) if candidates else None


# ── Chorus detection ──────────────────────────────────────────────────────────

def _lrclib_start(title: str, artist: str, duration: float) -> float | None:
    if not _HAS_REQUESTS or not title or not artist:
        return None
    try:
        params = {"track_name": title, "artist_name": artist}
        if duration:
            params["duration"] = int(duration)
        r = _req.get(
            "https://lrclib.net/api/get", params=params, timeout=6,
            headers={"User-Agent": "Sift/1.0"},
        )
        if r.status_code != 200:
            return None
        synced = r.json().get("syncedLyrics") or ""
        return _chorus_from_lrc(_parse_lrc(synced)) if synced else None
    except Exception as e:
        print(f"[lrclib] {e}")
        return None

def _librosa_start(y: np.ndarray, sr: int) -> float:
    hop    = 512
    rms    = librosa.feature.rms(y=y, hop_length=hop)[0]
    times  = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    win    = int(CLIP_SECS * sr / hop)
    lo     = max(0, int(len(rms) * 0.15))
    hi     = max(int(len(rms) * 0.80), lo + win + 1)
    best_i = lo
    best_s = -1.0
    for i in range(lo, min(hi, len(rms) - win)):
        s = float(np.mean(rms[i : i + win]))
        if s > best_s:
            best_s = s
            best_i = i
    return float(times[best_i]) if best_i < len(times) else len(y) / sr * 0.25

def find_start(path: str, title: str, artist: str,
               y: np.ndarray, sr: int, duration: float) -> tuple[float, str]:
    t = _lrclib_start(title, artist, duration)
    if t is not None:
        print(f"[lrclib]  {os.path.basename(path)} → {t:.1f}s")
        return t, "lrclib"
    t = _librosa_start(y, sr)
    print(f"[librosa] {os.path.basename(path)} → {t:.1f}s")
    return t, "librosa"


# ── Clip extraction ───────────────────────────────────────────────────────────

def extract_clip(src: str, start: float) -> str | None:
    try:
        os.makedirs(CLIP_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in base)
        out  = os.path.join(CLIP_DIR, f"{safe}_clip.wav")
        y, sr = librosa.load(src, sr=SR, mono=True,
                             offset=start, duration=float(CLIP_SECS))
        sf.write(out, y, sr)
        return out
    except Exception as e:
        print(f"[clip] {e}")
        return None


# ── Mutagen / tag helpers ─────────────────────────────────────────────────────

def _cover_bytes(path: str) -> bytes | None:
    try:
        f = MutagenFile(path, easy=False)
        if f is None:
            return None
        if hasattr(f, "pictures") and f.pictures:
            return f.pictures[0].data
        if isinstance(f, MP3):
            frames = ID3(path).getall("APIC")
            if frames:
                return frames[0].data
        if hasattr(f, "tags") and f.tags:
            v = f.tags.get("covr")
            if v:
                return bytes(v[0])
    except Exception:
        pass
    return None

def _tag(audio, *keys) -> str:
    if audio is None:
        return ""
    for k in keys:
        v = audio.get(k)
        if v:
            return str(v[0]) if isinstance(v, list) else str(v)
    return ""

def read_tags(path: str) -> tuple[str, str, float]:
    try:
        f      = MutagenFile(path, easy=True)
        title  = _tag(f, "title") or os.path.splitext(os.path.basename(path))[0]
        artist = _tag(f, "artist") or ""
        dur    = getattr(getattr(f, "info", None), "length", 0.0) or 0.0
        return title, artist, dur
    except Exception:
        return os.path.basename(path), "", 0.0


# ── Persistence helpers ───────────────────────────────────────────────────────

def load_set(fname: str) -> set:
    if os.path.exists(fname):
        with open(fname) as f:
            return {l.strip() for l in f if l.strip()}
    return set()

def save_set(fname: str, songs: set) -> None:
    tmp = fname + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(s + "\n" for s in sorted(songs))
    os.replace(tmp, fname)

def load_state(fname: str) -> dict:
    try:
        with open(fname) as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(fname: str, data: dict) -> None:
    with open(fname, "w") as f:
        json.dump(data, f)

def safe_uri(path: str) -> str:
    return GLib.filename_to_uri(os.path.abspath(path), None)


# ── Size formatting ───────────────────────────────────────────────────────────

def _file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return 0

def _fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ── Spectrogram widget ────────────────────────────────────────────────────────

class Spectrogram(Gtk.DrawingArea):
    """
    Mel-spectrogram visualiser drawn with Cairo.
    Bars left of the playhead are lit in accent green; right are dimmed.
    Computed in a background thread so it doesn't block the UI.
    """

    @property
    def N_BARS(self):
        width = self.get_width()
        if width < 100:
            return 40
        elif width < 250:
            return 60
        else:
            return 80

    H   = 52
    FG  = (0.18, 0.78, 0.49)
    DIM = (0.38, 0.38, 0.42)

    def __init__(self):
        super().__init__()
        self.set_size_request(-1, self.H)
        self.set_draw_func(self._draw)
        self._bars: list[float] = []
        self._pos:  float       = 0.0
        self._path: str | None  = None

    def load(self, path: str):
        self._bars = []
        self._path = path
        self._pos  = 0.0
        self.queue_draw()
        threading.Thread(target=self._compute, args=(path,), daemon=True).start()

    def set_pos(self, pos_s: float, dur_s: float):
        if dur_s > 0:
            self._pos = max(0.0, min(1.0, pos_s / dur_s))
            self.queue_draw()

    def _compute(self, path: str):
        try:
            y, sr  = librosa.load(path, sr=SR, mono=True, duration=120)
            mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            db     = librosa.power_to_db(mel, ref=np.max)
            bins   = np.array_split(np.arange(db.shape[1]), self.N_BARS)
            bars   = [float(db[:, b].mean()) if len(b) else -80.0 for b in bins]
            lo, hi = min(bars), max(bars)
            rng    = hi - lo if hi != lo else 1.0
            bars   = [(v - lo) / rng for v in bars]
            if self._path == path:
                self._bars = bars
                GLib.idle_add(self.queue_draw)
        except Exception as e:
            print(f"[spectrogram] {e}")

    def _draw(self, _area, cr, w, h):
        bars = self._bars or [0.15] * self.N_BARS
        bw   = w / len(bars)
        hx   = w * self._pos
        r    = min(bw * 0.28, 2.2)

        if bw < 2:
            return

        for i, v in enumerate(bars):
            x  = i * bw
            bh = max(3, v * (h - 8))
            y  = (h - bh) / 2
            cx = x + bw / 2

            if cx <= hx:
                cr.set_source_rgba(*self.FG, 0.88 if self._bars else 0.25)
            else:
                cr.set_source_rgba(*self.DIM, 0.40 if self._bars else 0.18)

            cr.arc(x + r + 1,        y + r,      r, math.pi,         3 * math.pi / 2)
            cr.arc(x + bw - r - 1,   y + r,      r, 3 * math.pi / 2, 0)
            cr.arc(x + bw - r - 1,   y + bh - r, r, 0,               math.pi / 2)
            cr.arc(x + r + 1,        y + bh - r, r, math.pi / 2,     math.pi)
            cr.close_path()
            cr.fill()

        if self._bars:
            cr.set_source_rgba(1, 1, 1, 0.65)
            cr.set_line_width(1.5)
            cr.move_to(hx, 4)
            cr.line_to(hx, h - 4)
            cr.stroke()


# ── Application ───────────────────────────────────────────────────────────────

class Sift(Adw.Application):

    def __init__(self):
        super().__init__(
            application_id="io.github.IdleEndeavor.Sift",
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        self.player = Gst.ElementFactory.make("playbin", "player")

        self.queue:   list[str]                               = []
        self.cache:   dict[str, tuple[float, str | None, str]] = {}
        self.pending: set[str]                                = set()

        self.idx       = 0
        self.music_dir = ""
        self.history:    list[tuple[str, str]] = []

        # New music mode — plays full songs, forgets folder when done
        self._new_music_mode: bool = False

        cfg = load_config()
        self.workspace = cfg.get("workspace", _DEFAULT_WORKSPACE)

        ensure_workspace(self.workspace)
        self._liked_file, self._trash_file, self._state_file, self._stats_file = \
            workspace_paths(self.workspace)

        self.liked = load_set(self._liked_file)
        self.trash = load_set(self._trash_file)
        self._stats = load_stats(self._stats_file)

        self._backfill_stats()

        state           = load_state(self._state_file)
        self._saved_dir = state.get("dir", "")
        self._saved_idx = state.get("index", 0)

        self._seek_id: int = 0
        self._dash_selection: dict[str, set] = {"liked": set(), "trash": set()}

    # ── Stats recording ───────────────────────────────────────────────────────

    def _mode_key(self) -> str:
        """Return the stats key for the current mode."""
        return "new_music" if self._new_music_mode else "library"

    def _record_decision(self, kind: str, path: str) -> None:
        """Record a judging decision to persistent stats."""
        mode = self._mode_key()
        self._stats[mode]["judged"] += 1
        if kind == "heart":
            self._stats[mode]["kept"] += 1
        elif kind == "trash":
            self._stats[mode]["trashed"] += 1
        elif kind == "skip":
            self._stats[mode]["skipped"] += 1

        # Artist and genre counts
        try:
            f      = MutagenFile(path, easy=True)
            artist = _tag(f, "artist") or "Unknown Artist"
            genre  = _tag(f, "genre")  or "Unknown"
        except Exception:
            artist = "Unknown Artist"
            genre  = "Unknown"

        artists = self._stats[mode]["artists"]
        artists[artist] = artists.get(artist, 0) + 1

        genres = self._stats[mode]["genres"]
        genres[genre] = genres.get(genre, 0) + 1

        save_stats(self._stats_file, self._stats)

    def _record_deletion(self, path: str, size_bytes: int) -> None:
        """Append a deletion record to stats."""
        self._stats["deleted"].append({
            "path":       path,
            "size":       size_bytes,
            "deleted_at": datetime.datetime.now().isoformat(),
        })
        save_stats(self._stats_file, self._stats)

    def _backfill_stats(self) -> None:
        """
        Populate stats from existing liked/trash lists if stats are empty.
        Only runs once — marks itself done in the stats file so it doesn't
        double-count on subsequent launches.
        """
        if self._stats.get("backfilled"):
            return

        for path in self.liked:
            try:
                f      = MutagenFile(path, easy=True)
                artist = _tag(f, "artist") or "Unknown Artist"
                genre  = _tag(f, "genre")  or "Unknown"
            except Exception:
                artist = "Unknown Artist"
                genre  = "Unknown"

            self._stats["library"]["judged"] += 1
            self._stats["library"]["kept"]   += 1
            self._stats["library"]["artists"][artist] = \
                self._stats["library"]["artists"].get(artist, 0) + 1
            self._stats["library"]["genres"][genre] = \
                self._stats["library"]["genres"].get(genre, 0) + 1

        for path in self.trash:
            try:
                f      = MutagenFile(path, easy=True)
                artist = _tag(f, "artist") or "Unknown Artist"
                genre  = _tag(f, "genre")  or "Unknown"
            except Exception:
                artist = "Unknown Artist"
                genre  = "Unknown"

            self._stats["library"]["judged"]  += 1
            self._stats["library"]["trashed"] += 1
            self._stats["library"]["artists"][artist] = \
                self._stats["library"]["artists"].get(artist, 0) + 1
            self._stats["library"]["genres"][genre] = \
                self._stats["library"]["genres"].get(genre, 0) + 1

        self._stats["backfilled"] = True
        save_stats(self._stats_file, self._stats)


    # ── Workspace management ──────────────────────────────────────────────────

    def _set_workspace(self, new_path: str) -> None:
        save_set(self._liked_file, self.liked)
        save_set(self._trash_file, self.trash)

        self.workspace = new_path
        ensure_workspace(self.workspace)
        self._liked_file, self._trash_file, self._state_file, self._stats_file = \
            workspace_paths(self.workspace)

        self.liked  = load_set(self._liked_file)
        self.trash  = load_set(self._trash_file)
        self._stats = load_stats(self._stats_file)

        state           = load_state(self._state_file)
        self._saved_dir = state.get("dir", "")
        self._saved_idx = state.get("index", 0)

        save_config({"workspace": self.workspace})
        self._toast(f"Workspace set to {os.path.basename(new_path)}")

    def _pick_workspace(self) -> None:
        dialog = Gtk.FileDialog.new()
        dialog.set_title("Choose Workspace Folder")
        dialog.select_folder(self.win, None, self._workspace_chosen)

    def _workspace_chosen(self, dialog, result) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error as e:
            print(f"[workspace] {e}")
            return
        if folder:
            self._set_workspace(folder.get_path())

    def _reset_workspace(self) -> None:
        self._set_workspace(_DEFAULT_WORKSPACE)


    # ── App startup ───────────────────────────────────────────────────────────

    def do_activate(self):
        icon_theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())
        icon_theme.add_search_path(os.path.dirname(os.path.abspath(__file__)))

        self.win = Adw.ApplicationWindow(application=self)
        self.win.set_default_size(700, 850)
        self.win.set_title("Sift")

        kc = Gtk.EventControllerKey.new()
        kc.connect("key-pressed", self._key)
        self.win.add_controller(kc)

        self._css()

        self.stack = Gtk.Stack()
        self.stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self.stack.add_named(self._build_setup(),     "setup")
        self.stack.add_named(self._build_player(),    "player")
        self.stack.add_named(self._build_dashboard(), "dashboard")
        self.stack.connect("notify::visible-child", self._on_screen_changed)

        self.toast_overlay = Adw.ToastOverlay()
        self.toast_overlay.set_child(self.stack)
        self.win.set_content(self.toast_overlay)

        if self._saved_dir and os.path.isdir(self._saved_dir):
            self.music_dir = self._saved_dir
            self._index_library(resume_idx=self._saved_idx)
            if self.queue:
                self.stack.set_visible_child_name("player")
                self._load_song()

        self.win.present()

    def _css(self):
        p = Gtk.CssProvider()
        p.load_from_data(b"""
            .action-btn  { border-radius: 99px; min-width: 78px;  min-height: 78px;  }
            .play-btn    { border-radius: 99px; min-width: 94px;  min-height: 94px;  }
            .info-btn    { border-radius: 99px; min-width: 52px;  min-height: 52px;  }
            .trash-btn   { background: #c01c28; color: white; }
            .heart-btn   { background: #26a269; color: white; }
            .cover-frame { border-radius: 16px; background: transparent; }
        """)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), p,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def _on_screen_changed(self, _stack, _param):
        if self.stack.get_visible_child_name() == "player":
            self.play_btn.grab_focus()


    # ── Setup screen ──────────────────────────────────────────────────────────

    def _build_setup(self) -> Gtk.Widget:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Header bar with close button
        hdr = Adw.HeaderBar()
        hdr.add_css_class("flat")
        hdr.set_show_end_title_buttons(False)

        close_btn = Gtk.Button(icon_name="window-close-symbolic")
        close_btn.set_tooltip_text("Quit Sift")
        close_btn.connect("clicked", lambda _: self.quit())
        hdr.pack_end(close_btn)
        root.append(hdr)

        # Scrollable centred content
        scroll = Gtk.ScrolledWindow(vexpand=True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=8,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        box.set_vexpand(True)
        box.set_margin_start(48)
        box.set_margin_end(48)
        box.set_margin_top(24)
        box.set_margin_bottom(24)

        icon = Gtk.Image.new_from_icon_name("io.github.IdleEndeavor.Sift")
        icon.set_pixel_size(128)
        icon.set_opacity(0.85)
        box.append(icon)

        box.append(self._spacer(8))

        title = Gtk.Label(label="Sift")
        title.add_css_class("title-1")
        box.append(title)

        tagline = Gtk.Label(label="Tinder for Your Music Library")
        tagline.add_css_class("body")
        tagline.set_opacity(0.5)
        box.append(tagline)

        box.append(self._spacer(20))

        # ── Mode toggle ───────────────────────────────────────────────────
        mode_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=0,
            halign=Gtk.Align.CENTER,
        )
        mode_box.add_css_class("linked")

        self._mode_library_btn = Gtk.ToggleButton(label="Judge Library")
        self._mode_library_btn.set_active(True)
        self._mode_library_btn.set_tooltip_text(
            "Seek to the chorus of each song and judge your library")

        self._mode_new_btn = Gtk.ToggleButton(label="New Music")
        self._mode_new_btn.set_tooltip_text(
            "Play full songs from the start — forgets folder when done")
        self._mode_new_btn.set_group(self._mode_library_btn)

        mode_box.append(self._mode_library_btn)
        mode_box.append(self._mode_new_btn)
        box.append(mode_box)

        box.append(self._spacer(8))

        # ── Main buttons ──────────────────────────────────────────────────
        pick = Gtk.Button(label="Select Music Folder")
        pick.add_css_class("suggested-action")
        pick.add_css_class("pill")
        pick.connect("clicked", lambda _: self._pick_folder_for_mode())
        box.append(pick)

        dash = Gtk.Button(label="Library Dashboard")
        dash.add_css_class("pill")
        dash.connect("clicked", lambda _: self._open_dashboard())
        box.append(dash)

        box.append(self._spacer(20))

        instructions = Adw.PreferencesGroup(title="How to use")
        for key, desc in [
            ("→  Keep",  "Like the song and move on"),
            ("←  Trash", "Mark the song for removal"),
            ("↓  Skip",  "Skip without deciding"),
            ("Space",    "Play / pause"),
        ]:
            row = Adw.ActionRow(title=key, subtitle=desc)
            row.set_use_markup(False)
            instructions.add(row)
        box.append(instructions)

        scroll.set_child(box)
        root.append(scroll)
        return root


    # ── Player screen ─────────────────────────────────────────────────────────

    def _build_player(self) -> Gtk.Widget:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        hdr = Adw.HeaderBar()
        hdr.add_css_class("flat")

        dash_btn = Gtk.Button(icon_name="view-grid-symbolic")
        dash_btn.set_tooltip_text("Dashboard  Ctrl+D")
        dash_btn.connect("clicked", lambda _: self._open_dashboard())
        hdr.pack_start(dash_btn)

        menu_btn = Gtk.MenuButton()
        menu_btn.set_icon_name("open-menu-symbolic")
        menu_btn.set_tooltip_text("Menu")
        menu_btn.set_menu_model(self._build_menu())
        hdr.pack_end(menu_btn)

        # Title shows mode; counter shows songs left
        self._player_title_lbl = Gtk.Label(label="Judge Library")
        self._player_title_lbl.add_css_class("heading")
        self.lbl_counter = Gtk.Label(label="")
        self.lbl_counter.add_css_class("caption")
        self.lbl_counter.set_opacity(0.6)

        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.set_halign(Gtk.Align.CENTER)
        title_box.append(self._player_title_lbl)
        title_box.append(self.lbl_counter)
        hdr.set_title_widget(title_box)

        root.append(hdr)

        col = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=14,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        col.set_vexpand(True)
        col.set_margin_top(12)
        col.set_margin_bottom(32)
        col.set_margin_start(40)
        col.set_margin_end(40)

        self.cover_frame = Gtk.AspectFrame(ratio=1.0, obey_child=False)
        self.cover_frame.set_size_request(320, 320)
        self.cover_frame.add_css_class("cover-frame")
        self.cover_pic = Gtk.Picture()
        self.cover_pic.set_content_fit(Gtk.ContentFit.COVER)
        self.cover_pic.set_can_shrink(True)
        self.cover_frame.set_child(self.cover_pic)
        col.append(self.cover_frame)

        meta = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.lbl_title = Gtk.Label()
        self.lbl_title.add_css_class("title-2")
        self.lbl_title.set_justify(Gtk.Justification.CENTER)
        self.lbl_title.set_wrap(True)
        self.lbl_title.set_max_width_chars(36)
        self.lbl_artist = Gtk.Label()
        self.lbl_artist.add_css_class("body")
        self.lbl_artist.set_opacity(0.55)
        self.lbl_artist.set_wrap(True)
        self.lbl_artist.set_max_width_chars(36)
        self.lbl_artist.set_justify(Gtk.Justification.CENTER)
        meta.append(self.lbl_title)
        meta.append(self.lbl_artist)
        col.append(meta)

        self.lbl_method = Gtk.Label()
        self.lbl_method.add_css_class("caption")
        self.lbl_method.set_opacity(0.38)
        col.append(self.lbl_method)

        self.spectro = Spectrogram()
        self.spectro.set_size_request(380, Spectrogram.H)
        col.append(self.spectro)

        trow = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        trow.set_size_request(380, -1)
        self.lbl_pos = Gtk.Label(label="0:00")
        self.lbl_pos.add_css_class("caption-heading")
        self.lbl_dur = Gtk.Label(label="0:00")
        self.lbl_dur.add_css_class("caption-heading")
        self.lbl_dur.set_hexpand(True)
        self.lbl_dur.set_halign(Gtk.Align.END)
        trow.append(self.lbl_pos)
        trow.append(self.lbl_dur)
        col.append(trow)

        self.seek = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        self.seek.set_draw_value(False)
        self.seek.set_size_request(380, -1)
        self._seek_id = self.seek.connect("value-changed", self._seek_manual)
        col.append(self.seek)
        GLib.timeout_add(500, self._tick_position)

        btns = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=26,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )

        undo_btn = self._mkbtn("edit-undo-symbolic", ["info-btn"], "Undo  Ctrl+Z")
        undo_btn.set_size_request(52, 52)
        undo_btn.set_valign(Gtk.Align.CENTER)
        undo_btn.connect("clicked", lambda _: self._undo())

        trash_btn = self._mkbtn("user-trash-full-symbolic",
                                ["action-btn", "trash-btn"], "Trash  ←")
        trash_btn.connect("clicked", lambda _: self._action("trash"))

        mid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.play_btn = self._mkbtn("media-playback-pause-symbolic",
                                    ["play-btn"], "Play/Pause  Space")
        self.play_btn.connect("clicked", lambda _: self._toggle_play())
        skip_btn = Gtk.Button(label="Skip")
        skip_btn.add_css_class("pill")
        skip_btn.set_tooltip_text("Skip  ↓")
        skip_btn.connect("clicked", lambda _: self._action("skip"))
        mid.append(self.play_btn)
        mid.append(skip_btn)

        heart_btn = self._mkbtn("starred-symbolic",
                                ["action-btn", "heart-btn"], "Keep  →")
        heart_btn.connect("clicked", lambda _: self._action("heart"))

        info_btn = self._mkbtn("dialog-information-symbolic",
                               ["info-btn"], "Song info  I")
        info_btn.set_size_request(52, 52)
        info_btn.set_valign(Gtk.Align.CENTER)
        info_btn.connect("clicked", lambda _: self._show_info())

        btns.append(undo_btn)
        btns.append(trash_btn)
        btns.append(mid)
        btns.append(heart_btn)
        btns.append(info_btn)

        col.append(btns)
        root.append(col)
        self.play_btn.grab_focus()
        return root


    # ── Dashboard ─────────────────────────────────────────────────────────────

    def _build_dashboard(self) -> Gtk.Widget:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        self._dash_stack = Adw.ViewStack()
        self._liked_lb   = self._song_listbox()
        self._trash_lb   = self._song_listbox()
        self._stats_content_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=16)
        self._stats_content_box.set_margin_top(16)
        self._stats_content_box.set_margin_bottom(16)
        self._stats_content_box.set_margin_start(16)
        self._stats_content_box.set_margin_end(16)

        def _scroll(child):
            s = Gtk.ScrolledWindow(vexpand=True)
            s.set_child(child)
            return s

        self._dash_stack.add_titled_with_icon(
            _scroll(self._liked_lb),          "liked", "Liked",      "starred-symbolic")
        self._dash_stack.add_titled_with_icon(
            _scroll(self._trash_lb),          "trash", "Trashed",    "user-trash-symbolic")
        self._dash_stack.add_titled_with_icon(
            _scroll(self._stats_content_box), "stats", "Statistics", "chart-line-symbolic")

        self._dash_stack.connect("notify::visible-child", self._on_dash_tab_changed)

        switcher = Adw.ViewSwitcher()
        switcher.set_stack(self._dash_stack)
        switcher.set_policy(Adw.ViewSwitcherPolicy.WIDE)

        hdr = Adw.HeaderBar()
        hdr.add_css_class("flat")
        back = Gtk.Button(icon_name="go-previous-symbolic")
        back.set_tooltip_text("Back")
        back.connect("clicked", self._dash_back)
        hdr.pack_start(back)
        hdr.set_title_widget(switcher)

        root.append(hdr)
        root.append(self._dash_stack)

        # Bottom action bar
        self._dash_action_bar = Gtk.ActionBar()

        select_all_btn = Gtk.Button(label="Select All")
        select_all_btn.add_css_class("flat")
        select_all_btn.connect("clicked", lambda _: self._dash_select_all())
        self._dash_action_bar.pack_start(select_all_btn)

        self._dash_selection_label = Gtk.Label(label="")
        self._dash_selection_label.add_css_class("caption")
        self._dash_action_bar.set_center_widget(self._dash_selection_label)

        self._dash_delete_btn = Gtk.Button(label="Move to Trash")
        self._dash_delete_btn.add_css_class("destructive-action")
        self._dash_delete_btn.add_css_class("pill")
        self._dash_delete_btn.connect("clicked", lambda _: self._dash_bulk_delete())

        self._dash_restore_btn = Gtk.Button(label="Restore to Queue")
        self._dash_restore_btn.add_css_class("pill")
        self._dash_restore_btn.connect("clicked", lambda _: self._dash_bulk_restore())

        self._dash_action_bar.pack_end(self._dash_delete_btn)
        self._dash_action_bar.pack_end(self._dash_restore_btn)
        self._dash_action_bar.set_revealed(False)

        root.append(self._dash_action_bar)
        return root


    # ── Statistics page ───────────────────────────────────────────────────────

    def _refresh_stats(self):
        """Recompute and redraw the statistics page from stored + live data."""
        while (child := self._stats_content_box.get_first_child()):
            self._stats_content_box.remove(child)

        s  = self._stats
        lib = s["library"]
        nm  = s["new_music"]
        deleted = s["deleted"]

        total_judged  = lib["judged"]  + nm["judged"]
        total_kept    = lib["kept"]    + nm["kept"]
        total_trashed = lib["trashed"] + nm["trashed"]
        total_skipped = lib["skipped"] + nm["skipped"]
        total_pct_t   = (total_trashed / total_judged * 100) if total_judged else 0.0
        total_pct_k   = (total_kept    / total_judged * 100) if total_judged else 0.0

        # Live file sizes
        liked_size       = sum(_file_size_bytes(p) for p in self.liked if os.path.exists(p))
        trash_on_disk    = [p for p in self.trash if os.path.exists(p)]
        trash_on_disk_sz = sum(_file_size_bytes(p) for p in trash_on_disk)
        deleted_count = sum(r.get("count", 1) for r in deleted)
        deleted_size     = sum(r.get("size", 0) for r in deleted)

        # Combined artist / genre counts
        combined_artists: dict[str, int] = {}
        combined_genres:  dict[str, int] = {}
        for mode_key in ("library", "new_music"):
            for artist, count in s[mode_key]["artists"].items():
                combined_artists[artist] = combined_artists.get(artist, 0) + count
            for genre, count in s[mode_key]["genres"].items():
                combined_genres[genre] = combined_genres.get(genre, 0) + count

        top_artists = sorted(combined_artists.items(), key=lambda x: x[1], reverse=True)[:5]
        top_genres  = sorted(combined_genres.items(),  key=lambda x: x[1], reverse=True)[:5]

        def _group(title: str, description: str = "") -> Adw.PreferencesGroup:
            g = Adw.PreferencesGroup()
            g.set_title(title)
            if description:
                g.set_description(description)
            return g

        def _row(title: str, value: str) -> Adw.ActionRow:
            row = Adw.ActionRow()
            row.set_title(title)
            lbl = Gtk.Label(label=value)
            lbl.add_css_class("body")
            lbl.set_valign(Gtk.Align.CENTER)
            row.add_suffix(lbl)
            return row

        if total_judged == 0:
            lbl = Gtk.Label(label="No songs judged yet. Start sifting!")
            lbl.add_css_class("body")
            lbl.set_opacity(0.5)
            self._stats_content_box.append(lbl)
            return

        # ── Total Overview ────────────────────────────────────────────────
        ov = _group("Total Overview")
        ov.add(_row("Total Judged",   str(total_judged)))
        ov.add(_row("Kept",           f"{total_kept} ({total_pct_k:.1f}%)"))
        ov.add(_row("Trashed",        f"{total_trashed} ({total_pct_t:.1f}%)"))
        ov.add(_row("Skipped",        str(total_skipped)))
        ov.add(_row("Liked Library Size", _fmt_bytes(liked_size)))
        self._stats_content_box.append(ov)

        # ── Space ─────────────────────────────────────────────────────────
        sp = _group("Space",
            "Files marked as trash but still on disk can be deleted from the Trashed tab.")
        sp.add(_row("Deleted from Disk",
                    f"{deleted_count} files · {_fmt_bytes(deleted_size)} freed"))
        sp.add(_row("Marked Trash Still on Disk",
                    f"{len(trash_on_disk)} files · {_fmt_bytes(trash_on_disk_sz)} reclaimable"))
        self._stats_content_box.append(sp)

        # ── By Mode ───────────────────────────────────────────────────────
        modes_group = _group("By Mode")

        def _mode_pct(kept, trashed, judged):
            if judged == 0:
                return "—"
            return f"{judged} judged · {kept} kept · {trashed} trashed"

        modes_group.add(_row("Judge Library",
                             _mode_pct(lib["kept"], lib["trashed"], lib["judged"])))
        modes_group.add(_row("New Music",
                             _mode_pct(nm["kept"], nm["trashed"], nm["judged"])))
        self._stats_content_box.append(modes_group)

        # ── Top Artists ───────────────────────────────────────────────────
        if top_artists:
            ag = _group("Most Judged Artists")
            for artist, count in top_artists:
                ag.add(_row(GLib.markup_escape_text(artist),
                            f"{count} song{'s' if count != 1 else ''}"))
            self._stats_content_box.append(ag)

        # ── Top Genres ────────────────────────────────────────────────────
        if top_genres:
            gg = _group("Top Genres")
            for genre, count in top_genres:
                gg.add(_row(GLib.markup_escape_text(genre),
                            f"{count} song{'s' if count != 1 else ''}"))
            self._stats_content_box.append(gg)


    # ── Dashboard helpers ─────────────────────────────────────────────────────

    def _current_dash_kind(self) -> str:
        name = self._dash_stack.get_visible_child_name()
        return name if name in ("liked", "trash") else "liked"

    def _on_dash_tab_changed(self, _stack, _param):
        name = self._dash_stack.get_visible_child_name()
        if name == "stats":
            self._refresh_stats()
            self._dash_action_bar.set_revealed(False)
        else:
            self._dash_selection["liked"].clear()
            self._dash_selection["trash"].clear()
            self._update_dash_action_bar()
            self._refresh_dash()

    def _dash_toggle(self, checkbox: Gtk.CheckButton, path: str, kind: str):
        if checkbox.get_active():
            self._dash_selection[kind].add(path)
        else:
            self._dash_selection[kind].discard(path)
        self._update_dash_action_bar()

    def _dash_select_all(self):
        kind = self._current_dash_kind()
        source = self.liked if kind == "liked" else self.trash
        self._dash_selection[kind] = set(source)
        self._update_dash_action_bar()
        self._refresh_dash()

    def _update_dash_action_bar(self):
        if self._dash_stack.get_visible_child_name() == "stats":
            self._dash_action_bar.set_revealed(False)
            return
        kind  = self._current_dash_kind()
        count = len(self._dash_selection[kind])
        self._dash_action_bar.set_revealed(count > 0)
        self._dash_selection_label.set_text(
            f"{count} song{'s' if count != 1 else ''} selected")
        self._dash_restore_btn.set_visible(kind == "trash")

    def _dash_bulk_delete(self):
        kind  = self._current_dash_kind()
        paths = list(self._dash_selection[kind])
        if not paths:
            return
        names = "\n".join(os.path.basename(p) for p in paths[:5])
        if len(paths) > 5:
            names += f"\n… and {len(paths) - 5} more"
        dlg = Adw.AlertDialog(
            heading=f"Move {len(paths)} file{'s' if len(paths) != 1 else ''} to trash?",
            body=names,
        )
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("delete", "Move to Trash")
        dlg.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.set_default_response("cancel")
        dlg.set_close_response("cancel")
        dlg.connect("response", lambda d, r, p=paths, k=kind:
                    self._dash_bulk_delete_confirmed(d, r, p, k))
        dlg.present(self.win)

    def _dash_bulk_delete_confirmed(self, _d, resp: str, paths: list, kind: str):
        if resp != "delete":
            return
        failed = []
        for path in paths:
            try:
                size = _file_size_bytes(path)
                send2trash(path)
                self._record_deletion(path, size)
                s = self.liked if kind == "liked" else self.trash
                f = self._liked_file if kind == "liked" else self._trash_file
                s.discard(path)
                save_set(f, s)
                if path in self.queue:
                    i = self.queue.index(path)
                    self.queue.remove(path)
                    if i < self.idx:
                        self.idx = max(0, self.idx - 1)
            except Exception as e:
                print(f"[bulk delete] {path}: {e}")
                failed.append(path)
        self._dash_selection[kind].clear()
        self._refresh_dash()
        self._update_dash_action_bar()
        count = len(paths) - len(failed)
        self._toast(f"{count} file{'s' if count != 1 else ''} moved to trash")

    def _dash_bulk_restore(self):
        paths = list(self._dash_selection["trash"])
        if not paths:
            return
        for path in paths:
            self.trash.discard(path)
            if os.path.exists(path) and path not in self.queue:
                self.queue.insert(self.idx, path)
        save_set(self._trash_file, self.trash)
        self._dash_selection["trash"].clear()
        self._refresh_dash()
        self._update_dash_action_bar()
        self._toast(f"{len(paths)} song{'s' if len(paths) != 1 else ''} restored to queue")

    def _song_listbox(self) -> Gtk.ListBox:
        lb = Gtk.ListBox()
        lb.set_selection_mode(Gtk.SelectionMode.NONE)
        lb.add_css_class("boxed-list")
        lb.set_margin_top(8)
        lb.set_margin_bottom(8)
        lb.set_margin_start(12)
        lb.set_margin_end(12)
        return lb

    def _open_dashboard(self):
        self._refresh_dash()
        self.stack.set_visible_child_name("dashboard")

    def _dash_back(self, _btn):
        target = "player" if (self.queue or self.idx > 0) else "setup"
        self.stack.set_visible_child_name(target)

    def _refresh_dash(self):
        self._fill_lb(self._liked_lb, sorted(self.liked), "liked")
        self._fill_lb(self._trash_lb, sorted(self.trash),  "trash")

    def _fill_lb(self, lb: Gtk.ListBox, paths: list, kind: str):
        while (r := lb.get_row_at_index(0)) is not None:
            lb.remove(r)
        if not paths:
            lb.append(Adw.ActionRow(title="Nothing here yet."))
            return
        for p in paths:
            lb.append(self._song_row(p, kind))

    def _song_row(self, path: str, kind: str) -> Gtk.Widget:
        title, artist, _ = read_tags(path)
        exists = os.path.exists(path)

        row = Adw.ActionRow()
        row.set_title(GLib.markup_escape_text(title if exists else f"[Missing] {title}"))
        row.set_subtitle(GLib.markup_escape_text(artist or os.path.basename(path)))
        if not exists:
            row.set_sensitive(False)

        check = Gtk.CheckButton()
        check.set_active(path in self._dash_selection[kind])
        check.set_valign(Gtk.Align.CENTER)
        check.connect("toggled", lambda cb, p=path, k=kind: self._dash_toggle(cb, p, k))
        row.add_prefix(check)
        row.set_activatable_widget(check)

        bbox = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            valign=Gtk.Align.CENTER,
        )

        if kind == "trash":
            b = Gtk.Button(label="Judge Later")
            b.add_css_class("pill")
            b.set_tooltip_text("Restore to judging queue")
            b.connect("clicked", lambda _b, p=path: self._rescue(p))
            bbox.append(b)
        else:
            b = Gtk.Button(label="Un-like")
            b.add_css_class("pill")
            b.set_tooltip_text("Remove from liked, back to queue")
            b.connect("clicked", lambda _b, p=path: self._unlike(p))
            bbox.append(b)

        if exists:
            d = Gtk.Button(icon_name="user-trash-full-symbolic")
            d.add_css_class("destructive-action")
            d.add_css_class("circular")
            d.set_tooltip_text("Move to system trash")
            d.connect("clicked", lambda _b, p=path, k=kind: self._confirm_delete(p, k))
            bbox.append(d)

        row.add_suffix(bbox)
        return row

    def _rescue(self, path: str):
        self.trash.discard(path)
        save_set(self._trash_file, self.trash)
        if os.path.exists(path) and path not in self.queue:
            self.queue.insert(self.idx, path)
        self._refresh_dash()
        self._toast("Song restored to queue")

    def _unlike(self, path: str):
        self.liked.discard(path)
        save_set(self._liked_file, self.liked)
        if os.path.exists(path) and path not in self.queue:
            self.queue.insert(self.idx, path)
        self._refresh_dash()
        self._toast("Song removed from liked")

    def _confirm_delete(self, path: str, kind: str):
        dlg = Adw.AlertDialog(
            heading="Move to trash?",
            body=f"{os.path.basename(path)}\n\nThe file will be moved to your system trash.",
        )
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("delete", "Move to Trash")
        dlg.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.set_default_response("cancel")
        dlg.set_close_response("cancel")
        dlg.connect("response", lambda d, r, p=path, k=kind: self._do_delete(d, r, p, k))
        dlg.present(self.win)

    def _do_delete(self, _d, resp: str, path: str, kind: str):
        if resp != "delete":
            return
        try:
            size = _file_size_bytes(path)
            send2trash(path)
            self._record_deletion(path, size)
        except Exception as e:
            print(f"[delete] {e}")
            return
        s = self.liked if kind == "liked" else self.trash
        f = self._liked_file if kind == "liked" else self._trash_file
        s.discard(path)
        save_set(f, s)
        if path in self.queue:
            i = self.queue.index(path)
            self.queue.remove(path)
            if i < self.idx:
                self.idx = max(0, self.idx - 1)
        self._refresh_dash()
        self._toast("File moved to system trash")


    # ── Song info dialog ──────────────────────────────────────────────────────

    def _show_info(self):
        if self.idx >= len(self.queue):
            return
        path = self.queue[self.idx]

        try:
            audio = MutagenFile(path, easy=False)
            easy  = MutagenFile(path, easy=True)
        except Exception:
            audio = easy = None

        def tag(*keys):
            return _tag(easy, *keys) or "—"

        def fmt_size(p):
            try:
                b = os.path.getsize(p)
                for unit in ("B", "KB", "MB", "GB"):
                    if b < 1024:
                        return f"{b:.1f} {unit}"
                    b /= 1024
            except Exception:
                return "—"

        def fmt_bitrate():
            try:
                return f"{int(audio.info.bitrate / 1000)} kbps"
            except Exception:
                return "—"

        def fmt_samplerate():
            try:
                return f"{audio.info.sample_rate / 1000:.1f} kHz"
            except Exception:
                return "—"

        def fmt_duration():
            try:
                s = int(audio.info.length)
                return f"{s // 60}:{s % 60:02d}"
            except Exception:
                return "—"

        def fmt_channels():
            try:
                return "Stereo" if audio.info.channels == 2 else str(audio.info.channels)
            except Exception:
                return "—"

        rows = [
            ("Title",        tag("title")),
            ("Artist",       tag("artist")),
            ("Album",        tag("album")),
            ("Album Artist", tag("albumartist", "album artist")),
            ("Track",        tag("tracknumber")),
            ("Date",         tag("date", "year")),
            ("Genre",        tag("genre")),
            ("Composer",     tag("composer")),
            ("Comment",      tag("comment")),
            ("Duration",     fmt_duration()),
            ("Bitrate",      fmt_bitrate()),
            ("Sample Rate",  fmt_samplerate()),
            ("Channels",     fmt_channels()),
            ("File Size",    fmt_size(path)),
            ("Format",       os.path.splitext(path)[1].lstrip(".").upper()),
            ("Path",         path),
            ("Workspace",    self.workspace),
            ("Mode",         "New Music" if self._new_music_mode else "Judge Library"),
        ]

        grid = Gtk.Grid()
        grid.set_column_spacing(24)
        grid.set_row_spacing(8)
        grid.set_margin_top(12)
        grid.set_margin_bottom(12)
        grid.set_margin_start(16)
        grid.set_margin_end(16)

        for i, (label, value) in enumerate(rows):
            key_lbl = Gtk.Label(label=label)
            key_lbl.set_halign(Gtk.Align.START)
            key_lbl.set_valign(Gtk.Align.START)
            key_lbl.add_css_class("caption-heading")
            key_lbl.set_opacity(0.55)

            val_lbl = Gtk.Label(label=value)
            val_lbl.set_use_markup(False)
            val_lbl.set_halign(Gtk.Align.START)
            val_lbl.set_valign(Gtk.Align.START)
            val_lbl.set_selectable(True)
            val_lbl.set_wrap(True)
            val_lbl.set_xalign(0)
            val_lbl.add_css_class("body")

            grid.attach(key_lbl, 0, i, 1, 1)
            grid.attach(val_lbl, 1, i, 1, 1)

        scroll = Gtk.ScrolledWindow()
        scroll.set_min_content_height(300)
        scroll.set_max_content_height(500)
        scroll.set_propagate_natural_height(True)
        scroll.set_child(grid)

        dlg = Adw.Dialog()
        dlg.set_title("Song Info")
        dlg.set_content_width(420)

        toolbar_view = Adw.ToolbarView()
        sub_hdr = Adw.HeaderBar()
        sub_hdr.add_css_class("flat")
        toolbar_view.add_top_bar(sub_hdr)
        toolbar_view.set_content(scroll)
        dlg.set_child(toolbar_view)
        dlg.present(self.win)


    # ── Hamburger menu ────────────────────────────────────────────────────────

    def _build_menu(self) -> Gio.MenuModel:
        menu = Gio.Menu()
        menu.append("Preferences",        "app.preferences")
        menu.append("Keyboard Shortcuts", "app.shortcuts")
        menu.append("About Sift",         "app.about")

        actions = {
            "preferences": self._show_preferences,
            "shortcuts":   self._show_shortcuts,
            "about":       self._show_about,
        }
        for name, cb in actions.items():
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", lambda _a, _p, fn=cb: fn())
            self.add_action(a)

        return menu

    def _show_preferences(self):
        dlg = Adw.PreferencesDialog()
        dlg.set_title("Preferences")

        page = Adw.PreferencesPage()
        page.set_title("General")
        page.set_icon_name("preferences-system-symbolic")

        folder_group = Adw.PreferencesGroup()
        folder_group.set_title("Music Folder")
        folder_group.set_description("The folder Sift scans for tracks to judge.")

        folder_row = Adw.ActionRow()
        folder_row.set_title("Current Folder")
        folder_row.set_subtitle(self.music_dir or "No folder selected")
        folder_row.set_subtitle_selectable(True)

        change_btn = Gtk.Button(label="Change…")
        change_btn.add_css_class("pill")
        change_btn.set_valign(Gtk.Align.CENTER)
        change_btn.connect("clicked", lambda _: (
            dlg.close(),
            GLib.idle_add(lambda: Gtk.FileDialog.new().select_folder(
                self.win, None, self._folder_chosen) or False),
        ))
        folder_row.add_suffix(change_btn)

        forget_btn = Gtk.Button(label="Forget")
        forget_btn.add_css_class("pill")
        forget_btn.set_valign(Gtk.Align.CENTER)
        forget_btn.connect("clicked", lambda _: (
            self._forget_folder(None),
            folder_row.set_subtitle("No folder selected"),
        ))
        folder_row.add_suffix(forget_btn)

        folder_group.add(folder_row)
        page.add(folder_group)

        ws_group = Adw.PreferencesGroup()
        ws_group.set_title("Workspace")
        ws_group.set_description(
            "Where Sift stores your liked list, trash list, session state, and statistics. "
            "Defaults to ~/Music/sift-workspace."
        )

        ws_row = Adw.ActionRow()
        ws_row.set_title("Current Workspace")
        ws_row.set_subtitle(self.workspace)
        ws_row.set_subtitle_selectable(True)

        ws_change_btn = Gtk.Button(label="Change…")
        ws_change_btn.add_css_class("pill")
        ws_change_btn.set_valign(Gtk.Align.CENTER)
        ws_change_btn.connect("clicked", lambda _: (
            dlg.close(),
            GLib.idle_add(lambda: self._pick_workspace() or False),
        ))
        ws_row.add_suffix(ws_change_btn)

        ws_reset_btn = Gtk.Button(label="Reset")
        ws_reset_btn.add_css_class("pill")
        ws_reset_btn.set_valign(Gtk.Align.CENTER)
        ws_reset_btn.connect("clicked", lambda _: (
            self._reset_workspace(),
            ws_row.set_subtitle(self.workspace),
        ))
        ws_row.add_suffix(ws_reset_btn)

        ws_group.add(ws_row)
        page.add(ws_group)

        dlg.add(page)
        dlg.present(self.win)

    def _show_shortcuts(self):
        section = Gtk.ShortcutsSection(section_name="main", title="Sift")
        section.set_property("max-height", 12)

        group = Gtk.ShortcutsGroup(title="Judging")
        for title, accel in [
            ("Keep song",        "Right"),
            ("Trash song",       "Left"),
            ("Skip song",        "Down"),
            ("Undo last action", "<ctrl>z"),
        ]:
            group.append(Gtk.ShortcutsShortcut(title=title, accelerator=accel))
        section.append(group)

        group2 = Gtk.ShortcutsGroup(title="Playback")
        group2.append(Gtk.ShortcutsShortcut(title="Play / pause", accelerator="space"))
        section.append(group2)

        group3 = Gtk.ShortcutsGroup(title="Navigation")
        for title, accel in [
            ("Song info",         "i"),
            ("Library dashboard", "<ctrl>d"),
            ("Change folder",     "<ctrl>o"),
        ]:
            group3.append(Gtk.ShortcutsShortcut(title=title, accelerator=accel))
        section.append(group3)

        shortcuts_window = Gtk.ShortcutsWindow(child=section)
        shortcuts_window.set_transient_for(self.win)
        shortcuts_window.set_application(self)
        shortcuts_window.present()

    def _show_about(self):
        dlg = Adw.AboutDialog(
            application_name="Sift",
            application_icon="io.github.IdleEndeavor.Sift",
            developer_name="IdleEndeavor",
            version="1.3.0",
            comments="Tinder for Your Music Library",
            website="https://github.com/IdleEndeavor/sift_music_sorter",
            issue_url="https://github.com/IdleEndeavor/sift_music_sorter/issues",
            license_type=Gtk.License.GPL_3_0,
        )
        dlg.add_acknowledgement_section(
            "Libraries",
            [
                "librosa https://librosa.org",
                "mutagen https://mutagen.readthedocs.io",
                "soundfile https://pysoundfile.readthedocs.io",
                "NumPy https://numpy.org",
                "send2trash https://github.com/arsenetar/send2trash",
                "requests https://requests.readthedocs.io",
            ],
        )
        dlg.add_acknowledgement_section(
            "Built with",
            [
                "GTK4 https://gtk.org",
                "Libadwaita https://gnome.pages.gitlab.gnome.org/libadwaita",
                "GStreamer https://gstreamer.freedesktop.org",
                "LRCLIB https://lrclib.net",
            ],
        )
        dlg.present(self.win)


    # ── Library indexing ──────────────────────────────────────────────────────

    def _index_library(self, resume_idx: int = 0):
        exts  = (".flac", ".mp3", ".wav", ".ogg", ".m4a", ".opus")
        files = []
        for root, _, names in os.walk(self.music_dir):
            for n in names:
                if n.lower().endswith(exts):
                    full = os.path.join(root, n)
                    if full not in self.liked and full not in self.trash:
                        files.append(full)
        files.sort(key=os.path.getatime)
        self.queue = files
        self.idx   = min(resume_idx, max(0, len(files) - 1))
        self._analyse_ahead()

    def _analyse_ahead(self):
        """Background: analyse the next 10 un-cached songs.
        Skipped entirely in new music mode."""
        if self._new_music_mode:
            return

        def work():
            count = 0
            for i in range(self.idx, len(self.queue)):
                if count >= 10:
                    break
                path = self.queue[i]
                if path in self.cache or path in self.pending:
                    continue
                self.pending.add(path)
                count += 1
                try:
                    title, artist, dur = read_tags(path)
                    y, sr = librosa.load(path, sr=SR, mono=True, duration=150)
                    if y.ndim > 1:
                        y = librosa.to_mono(y)
                    dur   = dur or librosa.get_duration(y=y, sr=sr)
                    start, method = find_start(path, title, artist, y, sr, dur)
                    clip  = extract_clip(path, start)
                    self.cache[path] = (start, clip, method)
                except Exception as e:
                    print(f"[analysis] {os.path.basename(path)}: {e}")
                    self.cache[path] = (0.0, None, "error")
                finally:
                    self.pending.discard(path)
        threading.Thread(target=work, daemon=True).start()


    # ── Song loading ──────────────────────────────────────────────────────────

    def _load_song(self):
        # Update the player header title to reflect current mode
        mode_label = "New Music" if self._new_music_mode else "Judge Library"
        self._player_title_lbl.set_text(mode_label)

        if self.idx >= len(self.queue):
            self.player.set_state(Gst.State.NULL)
            if self._new_music_mode:
                self._new_music_mode = False
                self.music_dir = ""
                self._forget_folder(None)
                self._toast("All new music judged!", timeout=3)
                GLib.timeout_add(1500, lambda: (
                    self.stack.set_visible_child_name("setup"), False))
            else:
                self.lbl_title.set_text("All done!")
                self.lbl_artist.set_text("Your library is sifted ✓")
                self.lbl_counter.set_text("")
                self.lbl_method.set_text("")
                self.cover_pic.set_paintable(None)
                self._toast("Library fully sifted!", timeout=4)
            return

        path      = self.queue[self.idx]
        remaining = len(self.queue) - self.idx
        self.lbl_counter.set_text(
            f"{remaining} song{'s' if remaining != 1 else ''} left")

        self.lbl_method.set_text("full song" if self._new_music_mode else "analysing…")

        title, artist, _ = read_tags(path)
        self.lbl_title.set_text(title)
        self.lbl_artist.set_text(artist or "Unknown Artist")

        cover = _cover_bytes(path)
        if cover:
            try:
                tex = Gdk.Texture.new_from_bytes(GLib.Bytes.new(cover))
                self.cover_pic.set_paintable(tex)
            except Exception:
                self.cover_pic.set_paintable(None)
        else:
            self.cover_pic.set_paintable(None)

        self.spectro.load(path)

        self.player.set_state(Gst.State.NULL)
        self.player.set_property("uri", safe_uri(path))
        self.player.set_state(Gst.State.PLAYING)
        self._set_play_icon(True)

        if self._new_music_mode:
            # Play from the beginning — no chorus seeking
            pass
        else:
            self._wait_for_analysis(path, 0)
            save_state(self._state_file, {"dir": self.music_dir, "index": self.idx})


    # ── Playback position ─────────────────────────────────────────────────────

    def _tick_position(self) -> bool:
        ok_d, dur = self.player.query_duration(Gst.Format.TIME)
        ok_p, pos = self.player.query_position(Gst.Format.TIME)
        if ok_d and ok_p:
            d, p = dur / Gst.SECOND, pos / Gst.SECOND
            with self._no_seek_signal():
                self.seek.set_range(0, d)
                self.seek.set_value(p)
            self.lbl_pos.set_text(self._fmt(p))
            self.lbl_dur.set_text(self._fmt(d))
            self.spectro.set_pos(p, d)
        return True

    def _seek_manual(self, _s):
        self.player.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
            int(self.seek.get_value() * Gst.SECOND))

    @contextmanager
    def _no_seek_signal(self):
        self.seek.handler_block(self._seek_id)
        try:
            yield
        finally:
            self.seek.handler_unblock(self._seek_id)


    # ── Chorus seeking ────────────────────────────────────────────────────────

    def _wait_for_analysis(self, path: str, attempts: int):
        if not self.queue or self.idx >= len(self.queue) \
                or self.queue[self.idx] != path:
            return
        if path in self.cache:
            start, clip, method = self.cache[path]
            GLib.idle_add(lambda: self.lbl_method.set_text(f"via {method}") or False)
            if start > 0:
                self.seek.add_mark(start, Gtk.PositionType.BOTTOM, None)
                GLib.timeout_add(700, lambda: self._seek_to(path, start, clip) or False)
            return
        if attempts >= 20:
            self.cache[path] = (0.0, None, "timeout")
            GLib.idle_add(lambda: self.lbl_method.set_text("no chorus found") or False)
            return
        GLib.timeout_add(500,
            lambda: self._wait_for_analysis(path, attempts + 1) or False)

    def _seek_to(self, path: str, start: float, clip: str | None):
        if not self.queue or self.idx >= len(self.queue) \
                or self.queue[self.idx] != path:
            return
        ok = self.player.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
            int(start * Gst.SECOND))
        if not ok and clip and os.path.exists(clip):
            self.player.set_state(Gst.State.NULL)
            self.player.set_property("uri", safe_uri(clip))
            self.player.set_state(Gst.State.PLAYING)


    # ── Playback controls ─────────────────────────────────────────────────────

    def _toggle_play(self):
        _, state, _ = self.player.get_state(0)
        playing = state == Gst.State.PLAYING
        self.player.set_state(Gst.State.PAUSED if playing else Gst.State.PLAYING)
        self._set_play_icon(not playing)

    def _set_play_icon(self, playing: bool):
        icon = ("media-playback-pause-symbolic" if playing
                else "media-playback-start-symbolic")
        self.play_btn.get_child().set_from_icon_name(icon)


    # ── Judging actions ───────────────────────────────────────────────────────

    def _action(self, kind: str):
        if self.idx >= len(self.queue):
            return
        self._commit(kind)

    def _commit(self, kind: str):
        if self.idx >= len(self.queue):
            return
        path = self.queue[self.idx]
        if kind == "heart":
            self.liked.add(path)
            save_set(self._liked_file, self.liked)
        elif kind == "trash":
            self.trash.add(path)
            save_set(self._trash_file, self.trash)
        self._record_decision(kind, path)
        self.history.append((kind, path))
        self._next()

    def _undo(self):
        if not self.history:
            self._toast("Nothing to undo")
            return
        kind, path = self.history.pop()
        if kind == "heart":
            self.liked.discard(path)
            save_set(self._liked_file, self.liked)
        elif kind == "trash":
            self.trash.discard(path)
            save_set(self._trash_file, self.trash)

        # Reverse the stats record
        mode = self._mode_key()
        self._stats[mode]["judged"] = max(0, self._stats[mode]["judged"] - 1)
        if kind == "heart":
            self._stats[mode]["kept"] = max(0, self._stats[mode]["kept"] - 1)
        elif kind == "trash":
            self._stats[mode]["trashed"] = max(0, self._stats[mode]["trashed"] - 1)
        elif kind == "skip":
            self._stats[mode]["skipped"] = max(0, self._stats[mode]["skipped"] - 1)
        save_stats(self._stats_file, self._stats)

        self.idx = max(0, self.idx - 1)
        self.seek.clear_marks()
        self._toast(f"Undid {kind}")
        self._load_song()

    def _next(self):
        self.idx += 1
        self.seek.clear_marks()
        self._analyse_ahead()
        self._load_song()


    # ── Folder management ─────────────────────────────────────────────────────

    def _pick_folder_for_mode(self):
        """Open folder picker using whichever mode toggle is active."""
        self._new_music_mode = self._mode_new_btn.get_active()
        Gtk.FileDialog.new().select_folder(self.win, None, self._folder_chosen)

    def _folder_chosen(self, dialog, result):
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error as e:
            print(f"[folder] {e}")
            return
        if not folder:
            return
        self.music_dir = folder.get_path()
        self._toast(f"Loaded {os.path.basename(self.music_dir)}")

        if self._new_music_mode:
            self._index_library(resume_idx=0)
        else:
            resume = self._saved_idx if self.music_dir == self._saved_dir else 0
            self._index_library(resume_idx=resume)

        if self.queue:
            self.stack.set_visible_child_name("player")
            self._load_song()
        else:
            self._toast("No music files found in that folder")

    def _go_setup(self):
        self.player.set_state(Gst.State.NULL)
        self._new_music_mode = False
        self.stack.set_visible_child_name("setup")

    def _forget_folder(self, _btn):
        self._saved_dir = ""
        self._saved_idx = 0
        save_state(self._state_file, {})
        if _btn is not None:
            self._toast("Saved folder cleared")


    # ── Toast helper ──────────────────────────────────────────────────────────

    def _toast(self, message: str, timeout: int = 2):
        self.toast_overlay.add_toast(Adw.Toast(title=message, timeout=timeout))


    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _key(self, _c, kv, _kc, state):
        ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)
        if   kv == Gdk.KEY_space:        self._toggle_play()
        elif kv == Gdk.KEY_Left:         self._action("trash")
        elif kv == Gdk.KEY_Right:        self._action("heart")
        elif kv == Gdk.KEY_Down:         self._action("skip")
        elif kv == Gdk.KEY_i:            self._show_info()
        elif ctrl and kv == Gdk.KEY_z:   self._undo()
        elif ctrl and kv == Gdk.KEY_d:   self._open_dashboard()
        elif ctrl and kv == Gdk.KEY_o:   self._go_setup()
        else:                            return False
        return True


    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _mkbtn(icon: str, classes: list, tip: str) -> Gtk.Button:
        b = Gtk.Button()
        b.set_child(Gtk.Image.new_from_icon_name(icon))
        for c in classes:
            b.add_css_class(c)
        b.set_tooltip_text(tip)
        return b

    @staticmethod
    def _spacer(height: int) -> Gtk.Box:
        s = Gtk.Box()
        s.set_size_request(-1, height)
        return s

    @staticmethod
    def _fmt(s: float) -> str:
        s = int(s)
        return f"{s // 60}:{s % 60:02d}"


if __name__ == "__main__":
    Sift().run(None)