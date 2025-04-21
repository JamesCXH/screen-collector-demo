"""
screen_action_recorder.py
─────────────────────────
Record a macOS screen‑capture video (via ffmpeg) **plus** a timestamped JSON
log of critical user actions:

* left_click, right_click, **ctrl_left_click**
* left_drag   (press–hold–move–release)
* command+…   key combos (⌘‑C, ⌘‑V, etc.)
* typing      (collapsed into phrases separated by > TYPING_GAP seconds)
  * now ALSO begins when **Enter**, **Backspace/Delete**, or any printable
    character is pressed, and those special keys are represented literally in
    the recorded phrase ("\n" for Enter, "<BACKSPACE>", "<DELETE>").

Enable `--debug` to snapshot full‑screen PNGs at the *start* **and** *end* of
every logged action.  Screenshots are stored as

    {i}_{phase}_{timestamp‑ms}.png
       ↑  ↑       ↑ milliseconds since recording began
       │  └── ('start' | 'end')
       └──── index of the action (1‑based)

Usage
-----
    python screen_action_recorder.py               # normal session
    python screen_action_recorder.py --debug       # with screenshots
    python screen_action_recorder.py -f 30         # 30 fps recording, DEFAULTS TO 30

Dependencies
------------
* `brew install ffmpeg`
* `pip install pynput`
* macOS‑only: built‑in `screencapture` (grant Screen‑Recording + Input‑Monitoring)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from pynput import keyboard, mouse

TYPING_GAP = 5.0          # inactivity (s) that ends a typing phrase
RECORDINGS_ROOT = Path("recordings")

# ---------------------------------------------------------------------------
# Keys that should COUNT as characters for typing‑phrase detection even though
# they are not printable.  The mapping is Key → literal representation that
# gets appended to the phrase.
# ---------------------------------------------------------------------------
SPECIAL_TYPING_KEYS: dict[keyboard.Key, str] = {
    keyboard.Key.enter: "\n",         # newline
    keyboard.Key.backspace: "<BACKSPACE>",
    keyboard.Key.delete: "<DELETE>",
}

# ───────────────────────────── ffmpeg helpers ────────────────────────────────

def ffmpeg_record_cmd(temp_path: Path, fps: int = 30) -> list[str]:
    """Record the main display to MKV (safer than MP4 while writing)."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",                          # overwrite silently
        "-f", "avfoundation",
        "-framerate", str(fps),
        "-i", "1:none",                # display 1, no audio
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(temp_path),
    ]


def ffmpeg_remux_cmd(temp_path: Path, out_path: Path) -> list[str]:
    """Fast‑remux MKV → MP4 (adds fast‑start for web players)."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(temp_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]


# ─────────────────────────────── data models ────────────────────────────────

@dataclass
class Action:
    kind: str          # e.g. "left_click", "left_drag", …
    description: str   # human‑readable detail
    start: float       # seconds since recording origin
    end: float         # seconds since recording origin

# ───────────────────────────── action tracking ──────────────────────────────

class ActionTracker:
    """Aggregates critical user actions and (optionally) screenshots."""

    def __init__(self, origin_ts: float, *, debug: bool = False, screenshots_dir: Path | None = None):
        self.origin = origin_ts
        self.actions: list[Action] = []

        # typing state
        self._typing_start: float | None = None
        self._last_key_ts: float | None = None
        self._typed_chars: list[str] = []
        self._typing_id: int | None = None

        # modifier‑key tracking
        self._command_down = False
        self._ctrl_down = False

        # drag tracking: btn → (x, y, t_press, ss_idx)
        self._drag_start: dict[str, tuple[int, int, float, int]] = {}


        # misc
        self.debug = debug
        self._ss_dir = screenshots_dir
        self._ss_idx = 1
        self._stop_check = threading.Event()
        threading.Thread(target=self._flush_checker, daemon=True).start()

    # ─────────── helpers ───────────
    def _next_idx(self) -> int:
        i = self._ss_idx
        self._ss_idx += 1
        return i

    def _capture(self, idx: int, phase: str, rel_ts: float) -> None:
        """Capture a screenshot if debug mode is on."""
        if not self.debug or self._ss_dir is None:
            return
        ts_ms = int(rel_ts * 1000)
        out = self._ss_dir / f"{idx}_{phase}_{ts_ms}.png"
        subprocess.run(["screencapture", "-x", str(out)])  # macOS builtin

    # ─────────── mouse listener ───────────
    def on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        now = time.time()

        # ── BUTTON PRESS ───────────────────────────────────────────────────────
        if pressed:
            # Detect what kind of click this *might* become …
            if button == mouse.Button.left and self._ctrl_down:
                kind = "ctrl_left_click"
            elif button == mouse.Button.left:
                kind = None  # could turn into a drag
            elif button == mouse.Button.right:
                kind = "right_click"
            else:
                return

            # ── Allocate an action index *now* and snapshot the screen ─────────
            idx = self._next_idx()
            rel_t = now - self.origin
            self._capture(idx, "start", rel_t)

            # Track potential drag (only plain left‑button press)
            if button == mouse.Button.left and kind is None:
                # save idx so we can finish the action on release
                self._drag_start["left"] = (x, y, now, idx)
                return

            # Immediate click logging (right‑ or ctrl‑click)
            self._finalize_typing(now)
            self.actions.append(Action(kind, f"{kind} @({x},{y})", rel_t, rel_t))
            self._capture(idx, "end", rel_t)
            return

        # ── BUTTON RELEASE ──────────────────────────────────────────────────────
        if button == mouse.Button.left and "left" in self._drag_start:
            sx, sy, t0, idx = self._drag_start.pop("left")
            moved = (sx, sy) != (x, y)
            kind = "left_drag" if moved else "left_click"

            self._finalize_typing(now)
            rel_start = t0 - self.origin
            rel_end = now - self.origin
            if moved:
                desc = f"{kind} ({sx},{sy}) → ({x},{y})"
            else:
                desc = f"{kind} @({x},{y})"
            self.actions.append(Action(kind, desc, rel_start, rel_end))
            self._capture(idx, "end", rel_end)

    # ─────────── keyboard listener ───────────
    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        now = time.time()

        # track modifier states
        if key in {keyboard.Key.cmd, keyboard.Key.cmd_r}:
            self._command_down = True
            return
        if key in {keyboard.Key.ctrl, keyboard.Key.ctrl_r}:
            self._ctrl_down = True
            return

        # command+… combo (e.g. ⌘‑C)
        if self._command_down:
            desc = f"command+{self._key_name(key)}"
            self._finalize_typing(now)
            rel_t = now - self.origin
            idx = self._next_idx()
            self._capture(idx, "start", rel_t)
            self.actions.append(Action("command_combo", desc, rel_t, rel_t))
            self._capture(idx, "end", rel_t)
            return

        # printable character → typing mode
        if hasattr(key, "char") and key.char and key.char.isprintable():
            self._begin_or_continue_typing(now, key.char)
            return

        # special non‑printable character that we still want to record
        if key in SPECIAL_TYPING_KEYS:
            self._begin_or_continue_typing(now, SPECIAL_TYPING_KEYS[key])
            return

    def _begin_or_continue_typing(self, ts: float, token: str) -> None:
        """Either starts a typing phrase or appends to the current one."""
        if self._typing_start is None:
            self._typing_start = ts
            self._typing_id = self._next_idx()
            self._capture(self._typing_id, "start", ts - self.origin)
        self._typed_chars.append(token)
        self._last_key_ts = ts

    def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key in {keyboard.Key.cmd, keyboard.Key.cmd_r}:
            self._command_down = False
        if key in {keyboard.Key.ctrl, keyboard.Key.ctrl_r}:
            self._ctrl_down = False

    # ─────────── typing flushing ───────────
    def _flush_checker(self) -> None:
        """Background thread: close a typing phrase after TYPING_GAP silence."""
        while not self._stop_check.is_set():
            time.sleep(0.5)
            if (
                self._typing_start is not None
                and self._last_key_ts is not None
                and (time.time() - self._last_key_ts) > TYPING_GAP
            ):
                self._finalize_typing(time.time())

    def _finalize_typing(self, ts: float) -> None:
        """Commit current typing phrase, if any."""
        if self._typing_start is None:
            return
        phrase = "".join(self._typed_chars)
        if phrase:
            self.actions.append(
                Action(
                    "typing",
                    f"typed: '{phrase}'",
                    self._typing_start - self.origin,
                    self._last_key_ts - self.origin,
                )
            )
        if self._typing_id is not None:
            self._capture(self._typing_id, "end", self._last_key_ts - self.origin)
        # reset
        self._typing_start = self._last_key_ts = None
        self._typed_chars.clear()
        self._typing_id = None

    def stop(self) -> None:
        """Stop the tracker and flush pending typing."""
        self._stop_check.set()
        self._finalize_typing(time.time())

    # ─────────── utility ───────────
    @staticmethod
    def _key_name(key: keyboard.Key | keyboard.KeyCode) -> str:
        return key.char if hasattr(key, "char") and key.char else str(key)


# ──────────────────────────────────── main ───────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Record screen + critical actions → MP4")
    ap.add_argument("-f", "--fps", type=int, default=30, help="frames per second")
    ap.add_argument("--debug", action="store_true", help="capture start/end screenshots for every action")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install via Homebrew: brew install ffmpeg")
    if args.debug and not shutil.which("screencapture"):
        sys.exit("macOS 'screencapture' utility not found (should be pre‑installed)")

    # ── session folder bookkeeping ───────────────────────────────────────────
    RECORDINGS_ROOT.mkdir(exist_ok=True)
    session_idx = 1
    while (RECORDINGS_ROOT / str(session_idx)).exists():
        session_idx += 1

    session_dir = RECORDINGS_ROOT / str(session_idx)
    screenshots_dir = session_dir / "screenshots"
    session_dir.mkdir(parents=True)
    if args.debug:
        screenshots_dir.mkdir(parents=True)

    outfile = session_dir / "recording.mp4"
    temp_mkv = outfile.with_suffix(".capture.mkv")

    # ── start recording ──────────────────────────────────────────────────────
    origin = time.time()
    tracker = ActionTracker(
        origin,
        debug=args.debug,
        screenshots_dir=screenshots_dir if args.debug else None,
    )
    rec_proc = subprocess.Popen(
        ffmpeg_record_cmd(temp_mkv, args.fps),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Recording to {session_dir} …  (Esc to stop)")

    stop_event = threading.Event()

    def _on_release(key):
        tracker.on_release(key)
        if key == keyboard.Key.esc:
            stop_event.set()
            return False

    with mouse.Listener(on_click=tracker.on_click) as _ml, keyboard.Listener(
        on_press=tracker.on_press,
        on_release=_on_release,
    ) as _kl:
        while not stop_event.is_set():
            time.sleep(0.1)

    # ── stop recording ───────────────────────────────────────────────────────
    tracker.stop()
    try:
        rec_proc.stdin.write(b"q\n")
        rec_proc.stdin.flush()
    except Exception:
        pass
    rec_proc.wait()

    # ── remux to MP4 for broad compatibility ────────────────────────────────
    print("Finalizing video …")
    subprocess.run(ffmpeg_remux_cmd(temp_mkv, outfile), check=True)
    temp_mkv.unlink(missing_ok=True)

    # ── save action log ──────────────────────────────────────────────────────
    json_path = outfile.with_suffix(".json")
    json_path.write_text(json.dumps([asdict(a) for a in tracker.actions], indent=2))

    # ── summary ──────────────────────────────────────────────────────────────
    print("\nHigh‑level actions:")
    for a in tracker.actions:
        print(f"{a.kind:<15} {a.start:8.2f}s – {a.end:8.2f}s | {a.description}")

    print(f"\nSaved video        → {outfile}\nSaved action log   → {json_path}")
    if args.debug:
        print(f"Screenshots folder → {screenshots_dir}")


if __name__ == "__main__":
    main()
