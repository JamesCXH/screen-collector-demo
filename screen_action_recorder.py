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
    keyboard.Key.tab:       "\t",
    keyboard.Key.space:     " "
}

# ───────────────────────────── ffmpeg helpers ────────────────────────────────


def ffmpeg_record_cmd(
    temp_path: Path,
    fps: int = 30,
    *,
    overlay_time: bool = False,
) -> list[str]:
    """
    Build an ffmpeg command that records the main display to `temp_path`.
    When `overlay_time` is True, a drawtext filter stamps the running
    timestamp (HH:MM:SS.xx) in **bold red** at the top‑right.

    The function tries a handful of bold system fonts; if none are found
    it falls back to the generic family name “Helvetica‑Bold”.
    """
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",                       # overwrite silently
        "-f",
        "avfoundation",
        "-framerate",
        str(fps),
        "-i",
        "1:none",                   # display 1, no audio
        "-vf",
        f"fps={fps}",
        # "-vsync",
        # "cfr",
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
    ]

    if overlay_time:
        # Look for a bold monospaced/system font.
        bold_fonts = [
            "/System/Library/Fonts/SFNSMono-Bold.ttf",
            "/System/Library/Fonts/SFNSText-Bold.otf",
            "/System/Library/Fonts/SFNSDisplay-Bold.otf",
            "/System/Library/Fonts/Menlo.ttc",
        ]
        font_path = next((p for p in bold_fonts if Path(p).exists()), "")

        drawtext = (
            "drawtext="
            f"{('fontfile=' + font_path + ':') if font_path else 'font=Helvetica-Bold:'}"
            "text='%{pts\\:hms}'"
            ":fontsize=48"
            ":fontcolor=red"
            ":x=w-tw-10"
            ":y=10"
            ":box=1"
            ":boxcolor=black@0.5"
        )
        cmd.extend(["-vf", drawtext])

    cmd.append(str(temp_path))
    return cmd


def ffmpeg_remux_cmd(temp_path: Path, out_path: Path) -> list[str]:
    """Fast‑remux MKV → MP4 (adds fast‑start for web players)."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-copytb",
        "1",
        "-i",
        str(temp_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

def ffmpeg_encode_cmd(
    temp_path: Path,
    out_path: Path,
    *,
    fps: int = 30,
    crf: int = 20,
    preset: str = "veryfast",
) -> list[str]:
    """
    Re‑encode the capture so every frame has monotonic timestamps and a
    constant frame‑rate stream that plays reliably in all browsers.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-fflags", "+genpts",       # rebuild presentation timestamps
        "-i", str(temp_path),

        # ── video ───────────────────────────────────────────────────────────
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-r", str(fps),             # force constant FPS

        # ── fast‑start ──────────────────────────────────────────────────────
        "-movflags", "+faststart",
        str(out_path),
    ]



# ─────────────────────────────── data models ────────────────────────────────

@dataclass
class Action:
    kind: str          # e.g. "left_click", "key_combo", …
    description: str   # human‑readable detail
    start: float       # seconds since recording origin
    end: float         # seconds since recording origin
    ss_idx: int        # 1‑based screenshot index



# ───────────────────────────── action tracking ──────────────────────────────


class ActionTracker:
    """Aggregates critical user actions and (optionally) screenshots."""

    # definitive ordering for modifier names (guarantees no collisions)
    _MOD_ORDER = ("ctrl", "command", "option", "shift")

    def __init__(
        self,
        origin_ts: float,
        *,
        debug: bool = False,
        screenshots_dir: Path | None = None,
    ):
        self.origin = origin_ts
        self.actions: list[Action] = []

        # typing state --------------------------------------------------------
        self._typing_start: float | None = None
        self._last_key_ts: float | None = None
        self._typed_chars: list[str] = []
        self._typing_id: int | None = None

        # modifier‑key tracking ----------------------------------------------
        self._command_down = False
        self._ctrl_down = False
        self._shift_down = False
        self._option_down = False

        # active “modifier‑hold” block: (mods, t_start, ss_idx, used)
        self._active_mod_block: tuple[frozenset[str], float, int, bool] | None = None

        # click tracking: btn → (x, y, t_press, ss_idx, mods) -----------------
        self._click_start: dict[str, tuple[int, int, float, int, frozenset[str]]] = {}

        # misc ----------------------------------------------------------------
        self.debug = debug
        self._ss_dir = screenshots_dir
        self._ss_idx = 1

        # NEW: queue of (idx, phase, rel_ts) screenshots to extract later -----
        self._ss_requests: list[tuple[int, str, float]] = []

        self._stop_check = threading.Event()
        self._typing_lock = threading.Lock()
        threading.Thread(target=self._flush_checker, daemon=True).start()

    # ─────────── helpers ───────────
    def _next_idx(self) -> int:
        i = self._ss_idx
        self._ss_idx += 1
        return i

    def _capture(self, idx: int, phase: str, rel_ts: float) -> None:
        """
        Queue a screenshot *request* if debug mode is active.
        Extraction happens after the video is finalised so every PNG
        is taken from the exact frame whose timestamp we snap below.
        """
        if not self.debug or self._ss_dir is None:
            return
        self._ss_requests.append((idx, phase, rel_ts))

    # -----------------------------------------------------------------------
    #  Modifier helpers
    # -----------------------------------------------------------------------
    def _update_modifier_state(
        self, key: keyboard.Key | keyboard.KeyCode, down: bool
    ) -> bool:
        """
        Set internal modifier flags; return True if `key` *was* a modifier
        and therefore should NOT be processed as a normal key event.
        """
        if key in {keyboard.Key.cmd, keyboard.Key.cmd_r}:
            self._command_down = down
            return True
        if key in {keyboard.Key.ctrl, keyboard.Key.ctrl_r}:
            self._ctrl_down = down
            return True
        if key in {keyboard.Key.shift, keyboard.Key.shift_r}:
            self._shift_down = down
            return True
        if key in {keyboard.Key.alt, keyboard.Key.alt_r}:
            self._option_down = down
            return True
        return False

    def _current_modifiers(self) -> list[str]:
        mods: list[str] = []
        if self._ctrl_down:
            mods.append("ctrl")
        if self._command_down:
            mods.append("command")
        if self._option_down:
            mods.append("option")
        if self._shift_down:
            mods.append("shift")
        return mods

    # -----------------------------------------------------------------------
    #  Modifier‑block helpers
    # -----------------------------------------------------------------------
    def _sorted_mods(self, mods: frozenset[str]) -> list[str]:
        """Return a list of modifiers sorted by the canonical order."""
        return sorted(mods, key=self._MOD_ORDER.index)

    def _mark_mod_block_used(self) -> None:
        """Flag that a non‑modifier key was pressed during the active mod hold."""
        if self._active_mod_block is not None:
            mods, t0, idx, _ = self._active_mod_block
            self._active_mod_block = (mods, t0, idx, True)

    def _maybe_rollover_modifier_block(self, ts: float) -> None:
        """
        Close/open a ‘modifier_hold’ block whenever the set of pressed
        modifiers changes. A block is only recorded if **some** non‑modifier
        key was pressed while the modifier(s) were held.
        """
        current = frozenset(self._current_modifiers())

        # nothing active before, and still nothing → nothing to do
        if self._active_mod_block is None and not current:
            return

        # first modifier pressed OR combination changed OR all released
        if (
            self._active_mod_block is None
            or current != self._active_mod_block[0]
        ):
            # close previous block (if any) ----------------------------------
            if self._active_mod_block is not None:
                prev_mods, t0, idx, used = self._active_mod_block
                if used:  # ← only record if meaningful
                    rel_start = t0 - self.origin
                    rel_end = ts - self.origin
                    desc = "+".join(self._sorted_mods(prev_mods)) + " held"
                    self.actions.append(
                        Action("modifier_hold", desc, rel_start, rel_end, idx)
                    )
                    self._capture(idx, "end", rel_end)
                self._active_mod_block = None

            # open new block (if any modifiers still down) -------------------
            if current:
                idx = self._next_idx()
                self._capture(idx, "start", ts - self.origin)
                # `used` flag starts as False
                self._active_mod_block = (current, ts, idx, False)

    # -----------------------------------------------------------------------
    #  Mouse listener
    # -----------------------------------------------------------------------
    def on_click(
        self, x: int, y: int, button: mouse.Button, pressed: bool
    ) -> None:
        now = time.perf_counter()

        # BUTTON PRESS --------------------------------------------------------
        if pressed and button in (mouse.Button.left, mouse.Button.right):
            idx = self._next_idx()
            rel_t = now - self.origin
            self._capture(idx, "start", rel_t)

            mods = frozenset(self._current_modifiers())
            # store info so we can finish the action on release
            self._click_start[button.name] = (x, y, now, idx, mods)
            return

        # BUTTON RELEASE ------------------------------------------------------
        if (not pressed) and (button.name in self._click_start):
            sx, sy, t0, idx, mods = self._click_start.pop(button.name)

            rel_start = t0 - self.origin
            rel_end = now - self.origin

            kind = f"{button.name}_click"  # "left_click" | "right_click"
            mod_prefix = (
                "+".join(self._sorted_mods(mods)) + " "
                if mods
                else ""
            )
            desc = f"{mod_prefix}{kind} ({sx},{sy}) → ({x},{y})"

            self._finalize_typing(now)
            self.actions.append(Action(kind, desc, rel_start, rel_end, idx))
            self._capture(idx, "end", rel_end)

    # -----------------------------------------------------------------------
    #  Keyboard listener
    # -----------------------------------------------------------------------
    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        now = time.perf_counter()

        # first, deal with modifiers themselves ------------------------------
        if self._update_modifier_state(key, True):
            self._maybe_rollover_modifier_block(now)
            return  # pure modifier press → nothing more to do

        # At this point we have a NON‑modifier key. If a modifier block is
        # active, flag it as used so it will be recorded when finished.
        self._mark_mod_block_used()

        # Prepare helper values ----------------------------------------------
        mods = self._current_modifiers()
        is_combo_context = any(m in mods for m in ("command", "ctrl", "option"))

        # Produce key‑combo action if relevant -------------------------------
        if is_combo_context:
            desc = "+".join(mods + [self._key_name(key)])
            self._finalize_typing(now)
            rel_t = now - self.origin
            idx = self._next_idx()
            self._capture(idx, "start", rel_t)
            self.actions.append(Action("key_combo", desc, rel_t, rel_t, idx))
            self._capture(idx, "end", rel_t)
            return

        # printable character → typing mode ----------------------------------
        if hasattr(key, "char") and key.char and key.char.isprintable():
            self._begin_or_continue_typing(now, key.char)
            return

        # special non‑printable character we still want to record ------------
        if key in SPECIAL_TYPING_KEYS:
            self._begin_or_continue_typing(now, SPECIAL_TYPING_KEYS[key])
            return

    def _begin_or_continue_typing(self, ts: float, token: str) -> None:
        """Either starts a typing phrase or appends to the current one."""
        with self._typing_lock:
            if self._typing_start is None:
                self._typing_start = ts
                self._typing_id = self._next_idx()
                self._capture(self._typing_id, "start", ts - self.origin)
            self._typed_chars.append(token)
            self._last_key_ts = ts

    def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if self._update_modifier_state(key, False):
            self._maybe_rollover_modifier_block(time.perf_counter())

    # -----------------------------------------------------------------------
    #  Typing flushing
    # -----------------------------------------------------------------------
    def _flush_checker(self) -> None:
        """Background thread: close a typing phrase after TYPING_GAP silence."""
        while not self._stop_check.is_set():
            time.sleep(0.5)
            if (
                self._typing_start is not None
                and self._last_key_ts is not None
                and (time.perf_counter() - self._last_key_ts) > TYPING_GAP
            ):
                self._finalize_typing(time.perf_counter())

    def _finalize_typing(self, ts: float) -> None:
        """Commit current typing phrase, if any."""
        with self._typing_lock:
            if self._typing_start is None:
                return
            phrase = "".join(self._typed_chars)
            if phrase:
                rel_start = self._typing_start - self.origin
                rel_end = ts - self.origin
                self.actions.append(
                  Action("typing", f"typed: '{phrase}'", rel_start, rel_end, self._typing_id)
                )

            if self._typing_id is not None:
                self._capture(
                    self._typing_id,
                    "end",
                    ts - self.origin,
                )
            # reset --------------------------------------------------------------
            self._typing_start = self._last_key_ts = None
            self._typed_chars.clear()
            self._typing_id = None

    def stop(self) -> None:
        """Stop the tracker and flush pending typing."""
        self._stop_check.set()
        self._finalize_typing(time.perf_counter())
        # close still‑open modifier block, if any
        self._maybe_rollover_modifier_block(time.perf_counter())

    # -----------------------------------------------------------------------
    #  Utility
    # -----------------------------------------------------------------------
    @staticmethod
    def _key_name(key: keyboard.Key | keyboard.KeyCode) -> str:
        if hasattr(key, "char") and key.char:
            return key.char
        # e.g. 'Key.space' -> 'space'
        return str(key).split(".")[-1]


# ──────────────────────────────────── main ───────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record screen + critical actions → MP4"
    )
    ap.add_argument(
        "-f",
        "--fps",
        type=int,
        default=30,
        help="frames per second",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="capture screenshots and overlay a running timer on the video",
    )
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit(
            "ffmpeg not found. Install via Homebrew: brew install ffmpeg"
        )
    if args.debug and not shutil.which("screencapture"):
        # screencapture is no longer mandatory (we use ffmpeg), but keep the
        # check so users know it's not used any more.
        print("NOTE: macOS 'screencapture' is no longer required; continuing …")

    # session folder bookkeeping --------------------------------------------
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

    # start recording --------------------------------------------------------
    rec_proc = subprocess.Popen(
        ffmpeg_record_cmd(
            temp_mkv,
            args.fps,
            overlay_time=args.debug,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # ── block until the first frame has been written ────────────────────────
    while not temp_mkv.exists() or temp_mkv.stat().st_size == 0:
        time.sleep(0.005)

    origin = time.perf_counter()
    tracker = ActionTracker(
        origin,
        debug=args.debug,
        screenshots_dir=screenshots_dir if args.debug else None,
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

    # stop recording ---------------------------------------------------------
    tracker.stop()
    try:
        rec_proc.stdin.write(b"q\n")
        rec_proc.stdin.flush()
        rec_proc.wait(timeout=3)
    except Exception:
        rec_proc.terminate()
        rec_proc.wait()

    # remux to MP4 for broad compatibility -----------------------------------
    print("Finalizing video …")
    # subprocess.run(ffmpeg_remux_cmd(temp_mkv, outfile), check=True)
    subprocess.run(
        ffmpeg_encode_cmd(temp_mkv, outfile, fps=args.fps),
        check=True,
    )

    # ── obtain PTS of the very first frame ─────────────────────────────────--
    try:
        video_t0 = float(
            subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "frame=pkt_pts_time",
                    "-read_intervals", "%+#1",
                    "-of", "default=nokey=1:noprint_wrappers=1",
                    str(temp_mkv),
                ],
                text=True,
            ).strip()
        )
    except Exception:
        print("WARNING! FAILED TO GET FIRST TIMESTEP, TIMESTEPS WILL BE SLIGHTLY OUT OF SYNC")
        video_t0 = 0.0

    def _snap(t: float, fps: int, offset: float) -> float:
        """Round *t* to the closest frame boundary (1 / fps)."""
        return round((t - offset) * fps) / fps

    for a in tracker.actions:
        a.start = _snap(a.start, args.fps, video_t0)
        a.end   = _snap(a.end,   args.fps, video_t0)
    tracker.actions.sort(key=lambda a: a.ss_idx)

    # ------------------------------------------------------------------
    # Extract queued screenshots on perfect frame boundaries -----------
    # ------------------------------------------------------------------
    # if args.debug:
    #     screenshots_dir.mkdir(exist_ok=True)
    #     for idx, phase, rel_ts in tracker._ss_requests:
    #         ts = _snap(rel_ts, args.fps, video_t0)
    #         out_png = screenshots_dir / f"{idx}_{phase}_{int(ts * 1000):06d}.png"
    #         subprocess.run(
    #             [
    #                 "ffmpeg", "-hide_banner", "-loglevel", "error",
    #                 "-y",
    #                 "-ss", f"{ts:.03f}",
    #                 "-i", str(outfile),
    #                 "-frames:v", "1",
    #                 "-q:v", "2",
    #                 str(out_png),
    #             ],
    #             check=True,
    #         )
    # ------------------------------------------------------------------
    # Extract queued screenshots
    #   • “start”  → snap to the exact frame boundary  (frame‑accurate)
    #   • “end”    → leave at the exact wall‑clock timestamp (natural)
    # ------------------------------------------------------------------
    if args.debug:
        screenshots_dir.mkdir(exist_ok=True)

        for idx, phase, rel_ts in tracker._ss_requests:
            # `rel_ts` is seconds since we started tracking.
            # Video PTSs begin at `video_t0`, so subtract it first.
            raw_ts = max(rel_ts - video_t0, 0.0)

            if phase == "start":
                ts = _snap(rel_ts, args.fps, video_t0)  # frame‑accurate
            else:  # phase == "end"
                ts = raw_ts  # natural timestamp

            out_png = screenshots_dir / f"{idx}_{phase}_{int(ts * 1000):06d}.png"

            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-y",
                    "-ss", f"{ts:.03f}",  # ← new timestamp logic
                    "-i", str(outfile),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(out_png),
                ],
                check=True,
            )

    temp_mkv.unlink(missing_ok=True)

    # save action log --------------------------------------------------------
    json_path = outfile.with_suffix(".json")
    json_path.write_text(json.dumps([asdict(a) for a in tracker.actions], indent=2))

    # summary ----------------------------------------------------------------
    print("\nHigh‑level actions:")
    for a in tracker.actions:
        print(
            f"{a.kind:<15} {a.start:8.2f}s – {a.end:8.2f}s | {a.description}"
        )

    print(f"\nSaved video        → {outfile}\nSaved action log   → {json_path}")
    if args.debug:
        print(f"Screenshots folder → {screenshots_dir}")


if __name__ == "__main__":
    main()