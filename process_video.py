#!/usr/bin/env python3
"""
process_video.py
=================
100% free / open-source short-clip generator.

Pipeline:
  1. Read source video URLs from video_pool.txt (one direct .mp4/.mov URL per line)
  2. Download each video
  3. Analyze the audio track (RMS energy windows) to find the highest-energy
     "hook" moment (usually a shout, laugh, punchline, or emphasis spike)
  4. Cut a clip around that spike (default 45s, configurable)
  5. Center-crop 16:9 -> 9:16 vertical (1080x1920)
  6. Transcribe the clip locally with faster-whisper (no API key, fully offline)
     and burn word-synced captions onto the video with FFmpeg drawtext
  7. Save final clip + thumbnail into public/clips/
  8. Update public/clips/manifest.json so the frontend can list all clips
     without needing server-side directory listing (static hosting on Vercel
     can't list a folder, so we generate an index file instead)

Design notes / honesty about limits:
  - "Hook detection" here is a VOLUME/ENERGY heuristic, not true semantic
    understanding of what's interesting. It reliably finds loud, emphatic,
    or high-energy moments (shouts, laughs, dramatic pauses -> spikes),
    which is a reasonable free proxy for a "hook", but it will occasionally
    pick a loud non-hook moment (cough, noise, applause). For smarter
    selection you'd eventually want a speech/semantic model, which is out
    of scope for a 100%-free pipeline.
  - faster-whisper's "tiny"/"base" model is used by default to keep this
    runnable on GitHub Actions' free Linux runners without a GPU. Swap to
    "small"/"medium" for better accuracy if your runner has more time/RAM.

Dependencies (see requirements.txt):
  moviepy, ffmpeg-python, numpy, requests, faster-whisper, Pillow
System dependency: ffmpeg (installed by the GitHub Actions workflow)
"""

import os
import sys
import json
import math
import shutil
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from moviepy.editor import VideoFileClip

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
POOL_FILE = ROOT_DIR / "video_pool.txt"
PROCESSED_LOG = ROOT_DIR / ".processed_urls.json"   # prevents re-processing same URL every run
OUTPUT_DIR = ROOT_DIR / "public" / "clips"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
TMP_DIR = Path(tempfile.gettempdir()) / "clip_engine"

CLIP_DURATION_SEC = 45          # length of the extracted short
ENERGY_WINDOW_SEC = 1.0         # window size used to scan for audio spikes
TARGET_W, TARGET_H = 1080, 1920 # 9:16 vertical output
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")  # tiny/base/small
FONT_PATH = os.environ.get("CAPTION_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# STEP 1: I/O helpers
# --------------------------------------------------------------------------
def load_urls() -> list[str]:
    if not POOL_FILE.exists():
        print(f"[!] {POOL_FILE} not found. Nothing to do.")
        return []
    urls = [
        line.strip()
        for line in POOL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return urls


def load_processed() -> set[str]:
    if PROCESSED_LOG.exists():
        return set(json.loads(PROCESSED_LOG.read_text(encoding="utf-8")))
    return set()


def save_processed(processed: set[str]) -> None:
    PROCESSED_LOG.write_text(json.dumps(sorted(processed), indent=2), encoding="utf-8")


def download_video(url: str) -> Path:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or ".mp4"
    local_path = TMP_DIR / f"src_{uuid.uuid4().hex}{ext}"
    print(f"    ↓ downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return local_path


# --------------------------------------------------------------------------
# STEP 2: Audio-energy "hook" detection
# --------------------------------------------------------------------------
def find_hook_start(clip: VideoFileClip, clip_len: float) -> float:
    """
    Scans the audio track in ENERGY_WINDOW_SEC windows, computes RMS energy
    per window, and returns the start timestamp (seconds) of the window with
    the highest sustained energy -- centering the extracted clip on it.
    """
    if clip.audio is None:
        return 0.0

    duration = clip.duration
    fps_audio = 22050  # downsample for fast analysis
    audio_array = clip.audio.to_soundarray(fps=fps_audio, nbytes=2)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # mono-mix

    window_samples = int(ENERGY_WINDOW_SEC * fps_audio)
    if window_samples <= 0 or len(audio_array) < window_samples:
        return 0.0

    n_windows = len(audio_array) // window_samples
    energies = np.zeros(n_windows)
    for i in range(n_windows):
        seg = audio_array[i * window_samples: (i + 1) * window_samples]
        energies[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))  # RMS

    # Smooth to avoid picking a single-sample transient (e.g. a click/cough)
    kernel = np.ones(3) / 3
    smoothed = np.convolve(energies, kernel, mode="same")

    peak_window = int(np.argmax(smoothed))
    peak_time = peak_window * ENERGY_WINDOW_SEC

    # Center the clip window on the spike, clamped to video bounds
    start = max(0.0, peak_time - clip_len / 2)
    start = min(start, max(0.0, duration - clip_len))
    return start


# --------------------------------------------------------------------------
# STEP 3: Vertical crop (center-crop 16:9 -> 9:16)
# --------------------------------------------------------------------------
def crop_to_vertical(clip: VideoFileClip) -> VideoFileClip:
    w, h = clip.size
    target_ratio = TARGET_W / TARGET_H  # 0.5625

    # First scale so the shorter dimension covers the target, then crop the rest
    if w / h > target_ratio:
        # source is wider than target -> crop sides
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        cropped = clip.crop(x1=x1, y1=0, x2=x1 + new_w, y2=h)
    else:
        # source is taller/narrower -> crop top/bottom
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        cropped = clip.crop(x1=0, y1=y1, x2=w, y2=y1 + new_h)

    return cropped.resize((TARGET_W, TARGET_H))


# --------------------------------------------------------------------------
# STEP 4: Transcription (local, free) for caption burn-in
# --------------------------------------------------------------------------
def transcribe(audio_path: Path) -> list[dict]:
    """Returns list of {start, end, text} word/segment chunks."""
    if not WHISPER_AVAILABLE:
        print("    [!] faster-whisper not installed; skipping captions.")
        return []

    print(f"    ✎ transcribing with faster-whisper ({WHISPER_MODEL_SIZE}) ...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)

    chunks = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                chunks.append({"start": w.start, "end": w.end, "text": w.word.strip()})
        else:
            chunks.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
    return chunks


def group_words_into_caption_lines(words: list[dict], max_chars: int = 28, max_group_sec: float = 2.2):
    """Groups whisper word timestamps into short on-screen caption lines."""
    lines = []
    current = []
    current_len = 0
    group_start = None

    for w in words:
        if not w["text"]:
            continue
        if group_start is None:
            group_start = w["start"]

        would_be_len = current_len + len(w["text"]) + 1
        too_long_duration = (w["end"] - group_start) > max_group_sec

        if current and (would_be_len > max_chars or too_long_duration):
            lines.append({
                "start": group_start,
                "end": current[-1]["end"],
                "text": " ".join(x["text"] for x in current),
            })
            current = [w]
            current_len = len(w["text"])
            group_start = w["start"]
        else:
            current.append(w)
            current_len += len(w["text"]) + 1

    if current:
        lines.append({
            "start": group_start,
            "end": current[-1]["end"],
            "text": " ".join(x["text"] for x in current),
        })
    return lines


# --------------------------------------------------------------------------
# STEP 5: Burn captions with FFmpeg drawtext (bottom-center, high visibility)
# --------------------------------------------------------------------------
def build_drawtext_filters(caption_lines: list[dict]) -> str:
    filters = []
    for line in caption_lines:
        text = line["text"].replace("'", "\u2019").replace(":", "\\:")
        start, end = line["start"], line["end"]
        filters.append(
            "drawtext="
            f"fontfile='{FONT_PATH}':"
            f"text='{text}':"
            "fontcolor=white:fontsize=64:borderw=6:bordercolor=black:"
            "x=(w-text_w)/2:y=h-(h*0.16):"
            f"enable='between(t,{start:.2f},{end:.2f})'"
        )
    return ",".join(filters) if filters else None


def burn_captions(input_path: Path, output_path: Path, caption_lines: list[dict]) -> None:
    drawtext_chain = build_drawtext_filters(caption_lines)
    if not drawtext_chain:
        shutil.copy(input_path, output_path)
        return

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", drawtext_chain,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "128k",
        str(output_path),
    ]
    print("    🔥 burning captions with ffmpeg ...")
    subprocess.run(cmd, check=True, capture_output=True)


# --------------------------------------------------------------------------
# STEP 6: Thumbnail
# --------------------------------------------------------------------------
def save_thumbnail(video_path: Path, thumb_path: Path) -> None:
    cmd = ["ffmpeg", "-y", "-ss", "1", "-i", str(video_path), "-frames:v", "1", str(thumb_path)]
    subprocess.run(cmd, check=True, capture_output=True)


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
def process_single_video(url: str) -> dict | None:
    print(f"[+] Processing: {url}")
    src_path = None
    try:
        src_path = download_video(url)
        with VideoFileClip(str(src_path)) as full_clip:
            start = find_hook_start(full_clip, CLIP_DURATION_SEC)
            end = min(full_clip.duration, start + CLIP_DURATION_SEC)
            sub_clip = full_clip.subclip(start, end)
            vertical_clip = crop_to_vertical(sub_clip)

            clip_id = uuid.uuid4().hex[:10]
            raw_out = TMP_DIR / f"raw_{clip_id}.mp4"
            audio_out = TMP_DIR / f"audio_{clip_id}.wav"
            final_out = OUTPUT_DIR / f"short_{clip_id}.mp4"
            thumb_out = OUTPUT_DIR / f"short_{clip_id}.jpg"

            vertical_clip.write_videofile(
                str(raw_out), codec="libx264", audio_codec="aac",
                fps=30, preset="veryfast", threads=2, logger=None,
            )
            vertical_clip.audio.write_audiofile(str(audio_out), logger=None)

        words = transcribe(audio_out)
        caption_lines = group_words_into_caption_lines(words)
        burn_captions(raw_out, final_out, caption_lines)
        save_thumbnail(final_out, thumb_out)

        for f in (src_path, raw_out, audio_out):
            if f and Path(f).exists():
                os.remove(f)

        return {
            "id": clip_id,
            "file": f"clips/{final_out.name}",
            "thumbnail": f"clips/{thumb_out.name}",
            "source_url": url,
            "duration_sec": round(end - start, 1),
            "caption_preview": (caption_lines[0]["text"] if caption_lines else ""),
        }

    except Exception as e:
        print(f"    [ERROR] failed to process {url}: {e}")
        return None
    finally:
        if src_path and Path(src_path).exists():
            os.remove(src_path)


def update_manifest(new_entries: list[dict]) -> None:
    existing = []
    if MANIFEST_FILE.exists():
        existing = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    existing = new_entries + existing  # newest first
    MANIFEST_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[✓] manifest.json updated -> {len(existing)} total clips")


def main():
    urls = load_urls()
    processed = load_processed()
    todo = [u for u in urls if u not in processed]

    if not todo:
        print("[i] No new URLs to process.")
        return

    results = []
    for url in todo:
        result = process_single_video(url)
        if result:
            results.append(result)
        processed.add(url)  # mark as attempted regardless of success, avoid infinite retry loop
        save_processed(processed)

    if results:
        update_manifest(results)
    else:
        print("[!] No clips were successfully generated.")


if __name__ == "__main__":
    main()
