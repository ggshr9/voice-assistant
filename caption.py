# /// script
# requires-python = ">=3.10"
# dependencies = ["sounddevice", "webrtcvad", "numpy", "setuptools<81", "pyobjc-framework-Cocoa"]
# ///
"""实时字幕编排器:采 BlackHole → VAD 切句 → whisper 转写 → Qwen3.6 译中文 → 底部浮窗。
只读 BlackHole(远端声音),不碰麦。启动切系统输出到「会议外放」,停止/异常切回。"""
import os, queue, signal, tempfile, threading, subprocess

import numpy as np
import sounddevice as sd
import webrtcvad
from AppKit import NSApplication, NSApp
from Foundation import NSTimer

import caption_core as cc
import caption_overlay

IN_DEV = "BlackHole 2ch"
MONITOR_OUTPUT = "会议外放"
FALLBACK_OUTPUT = "MacBook Pro Speakers"
SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE * 30 // 1000   # 480 samples / 30ms

ENV = {**os.environ,
       "PATH": "/opt/homebrew/bin:" + os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")}
_tmp = os.path.join(tempfile.gettempdir(), "caption_seg.wav")


def process_segment(pcm):
    """一段 PCM → (原文, 中文) 或 None(噪音/空)。"""
    cc.pcm_to_wav(pcm, _tmp)
    text, lang = cc.stt(_tmp)
    if cc.is_noise(text):
        return None
    zh = cc.translate(text, lang)
    if not zh:
        return None
    return (text, zh)


def get_output():
    try:
        r = subprocess.run(["SwitchAudioSource", "-c", "-t", "output"],
                           capture_output=True, text=True, env=ENV, timeout=8)
        return r.stdout.strip() or None
    except Exception:
        return None


def set_output(name):
    subprocess.run(["SwitchAudioSource", "-s", name, "-t", "output"],
                   capture_output=True, text=True, env=ENV, timeout=8)


def _find_input_index(name):
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name in d["name"]:
            return i, d["max_input_channels"]
    raise RuntimeError(f"找不到输入设备「{name}」")


class Caption:
    def __init__(self):
        self.overlay = caption_overlay.CaptionOverlay()
        self.q = queue.Queue()          # 主线程要显示的 (orig, zh)
        self.frames = queue.Queue()     # 采集帧
        self.pending = queue.Queue()    # 切好的句子 → worker
        self.running = True
        self.prev_output = None

    def start(self):
        self.prev_output = get_output() or FALLBACK_OUTPUT
        set_output(MONITOR_OUTPUT)
        idx, ch = _find_input_index(IN_DEV)
        self.overlay.show()
        threading.Thread(target=self._capture, args=(idx, ch), daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.2, True, lambda t: self._drain())

    def _capture(self, idx, ch):
        def cb(indata, frames, t, status):
            a = np.frombuffer(bytes(indata), dtype="<i2")
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1).astype("<i2")
            self.frames.put(a.tobytes())
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME,
                               dtype="int16", channels=ch, device=idx, callback=cb):
            vad = webrtcvad.Vad(2)

            def frame_iter():
                while self.running:
                    yield self.frames.get()
            for pcm in cc.segment_frames(frame_iter(), vad):
                if not self.running:
                    break
                self.pending.put(pcm)

    def _worker(self):
        while self.running:
            try:
                pcm = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                out = process_segment(pcm)
            except Exception as e:
                print("process error:", e); out = None
            if out:
                print(f"[字幕] {out[0]}  →  {out[1]}", flush=True)
                self.q.put(out)

    def _drain(self):
        latest = None
        while not self.q.empty():
            latest = self.q.get()
        if latest:
            self.overlay.update(latest[0], latest[1])

    def stop(self):
        self.running = False
        try:
            self.overlay.hide()
        finally:
            if self.prev_output:
                set_output(self.prev_output)


def main():
    app = NSApplication.sharedApplication()
    cap = Caption()
    try:
        cap.start()
    except Exception as e:
        print("启动失败:", e)
        if getattr(cap, "prev_output", None):
            set_output(cap.prev_output)
        return

    def _quit(*a):
        cap.stop(); NSApp().terminate_(None)
    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)
    try:
        app.run()
    finally:
        cap.stop()


if __name__ == "__main__":
    main()
