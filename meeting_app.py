# /// script
# requires-python = ">=3.10"
# dependencies = ["rumps"]
# ///
"""
会议纪要菜单栏小应用 —— 打开后顶部菜单栏出现 🎙️ 图标。
点它：录线上会议 / 录麦克风 → 再点「停止并出纪要」→ 自动转写+分人+纪要并打开。
只是个壳：真正干活仍调用现有 CLI `meeting`(转写+分人+minutes 纪要)，CLI 照常可用、逻辑不分叉。
"""
import os
import re
import signal
import subprocess
import threading
import time
import glob
import datetime

import rumps


def _stage_label(line):
    """从 meeting/minutes 的输出行解析出大致进度标签(显示在菜单栏)。"""
    m = re.search(r"\((\d+(?:\.\d+)?)%\)", line)        # Qwen3-ASR: "...(8.1%) ETA..."
    if m:
        return f"📝 {int(float(m.group(1)))}%"          # 转写进度
    if "已按说话人" in line:
        return "🗣 分人完成"
    if "转写完成" in line:
        return "🗣 整理中"
    mm = re.search(r"块\s*(\d+)/(\d+)", line)            # minutes 长会议 map-reduce
    if mm:
        return f"✍️ {mm.group(1)}/{mm.group(2)}"
    if "会议纪要" in line or "生成纪要" in line:
        return "✍️ 出纪要"
    return None

HOME = os.path.expanduser("~")
REC_DIR = os.path.join(HOME, "会议录音")
ONLINE_DEV = "会议录制"                       # 音频MIDI设置里的聚合设备(线上：对方+自己)
MIC_DEV = "MacBook Pro Microphone"           # 线下麦克风
MONITOR_OUTPUT = "会议外放"                   # 多输出(扬声器+BlackHole)：录线上时切到它，对方声音才录得到、自己也听得见
FALLBACK_OUTPUT = "MacBook Pro Speakers"     # 取不到原输出时切回这个
# .app 从 Finder 启动读不到 .zshrc，补好 PATH；HF_TOKEN 由启动脚本注入
ENV = {**os.environ,
       "PATH": "/opt/homebrew/bin:" + os.path.join(HOME, ".local/bin") + ":" + os.environ.get("PATH", "")}


def device_exists(dev):
    try:
        r = subprocess.run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                           capture_output=True, text=True, env=ENV, timeout=15)
        return f"] {dev}" in r.stderr
    except Exception:
        return False


def notify(title, subtitle, msg):
    try:
        rumps.notification(title, subtitle, msg)
    except Exception:
        pass


def get_output():
    try:
        r = subprocess.run(["SwitchAudioSource", "-c", "-t", "output"],
                           capture_output=True, text=True, env=ENV, timeout=8)
        return r.stdout.strip() or None
    except Exception:
        return None


def set_output(name):
    try:
        r = subprocess.run(["SwitchAudioSource", "-s", name, "-t", "output"],
                           capture_output=True, text=True, env=ENV, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


class MeetingApp(rumps.App):
    def __init__(self):
        super().__init__("🎙️", quit_button=None)
        self.proc = None
        self.outfile = None
        self.t0 = None
        self.prev_output = None      # 录线上前的系统输出，停止后切回
        self.timer = rumps.Timer(self._tick, 1)
        self._proc_result = None     # 后台转写结果：None=进行中，(md,)=完成
        self._progress = None        # 后台进度标签(后台线程写，主线程读到菜单栏)
        self._poll = None            # 主线程轮询定时器
        self._idle()

    # ---------- 菜单三态 ----------
    def _idle(self):
        self.title = "🎙️"
        self.menu.clear()
        cap_on = self._caption_running()
        self.menu = [
            rumps.MenuItem("🔴 录线上会议", callback=self.rec_online),
            rumps.MenuItem("🎤 录麦克风（线下）", callback=self.rec_mic),
            None,
            rumps.MenuItem("⏹ 停止实时字幕" if cap_on else "🌐 实时字幕（外语→中文）",
                           callback=self.toggle_caption),
            None,
            rumps.MenuItem("📂 打开录音文件夹", callback=self.open_folder),
            rumps.MenuItem("退出", callback=self.quit_app),
        ]

    CAPTION_SVC = "com.local.caption"

    def _caption_running(self):
        r = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{self.CAPTION_SVC}"],
                           capture_output=True, text=True, env=ENV)
        return "state = running" in r.stdout

    def toggle_caption(self, _):
        uid = os.getuid()
        svc = f"gui/{uid}/{self.CAPTION_SVC}"
        plist = os.path.expanduser("~/Library/LaunchAgents/com.local.caption.plist")
        if self._caption_running():
            subprocess.run(["launchctl", "kill", "TERM", svc], capture_output=True, env=ENV)
        else:
            ok = subprocess.run(
                ["bash", "-lc", "curl -s --max-time 2 http://127.0.0.1:8080/v1/models | grep -q qwen3.6"],
                env=ENV).returncode == 0
            if not ok:
                rumps.alert("大脑未就绪", "先在终端 `llm start`，再开实时字幕。"); return
            subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", plist], capture_output=True, env=ENV)
            subprocess.run(["launchctl", "kickstart", svc], capture_output=True, env=ENV)
        self._idle()

    def _recording(self):
        self.menu.clear()
        self.menu = [
            rumps.MenuItem("⏹ 停止并出纪要", callback=self.stop),
            rumps.MenuItem("✖︎ 丢弃本次录音", callback=self.discard),
            None,
            rumps.MenuItem("退出", callback=self.quit_app),
        ]

    def _processing(self, label):
        self.title = "⏳"
        self.menu.clear()
        self.menu = [rumps.MenuItem(label), rumps.MenuItem("退出", callback=self.quit_app)]

    # ---------- 录音 ----------
    def _start(self, dev, prefix, online=False):
        if self.proc:
            return
        if not device_exists(dev):
            rumps.alert("找不到音频设备",
                        f"「{dev}」不存在。\n线上录制需先在「音频 MIDI 设置」里建好名为「会议录制」的聚合设备。")
            return
        # 录线上：自动把系统输出切到「会议外放」(对方声音才进 BlackHole 被录到，自己也照常听见)
        if online:
            self.prev_output = get_output() or FALLBACK_OUTPUT   # 取不到也要兜底，否则停止时切不回
            if not set_output(MONITOR_OUTPUT):
                if not rumps.alert("切换输出失败",
                                   f"没能把系统输出切到「{MONITOR_OUTPUT}」，对方声音可能录不到。\n继续录吗？",
                                   ok="继续", cancel="取消"):
                    self.prev_output = None
                    return
        os.makedirs(REC_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.outfile = os.path.join(REC_DIR, f"{prefix}_{ts}.m4a")
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{dev}",
             "-ac", "1", "-ar", "16000", "-c:a", "aac", self.outfile],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ENV)
        self.t0 = time.time()
        self._recording()
        self.timer.start()

    def _restore_output(self):
        # 把系统输出切回录线上之前的设备(没记到就用扬声器兜底)
        if self.prev_output is not None:
            set_output(self.prev_output or FALLBACK_OUTPUT)
            self.prev_output = None

    def rec_online(self, _):
        self._start(ONLINE_DEV, "线上会议", online=True)

    def rec_mic(self, _):
        self._start(MIC_DEV, "会议")

    def _tick(self, _):
        if self.t0:
            s = int(time.time() - self.t0)
            self.title = f"● {s // 60:02d}:{s % 60:02d}"

    def _stop_ffmpeg(self):
        if not self.proc:
            return
        try:
            self.proc.stdin.write(b"q")     # ffmpeg 收到 q 优雅收尾并写完文件头
            self.proc.stdin.flush()
        except Exception:
            try:
                self.proc.send_signal(signal.SIGINT)
            except Exception:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    def stop(self, _):
        self.timer.stop()
        self._stop_ffmpeg()
        self._restore_output()        # 切回录线上前的输出
        f = self.outfile
        self.t0 = None
        self._processing("⏳ 转写 + 出纪要中…（首次需等大脑加载）")
        # 轮询定时器必须在主线程(本回调就是)创建，后台线程建的 NSTimer 不会触发
        self._proc_result = None
        self._progress = "⏳ 准备…"
        self._poll = rumps.Timer(self._poll_done, 0.5)
        self._poll.start()
        threading.Thread(target=self._process, args=(f,), daemon=True).start()

    def discard(self, _):
        self.timer.stop()
        self._stop_ffmpeg()
        self._restore_output()
        try:
            if self.outfile and os.path.exists(self.outfile):
                os.remove(self.outfile)
        except Exception:
            pass
        self.t0 = None
        self.outfile = None
        self._idle()

    # ---------- 转写+纪要(后台线程，调现有 meeting CLI) ----------
    def _process(self, f):
        md = None
        try:
            # 用 Popen 逐行读 meeting 输出，解析出大致进度推给菜单栏
            proc = subprocess.Popen(
                ["meeting", f], env=ENV,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                lbl = _stage_label(line)
                if lbl:
                    self._progress = lbl
            proc.wait()
            base = os.path.splitext(os.path.basename(f))[0]
            d = os.path.join(os.path.dirname(f), f"转写_{base}")
            cands = sorted(glob.glob(os.path.join(d, "纪要_*.md")))
            if cands:
                md = cands[0]
        except Exception as e:
            print("process error:", e)
        self._proc_result = (md,)        # 元组=完成(md 可能 None)；主线程轮询读它

    def _poll_done(self, _):
        # 主线程定时器：刷新进度标签 + 完成后更新 UI、打开纪要
        if self._progress:
            self.title = self._progress
        if self._proc_result is None:
            return
        if self._poll:
            self._poll.stop()
        md = self._proc_result[0]
        self._proc_result = None
        self._progress = None
        self._idle()
        if md:
            notify("会议纪要已生成", os.path.basename(md), "已自动打开")
            subprocess.run(["open", md], env=ENV)
        else:
            notify("转写完成，但没找到纪要", "", "看录音文件夹手动检查")
            subprocess.run(["open", REC_DIR], env=ENV)

    def open_folder(self, _):
        os.makedirs(REC_DIR, exist_ok=True)
        subprocess.run(["open", REC_DIR], env=ENV)

    def quit_app(self, _):
        self._stop_ffmpeg()
        self._restore_output()
        rumps.quit_application()


if __name__ == "__main__":
    try:
        # 只在菜单栏出现，不在 Dock 占图标
        from AppKit import NSApplication
        NSApplication.sharedApplication().setActivationPolicy_(1)  # Accessory
    except Exception:
        pass
    MeetingApp().run()
