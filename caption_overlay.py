"""底部双语字幕浮窗:无边框、置顶、点击穿透、不抢焦点、半透明。"""
from AppKit import (
    NSPanel, NSColor, NSTextField, NSView, NSScreen, NSFont,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSScreenSaverWindowLevel, NSBackingStoreBuffered,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
)
from Foundation import NSMakeRect


class CaptionOverlay:
    def __init__(self):
        scr = NSScreen.mainScreen().frame()
        w, h = scr.size.width * 0.8, 96
        x = (scr.size.width - w) / 2
        y = 60  # 距屏幕底部
        rect = NSMakeRect(x, y, w, h)
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False)
        panel.setLevel_(NSScreenSaverWindowLevel)        # 盖住会议/全屏
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setIgnoresMouseEvents_(True)               # 点击穿透
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary)

        bg = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
        bg.setWantsLayer_(True)
        bg.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.55).CGColor())
        bg.layer().setCornerRadius_(14.0)
        panel.setContentView_(bg)

        self.orig = self._label(NSMakeRect(16, h / 2, w - 32, h / 2 - 8), 18,
                                NSColor.colorWithCalibratedWhite_alpha_(0.85, 1.0))
        self.zh = self._label(NSMakeRect(16, 6, w - 32, h / 2 - 2), 26, NSColor.whiteColor())
        bg.addSubview_(self.orig)
        bg.addSubview_(self.zh)
        self.panel = panel

    def _label(self, frame, size, color):
        f = NSTextField.alloc().initWithFrame_(frame)
        f.setBezeled_(False); f.setEditable_(False); f.setSelectable_(False)
        f.setDrawsBackground_(False); f.setTextColor_(color)
        f.setFont_(NSFont.systemFontOfSize_(size))
        f.setStringValue_("")
        return f

    def update(self, orig, zh):
        self.orig.setStringValue_(orig or "")
        self.zh.setStringValue_(zh or "")

    def show(self):
        self.panel.orderFrontRegardless()

    def hide(self):
        self.panel.orderOut_(None)
