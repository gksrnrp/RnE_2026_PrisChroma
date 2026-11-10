"""v5 장면을 Tk 창에 1:1로 띄워 화면 캡처와 픽셀 비교한다. 응답은 저장하지 않는다."""
import ctypes
import json
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, r'C:\Users\user\RnE-pilot\tools\common_color_pilot')
from PIL import ImageChops, ImageGrab, ImageTk  # noqa: E402
import pilot  # noqa: E402
import lum_pilot as wide_pilot  # noqa: E402

LOG = Path(__file__).with_name('v6_pixel_check.json')
ctypes.windll.shcore.SetProcessDpiAwareness(2)
trials = wide_pilot.schedule()
picks = {}
for t in trials:
    picks.setdefault(t['width_px'], t)
root = tk.Tk()
root.configure(bg='#757575')
root.attributes('-fullscreen', True)
root.attributes('-topmost', True)
label = tk.Label(root, bd=0, highlightthickness=0, bg='#757575')
label.place(relx=0.5, rely=0.5, anchor='center')
results = []


def run(items):
    if not items:
        root.destroy()
        return
    width, trial = items[0]
    src = pilot.scene(trial)
    photo = ImageTk.PhotoImage(src)
    label.configure(image=photo)
    label.image = photo
    root.update()

    def grab():
        x, y = label.winfo_rootx(), label.winfo_rooty()
        shot = ImageGrab.grab(bbox=(x, y, x + src.width, y + src.height)).convert('RGB')
        diff = ImageChops.difference(shot, src).convert('L').point(lambda v: 255 if v else 0)
        results.append({'width_px': width, 'size': list(shot.size),
                        'mismatched_pixels': diff.histogram()[255], 'total': src.width * src.height})
        run(items[1:])
    root.after(700, grab)


root.after(600, lambda: run(sorted(picks.items())))
root.mainloop()
LOG.write_text(json.dumps(results, indent=1), encoding='utf-8')
