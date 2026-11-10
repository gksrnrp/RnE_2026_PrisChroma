"""보정 전후 색쌍 보기 (응답 저장 없음, 시연용).

실험과 같은 회색 배경(RGB 117)과 1:1 픽셀로, 위에는 원래 색쌍, 아래에는 보정 색쌍을 같은 굵기로 보여 준다.
→/Space: 다음, ←: 이전, Esc: 종료.
"""
import csv
import ctypes
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageTk

CSV = Path(r'C:\Users\user\OneDrive\바탕 화면\RnE-2026-prischroma-main\팀전달_20261109\보정_계산_P01_P03.csv')
BG = (117, 117, 117)
LENGTH = 240
participant = sys.argv[1] if len(sys.argv) > 1 else 'P01'

rows = [r for r in csv.DictReader(open(CSV, encoding='utf-8-sig')) if r['participant'] == participant]
parse = lambda s: tuple(int(x) for x in s.strip('[]').split(','))
slides = [r for r in rows if r['width_px'] in ('4', '8', '16')]

ctypes.windll.shcore.SetProcessDpiAwareness(2)
root = tk.Tk()
root.configure(bg='#757575', cursor='none')
root.attributes('-fullscreen', True)
label = tk.Label(root, bd=0, highlightthickness=0, bg='#757575')
label.place(relx=0.5, rely=0.5, anchor='center')
font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 28)
state = {'i': 0}


def render(r):
    w = int(r['width_px'])
    im = Image.new('RGB', (1400, 900), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 30), f"{participant} ({r['cvd']})  굵기 {w}px  원래 색차 {float(r['source_sim_dE']):.2f} → 보정 {float(r['achieved_sim_dE']):.2f}"
           f"  (목표 C80 {float(r['c80_target']):.2f})   [{state['i']+1}/{len(slides)}]", fill='white', font=font)
    for y0, title, a, b in [(220, '원래', parse(r['source_rgb_a']), parse(r['source_rgb_b'])),
                            (560, '보정', parse(r['corrected_rgb_a']), parse(r['corrected_rgb_b']))]:
        d.text((40, y0 - 70), title, fill='white', font=font)
        d.rectangle((300, y0, 300 + LENGTH - 1, y0 + w - 1), fill=a)
        d.rectangle((700, y0, 700 + LENGTH - 1, y0 + w - 1), fill=b)
    d.text((40, 840), '→/Space 다음   ← 이전   Esc 종료', fill='white', font=font)
    return im


def show():
    photo = ImageTk.PhotoImage(render(slides[state['i']]))
    label.configure(image=photo)
    label.image = photo


def step(k):
    state['i'] = (state['i'] + k) % len(slides)
    show()


root.bind('<Right>', lambda e: step(1))
root.bind('<space>', lambda e: step(1))
root.bind('<Left>', lambda e: step(-1))
root.bind('<Escape>', lambda e: root.destroy())
show()
root.mainloop()
