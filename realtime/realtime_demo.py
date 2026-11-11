"""실시간 데모: 화면 일부(또는 동영상·이미지)를 캡처해 보정한 결과를 창에 띄운다.

팀원 노트북에서 실행한다(Colab은 화면·창이 없어 불가).
    pip install mss opencv-python numpy scipy pillow
    python realtime_demo.py --profile profiles/user.json --display profiles/display.json \
        --mode student --student models/student_v1/student.npz --region 0 0 1280 720

키: q 종료, m 모드 순환(none → base_full → base_gated → student), s 현재 프레임 저장.
창 제목과 콘솔에 프레임당 시간(ms)과 fps가 표시된다. 측정값은 ../results/realtime/demo_timings.csv에 쌓인다.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import common as c
from base_models import load_lut
from pipeline import RealtimeCorrector
from student import Student

MODES = ("none", "base_full", "base_gated", "student")


def frames(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif args.image:
        rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        while True:
            yield rgb
    else:
        import mss

        x, y, w, h = args.region
        with mss.mss() as sct:
            while True:
                shot = np.asarray(sct.grab({"left": x, "top": y, "width": w, "height": h}))
                yield np.ascontiguousarray(shot[:, :, 2::-1])  # BGRA → RGB


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True)
    p.add_argument("--display", required=True)
    p.add_argument("--mode", default="student", choices=MODES)
    p.add_argument("--student", default="models/student_v1/student.npz")
    p.add_argument("--base-lut", default=None, help="Ebelin 참조 변환으로 만든 .npy (없으면 고전 달토니제이션)")
    p.add_argument("--region", type=int, nargs=4, default=[0, 0, 1280, 720])
    p.add_argument("--video")
    p.add_argument("--image")
    args = p.parse_args()

    profile = c.UserProfile.load(args.profile)
    display = c.DisplayGeometry(**json.loads(Path(args.display).read_text()))
    student = Student(args.student) if Path(args.student).exists() else None
    base = load_lut(args.base_lut) if args.base_lut else None
    mode = args.mode

    def make(m):
        return RealtimeCorrector(profile, display, mode=m, student=student, base_delta=base)

    rc = make(mode)
    log_path = Path("../results/realtime/demo_timings.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        log = csv.writer(fh)
        if new:
            log.writerow(["time", "mode", "width", "height", "total_ms", "apply_ms", "analyzed", "palette_ms", "width_ms", "judge_ms", "correct_ms", "lut_ms"])
        last = time.perf_counter()
        for frame in frames(args):
            out, t = rc.process(frame)
            now = time.perf_counter()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            log.writerow([f"{time.time():.3f}", mode, frame.shape[1], frame.shape[0]] + [round(t.get(k, 0), 3) for k in ("total_ms", "apply_ms", "analyzed", "palette_ms", "width_ms", "judge_ms", "correct_ms", "lut_ms")])
            view = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.putText(view, f"{mode}  {t['total_ms']:.1f} ms  {fps:.0f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(view, f"{mode}  {t['total_ms']:.1f} ms  {fps:.0f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.imshow("PrisChroma realtime", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
                if mode == "student" and student is None:
                    mode = "none"
                rc = make(mode)
            if key == ord("s"):
                cv2.imwrite(f"../results/realtime/snap_{mode}_{int(time.time())}.png", view)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
