"""혼동 쌍 단위로 '가는 쪽 선 굵기'별 해결률과 색 검출률을 센다 (evaluate.py와 같은 장면·판정).

python width_bins.py <realtime 폴더> <student.npz> <profile.json> <display.json> <scenes> <out.csv>
"""
import csv
import sys

import numpy as np

rt, student_path, prof_path, disp_path, n_scenes, out_csv = sys.argv[1:7]
sys.path.insert(0, rt)
import common as c  # noqa: E402
from evaluate import corrected_color  # noqa: E402
from pipeline import RealtimeCorrector  # noqa: E402
from scene_gen import gt_pairs, make_scene  # noqa: E402
from student import Student  # noqa: E402

profile = c.UserProfile.load(prof_path)
display = c.DisplayGeometry(**__import__("json").load(open(disp_path, encoding="utf-8")))
student = Student(student_path)
base = (__import__("base_models").load_lut(sys.argv[7]) if len(sys.argv) > 7 else None)  # 없으면 고전 달토니제이션
methods = ["base_full", "base_gated", "student", "student_fixed", "student_aa"]
rows = []
for s in range(int(n_scenes)):
    frame, meta = make_scene(1000 * 0 + s + 50000, profile, display)
    lines = meta["lines"]
    before = gt_pairs(meta, profile, display)
    conf = [q for q in before if q["sim"] < q["need"] and q["orig"] >= profile.original_min]
    if not conf:
        continue
    for m in methods:
        use = profile.with_fixed_threshold() if m == "student_fixed" else profile
        from pipeline import Settings
        rc = RealtimeCorrector(use, display, mode={"student_fixed": "student", "student_aa": "student"}.get(m, m), student=student,
                               base_delta=base, settings=Settings(aa_unmix=m.endswith("_aa")))
        rc.process(frame, instant=True)
        new_cols = [corrected_color(rc, l["color"]) for l in lines]
        after = {(q["i"], q["j"]): q for q in gt_pairs(meta, profile, display, colors=new_cols)}
        pal = c.srgb_to_oklab(rc.analysis.colors) if rc.analysis is not None else None
        for q in conf:
            i, j = q["i"], q["j"]
            a = after[(i, j)]
            det = np.nan
            if pal is not None:
                det = float(all(np.min(np.linalg.norm(pal - c.srgb_to_oklab(np.array(lines[k]["color"], float)), axis=1)) < 0.03 for k in (i, j)))
            wmin = min(lines[i]["width_px"], lines[j]["width_px"])
            rows.append({"seed": meta["seed"], "method": m, "wmin_px": wmin,
                         "wmin_deg": float(display.px_to_deg(wmin)),
                         "in_range": int(not profile.out_of_range(float(display.px_to_deg(wmin)))),
                         "resolved": int(a["sim"] >= a["need"]), "detected": det,
                         "west": float(rc.analysis.widths_px[int(np.argmin(np.linalg.norm(pal - c.srgb_to_oklab(np.array(lines[i]["color"], float)), axis=1)))]) if pal is not None else np.nan})
with open(out_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=['seed', 'method', 'wmin_px', 'wmin_deg', 'in_range', 'resolved', 'detected', 'west'])
    w.writeheader()
    w.writerows(rows)
print("pairs", len(rows) // len(methods))
