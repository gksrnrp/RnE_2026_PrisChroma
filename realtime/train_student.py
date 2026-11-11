"""교사 라벨로 학생 MLP를 증류 학습한다 (Colab T4/L4 또는 CPU, 수 분~30분).

    python train_student.py --labels ../data/teacher --out ../results/realtime/student

출력: student.npz(실행용 numpy 가중치), student.pt, train_report.json(분할·손실·테스트 지표).
테스트 분할은 라벨 id로 결정적으로 나눈다(id % 10 == 0 → test, == 1 → val).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import common as c
from student import OUTPUT_SCALE, Student, make_features


def load_labels(folder: str) -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(glob.glob(str(Path(folder) / "*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r['id'] in seen:
                    raise ValueError(f"duplicate label id: {r['id']}")
                seen.add(r['id'])
                if r["success"] and r["strategy"] != "error":
                    rows.append(r)
    return rows


def to_arrays(rows: list[dict]):
    a = np.array([r["a"] for r in rows], float)
    b = np.array([r["b"] for r in rows], float)
    a2 = np.array([r["a_new"] for r in rows], float)
    b2 = np.array([r["b_new"] for r in rows], float)
    t = np.array([r["cvd_type"] for r in rows])
    s = np.array([r["severity"] for r in rows], float)
    g = np.array([r["target"] for r in rows], float)
    x = make_features(a, b, t, s, g)
    y = np.column_stack([c.srgb_to_oklab(a2) - c.srgb_to_oklab(a), c.srgb_to_oklab(b2) - c.srgb_to_oklab(b)])
    return x, y * OUTPUT_SCALE, (a, b, t, s, g, a2, b2)


def swap_augment(x: np.ndarray, y: np.ndarray):
    """교사는 A·B 순서에 대칭이므로 A와 B를 바꾼 예시를 추가한다."""
    xs = x.copy()
    xs[:, 0:3], xs[:, 3:6] = x[:, 3:6], x[:, 0:3]
    ys = y.copy()
    ys[:, 0:3], ys[:, 3:6] = y[:, 3:6], y[:, 0:3]
    return np.vstack([x, xs]), np.vstack([y, ys])


def build_mlp(d_in: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.01)]
        d = hidden
    layers.append(nn.Linear(d, 6))
    return nn.Sequential(*layers)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="../data/teacher")
    p.add_argument("--out", default="../results/realtime/student")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_labels(args.labels)
    if not rows:
        raise ValueError('no successful teacher labels')
    ids = np.array([r["id"] for r in rows])
    split = {"train": ids % 10 >= 2, "val": ids % 10 == 1, "test": ids % 10 == 0}
    if any(not mask.any() for mask in split.values()):
        raise ValueError('train, validation and test splits must all be nonempty')
    x, y, raw = to_arrays(rows)
    xtr, ytr = swap_augment(x[split["train"]], y[split["train"]])
    mean, std = xtr.mean(0), xtr.std(0) + 1e-6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def tens(v):
        return torch.tensor(v, dtype=torch.float32, device=device)

    xt, yt = tens((xtr - mean) / std), tens(ytr)
    xv, yv = tens((x[split["val"]] - mean) / std), tens(y[split["val"]])
    model = build_mlp(x.shape[1], args.hidden, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.05)
    best, best_state, history = float("inf"), None, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), args.batch):
            j = perm[i : i + args.batch]
            loss = loss_fn(model(xt[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xv), yv).item() if len(xv) else float("nan")
        history.append(vl)
        if vl < best:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}  val {vl:.5f}  best {best:.5f}  {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "args": vars(args)}, out / "student.pt")
    linear = [m for m in model if isinstance(m, nn.Linear)]
    npz = {"mean": mean, "std": std, "n_layers": len(linear)}
    for i, m in enumerate(linear):
        npz[f"W{i}"] = m.weight.detach().cpu().numpy().astype(np.float64)
        npz[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float64)
    np.savez(out / "student.npz", **npz)

    # 테스트 분할: 학생(+재검사) vs 교사
    st = Student(out / "student.npz")
    te = split["test"]
    a, b, t, s, g, a2, b2 = (v[te] for v in raw)
    start = time.perf_counter()
    na, nb, ok, steps = st.correct(a, b, t, s, g)
    per_pair_ms = (time.perf_counter() - start) * 1000 / max(1, te.sum())
    cost_student = np.linalg.norm(c.srgb_to_oklab(na) - c.srgb_to_oklab(a), axis=1) + np.linalg.norm(c.srgb_to_oklab(nb) - c.srgb_to_oklab(b), axis=1)
    cost_teacher = np.linalg.norm(c.srgb_to_oklab(a2) - c.srgb_to_oklab(a), axis=1) + np.linalg.norm(c.srgb_to_oklab(b2) - c.srgb_to_oklab(b), axis=1)
    moved = cost_teacher > 1e-9
    ratio = cost_student[moved & ok] / cost_teacher[moved & ok]
    teacher_sec = [r["seconds"] for r, m in zip(rows, te) if m and r["strategy"] != "unchanged"]
    report = {
        "n_labels": len(rows),
        "n_train": int(split["train"].sum()),
        "n_val": int(split["val"].sum()),
        "n_test": int(te.sum()),
        "best_val_loss": best,
        "test_pass_rate": float(ok.mean()),
        "test_scale_used_median": float(np.nanmedian(steps)) if ok.any() else None,
        "test_cost_ratio_median": float(np.median(ratio)) if len(ratio) else None,
        "test_cost_ratio_p90": float(np.percentile(ratio, 90)) if len(ratio) else None,
        "student_ms_per_pair_incl_check": per_pair_ms,
        "teacher_sec_per_pair_median": float(np.median(teacher_sec)) if teacher_sec else None,
        "device": device,
        "git_commit": c.git_commit(),
        "labels_sha256": hashlib.sha256("".join(sorted(json.dumps(r, sort_keys=True) for r in rows)).encode()).hexdigest(),
        "args": vars(args),
        "note": "계산상 지표. 사람의 색 구별 향상 근거가 아님.",
    }
    (out / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
