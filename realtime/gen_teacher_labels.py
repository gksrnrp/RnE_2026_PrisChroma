"""교사(쌍 단위 근사 탐색기) 라벨 생성: 증류 학습용 데이터.

연구 코드 ``find_minimum_correction``이 푼 (색 A, 색 B, 사용자, 목표 C80) → 보정색
쌍을 JSONL로 저장한다. 한 쌍에 수 초가 걸리므로 여러 프로세스로 돌리고,
중간에 끊겨도 같은 명령을 다시 실행하면 이어서 만든다.

예 (Colab 백그라운드 또는 노트북):
    python gen_teacher_labels.py --n 30000 --out ../data/teacher --workers 8 --seed 20261105

여러 컴퓨터로 나눌 때는 --shard i --num-shards k 를 다르게 준다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import common as c

CVD_TYPES = ("protan", "deutan")


def sample_tasks(n: int, seed: int, c80_min: float, c80_max: float, confused_fraction: float) -> list[dict]:
    """재현 가능한 과제 목록. 혼동 쌍과 이미 구별되는 쌍을 정해진 비율로 따로 채운다."""
    if n <= 0 or not np.isfinite([c80_min, c80_max, confused_fraction]).all() or not 0 < c80_min < c80_max or not 0 <= confused_fraction <= 1:
        raise ValueError('require n>0, 0<c80-min<c80-max and confused-fraction in [0,1]')
    rng = np.random.default_rng(seed)
    quota = {True: round(n * confused_fraction), False: n - round(n * confused_fraction)}
    pools: dict[bool, list[dict]] = {True: [], False: []}
    while any(len(pools[k]) < quota[k] for k in pools):
        m = 4096
        cvd = rng.integers(0, 2, m)
        sev = np.round(rng.uniform(0.3, 1.0, m), 4)
        target = np.round(np.exp(rng.uniform(np.log(c80_min), np.log(c80_max), m)), 4)
        a = rng.integers(15, 241, (m, 3)).astype(float)
        step = rng.normal(size=(m, 3))
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        b_ok = c.srgb_to_oklab(a) + step * rng.uniform(0.02, 0.20, (m, 1))
        b = np.rint(c.oklab_to_srgb(b_ok)).clip(0, 255)
        keep = c.in_gamut_oklab(b_ok) & np.any(a != b, axis=1)
        sim = np.empty(m)
        for t_idx, t in enumerate(CVD_TYPES):  # 사용자마다 severity가 달라 행렬을 행별로 만든다
            rows = np.flatnonzero(cvd == t_idx)
            mats = np.stack([np.asarray(c.get_machado_matrix(t, float(s))) for s in sev[rows]])
            sim[rows] = c.delta_e_2000(c.simulate_lab_batch(a[rows], mats), c.simulate_lab_batch(b[rows], mats))
        orig = c.original_delta_e(a, b)
        confused = (sim < target) & (orig >= np.maximum(3.0, target))
        for i in np.flatnonzero(keep):
            k = bool(confused[i])
            if len(pools[k]) >= quota[k]:
                continue
            pools[k].append(
                {
                    "cvd_type": CVD_TYPES[cvd[i]],
                    "severity": float(sev[i]),
                    "target": float(target[i]),
                    "a": [int(v) for v in a[i]],
                    "b": [int(v) for v in b[i]],
                    "sim_de": float(sim[i]),
                    "orig_de": float(orig[i]),
                    "confused": k,
                }
            )
    tasks = pools[True] + pools[False]
    order = rng.permutation(len(tasks))
    return [{"id": int(j), **tasks[i]} for j, i in enumerate(order)]


def solve(task: dict) -> dict:
    start = time.perf_counter()
    try:
        r = c.find_minimum_correction(task["a"], task["b"], task["cvd_type"], task["severity"], task["target"])
        out = {
            "success": bool(r["success"]),
            "strategy": r["strategy"],
            "a_new": list(r["corrected_A"]),
            "b_new": list(r["corrected_B"]),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - 실패도 기록해 비율을 보고한다
        out = {"success": False, "strategy": "error", "a_new": task["a"], "b_new": task["b"], "error": repr(exc)}
    out["seconds"] = time.perf_counter() - start
    return {**task, **out}


def check_resume(out_dir: Path, out_file: Path, tasks: list[dict], settings: dict) -> set[int]:
    """Refuse to combine label runs with different inputs or implementation."""
    files = ['common.py', 'gen_teacher_labels.py']
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files}
    for name in ('correction.py', 'color_spaces.py', 'distance.py', 'machado.py', 'config.py'):
        hashes['src/' + name] = hashlib.sha256((c.SRC / name).read_bytes()).hexdigest()
    metadata = {'settings': settings, 'implementation_sha256': hashes,
                'task_sha256': hashlib.sha256(json.dumps(tasks, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    path = out_dir / 'META.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('label settings or implementation differ; use a new output folder')
    else:
        if any(out_dir.glob('*.jsonl')):
            raise ValueError('existing labels have no matching run metadata')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
    by_id = {task['id']: task for task in tasks}
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ident = row['id']
            if ident in done or ident not in by_id or any(row.get(k) != v for k, v in by_id[ident].items()):
                raise ValueError('duplicate or mismatched label record; original file was not changed')
            done.add(ident)
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--out", default="../data/teacher")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--c80-min", type=float, default=1.0, help="목표 C80 하한(ΔE00). 교정 결과가 나오면 그 범위를 넓게 포함하도록 조정")
    p.add_argument("--c80-max", type=float, default=10.0)
    p.add_argument("--confused-fraction", type=float, default=0.9)
    args = p.parse_args()
    if args.workers <= 0 or args.num_shards <= 0 or not 0 <= args.shard < args.num_shards:
        p.error('workers and num-shards must be positive; shard must be in range')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"labels_seed{args.seed}_shard{args.shard}of{args.num_shards}.jsonl"
    tasks = sample_tasks(args.n, args.seed, args.c80_min, args.c80_max, args.confused_fraction)
    settings = {key: getattr(args, key) for key in ('n', 'seed', 'c80_min', 'c80_max', 'confused_fraction', 'num_shards')}
    done = check_resume(out_dir, out_file, tasks, settings)
    tasks = [t for t in tasks if t["id"] % args.num_shards == args.shard and t["id"] not in done]
    print(f"남은 과제 {len(tasks)}개, 이미 완료 {len(done)}개, workers={args.workers}", flush=True)

    t0 = time.perf_counter()
    with Pool(args.workers) as pool, out_file.open("a", encoding="utf-8") as f:
        for k, rec in enumerate(pool.imap_unordered(solve, tasks, chunksize=1), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 50 == 0:
                f.flush()
                rate = k / (time.perf_counter() - t0)
                print(f"{k}/{len(tasks)}  {rate:.2f}쌍/초  남은 시간 약 {(len(tasks) - k) / rate / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
