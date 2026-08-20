"""이창민 파트 모듈과 값이 같은지 대조한다.

왜 필요한가
-----------
이 패키지의 `machado.py` 와 `colorspace.py` 는 원래 1번(이창민) 담당인데,
추정 모듈이 시뮬레이터 없이는 한 줄도 못 돌아가서 최소 구현을 넣어 둔 것이다.
두 구현이 조용히 어긋나면 추정 결과와 보정 결과가 서로 다른 전제 위에 서게
된다. 그래서 손으로 확인하는 대신 이 스크립트로 대조한다.

이쪽은 numpy 배열을 한 번에 처리하고 저쪽은 색 하나씩 처리한다. 속도 때문에
합치지 않고 **값이 같은지만** 확인한다.

실행
    python -m colorvision.crosscheck <창민 저장소>/src
"""

import argparse
import sys

import numpy as np

from . import colorspace as hcs
from . import machado as hm

RESULTS = []


def check(name, worst, tol, unit=""):
    ok = worst <= tol
    RESULTS.append(ok)
    print("  [%s] %-34s 최대차이 %.2e%s" % ("OK  " if ok else "FAIL", name, worst, unit))
    return ok


def run(src_path, n=300, seed=0, tol=1e-9):
    sys.path.insert(0, src_path)
    try:
        import machado as cm
        import distance as cd
        import color_spaces as ccs
    except ImportError as exc:
        print("창민 모듈을 불러오지 못했다: %s" % exc)
        print("경로가 맞는지 확인할 것: %s" % src_path)
        return False

    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 256, size=(n, 3))
    as_tuple = [tuple(int(v) for v in c) for c in cols]

    print("Machado 행렬 표")
    worst = 0.0
    for t in ("protan", "deutan"):
        for s in np.arange(0, 1.0001, 0.01):
            a = np.array(cm.get_machado_matrix(t, float(s)))
            worst = max(worst, float(np.abs(a - hm.matrix(t, float(s))).max()))
    check("severity 0.00~1.00 전 구간", worst, 1e-12)

    print("\n시뮬레이션 결과 (0~255 기준)")
    worst = 0.0
    for t in ("protan", "deutan"):
        for s in (0.15, 0.37, 0.62, 0.88, 1.0):
            mine = hm.simulate_srgb(cols / 255.0, t, s) * 255.0
            his = np.array([cm.simulate_color_float(c, t, s) for c in as_tuple])
            worst = max(worst, float(np.abs(mine - his).max()))
    check("여러 유형 x severity", worst, 1e-6, " 계조")

    print("\nΔE00 (CIEDE2000)")
    lab1 = rng.uniform([0, -80, -80], [100, 80, 80], size=(n, 3))
    lab2 = rng.uniform([0, -80, -80], [100, 80, 80], size=(n, 3))
    mine = hcs.delta_e00(lab1, lab2)
    his = np.array([cd.delta_e_2000(a, b) for a, b in zip(lab1, lab2)])
    check("같은 Lab 입력 시 공식 일치", float(np.abs(mine - his).max()), 1e-9)

    # Lab 변환은 백색점 처리가 미세하게 다르다. 무시할 수준인지 확인만 한다.
    mine_lab = hcs.srgb_to_lab(cols / 255.0)
    his_lab = np.array([ccs.srgb_to_lab(c) for c in as_tuple])
    d = float(np.abs(mine_lab - his_lab).max())
    check("sRGB->Lab (백색점 처리 차이)", d, 1e-3)
    if d > 1e-9:
        print("       참고: 이쪽은 변환 행렬에서 유도한 백색점을, 저쪽은 D65 상수를")
        print("       쓴다. L* 기준 %.1e 차이라 결과에 영향 없다." % d)

    print("\nOKLab")
    mine = hcs.srgb_to_oklab(cols / 255.0)
    his = np.array([ccs.srgb_to_oklab(c) for c in as_tuple])
    check("sRGB->OKLab", float(np.abs(mine - his).max()), 1e-9)

    n_fail = RESULTS.count(False)
    print("\n%d개 중 %d개 일치%s" % (len(RESULTS), len(RESULTS) - n_fail,
                                  "" if n_fail == 0 else "  <-- %d개 불일치" % n_fail))
    if n_fail == 0:
        print("두 구현이 같은 값을 낸다. 그대로 써도 된다.")
    else:
        print("불일치가 있다. 어느 쪽이 맞는지 정하고 맞춘 뒤에 진행할 것.")
    return n_fail == 0


def main(argv=None):
    p = argparse.ArgumentParser(description="이창민 파트 모듈과 값 대조")
    p.add_argument("src", help="창민 저장소의 src 폴더 경로")
    p.add_argument("-n", type=int, default=300, help="표본 색 개수")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    return 0 if run(args.src, n=args.n, seed=args.seed) else 1


if __name__ == "__main__":
    sys.exit(main())
