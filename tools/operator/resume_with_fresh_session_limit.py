"""P03 교정 블록 나머지를 이어 하기 위한 실행 보조 (운영자 승인 편차, 2026-11-09).

원래 규칙: 블록 최초 기록 시각부터 30분이 지나면 그 블록을 더 수집하지 않는다.
운영자 결정: 휴식 화면에서 창이 닫혀 절반만 저장된 블록을 이어서 채운다.
이 보조는 30분 제한을 '이번 실행 시작'부터 세도록만 바꾼다. 저장소 파일은 바꾸지 않으며,
다른 모든 검사(설정·그림·동의·화면·Git)는 그대로 실행된다. 실행마다 비공개 로그를 남긴다.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(r'C:\Users\user\RnE-2026-prischroma\src')
sys.path.insert(0, str(SRC))
import response_gui  # noqa: E402

LOG = SRC.parent/'data'/'experiment'/'private'/'P03-session-limit-deviation-log.jsonl'
RECORD = SRC.parent/'data'/'experiment'/'private'/'P03-session-limit-deviation.json'


def _limit_from_this_run(self) -> bool:
    if self.session_start_ns is None:
        return False
    import time
    minutes = (time.perf_counter_ns() - self.session_start_ns) / 60_000_000_000.0
    return minutes >= float(self.plan.config['timing']['max_session_duration_min'])


if __name__ == '__main__':
    if not RECORD.is_file():
        raise SystemExit(f'운영자 승인 기록이 없습니다: {RECORD}')
    argv = sys.argv[1:]
    if '--run-gui' in argv:
        with LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'argv': argv,
                                'deviation': 'session limit counted from this run start'},
                               ensure_ascii=False) + '\n')
    response_gui.ExperimentWindow._session_limit_reached = _limit_from_this_run
    raise SystemExit(response_gui.main(argv))
