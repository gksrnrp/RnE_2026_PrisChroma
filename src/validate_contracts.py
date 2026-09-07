"""공통 데이터 계약을 명령줄에서 검사한다."""

from __future__ import annotations

import argparse
import json
import sys

from data_contracts import SCHEMA_PATHS, validate_all_examples, validate_file, validate_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R&E 공통 데이터 형식을 검사합니다.")
    parser.add_argument(
        "--all-examples",
        action="store_true",
        help="저장소의 초안 예제 5개를 모두 검사합니다.",
    )
    parser.add_argument("--contract", choices=sorted(SCHEMA_PATHS))
    parser.add_argument("--data")
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = _parser().parse_args()
    if args.all_examples:
        print(json.dumps(validate_all_examples(), ensure_ascii=False, indent=2))
        return
    if not args.contract or not args.data:
        _parser().error("--all-examples 또는 --contract와 --data를 지정해야 합니다.")
    if args.contract == "trial":
        records = validate_jsonl(args.data)
        print(f"정상: trial {len(records)}개")
    else:
        validate_file(args.contract, args.data)
        print(f"정상: {args.contract}")


if __name__ == "__main__":
    main()
