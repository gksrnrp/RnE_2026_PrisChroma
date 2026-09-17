"""검증 색쌍과 개인 C80을 기존 최소 색이동 baseline에 일괄 연결한다.

현재 ``0.1-draft`` 계약은 synthetic 자료만 허용한다. 가짜 연결 시험에는
``development_mode=True``를 명시해야 하며, 실제 연구 자료로 자동 승격하지
않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from baseline import run_baseline
from config import SearchConfig
from data_contracts import validate_file, validate_payload
from threshold_adapter import correction_thresholds, load_threshold_estimate


def _result_row(pair: dict, outcome: dict) -> dict:
    correction = outcome["correction"]
    if correction is None:
        strategy = None
        change_a = 0.0
        change_b = 0.0
        total_change = 0.0
        in_gamut = True
    else:
        strategy = correction["strategy"]
        change_a = float(correction["color_change_A"])
        change_b = float(correction["color_change_B"])
        total_change = float(correction["total_color_change"])
        in_gamut = bool(correction["in_srgb_gamut"])

    target = float(outcome["before"]["confusion_threshold"])
    simulated_after = float(outcome["after"]["simulated_distance"])
    return {
        "pair_id": pair["pair_id"],
        "source_case": pair["source_case"],
        "selection_split": pair["selection_split"],
        "status": outcome["status"],
        "reason": outcome["reason"],
        "input_rgb_a": list(outcome["input"]["color_A"]),
        "input_rgb_b": list(outcome["input"]["color_B"]),
        "corrected_rgb_a": list(outcome["corrected_A"]),
        "corrected_rgb_b": list(outcome["corrected_B"]),
        "palette": pair["palette"],
        "original_delta_e00": float(outcome["before"]["original_distance"]),
        "simulated_delta_e00_before": float(
            outcome["before"]["simulated_distance"]
        ),
        "simulated_delta_e00_after": simulated_after,
        "target_met": simulated_after >= target,
        "strategy": strategy,
        "color_change_a": change_a,
        "color_change_b": change_b,
        "total_color_change": total_change,
        "in_srgb_gamut": in_gamut,
        "palette_constraint_used": bool(pair["palette"] and correction is not None),
    }


def run_personal_corrections(
    pair_path: str | Path,
    threshold_path: str | Path,
    *,
    participant_id: str,
    model: str,
    stroke_width_px: int,
    original_min: float,
    target_policy: str = "point",
    development_mode: bool = False,
    search_config: SearchConfig | None = None,
) -> dict:
    """정확히 일치하는 개인 C80 하나로 모든 검증 색쌍을 처리한다."""
    pairs_payload = validate_file("validation_pairs", pair_path)
    estimate = load_threshold_estimate(
        threshold_path,
        participant_id=participant_id,
        model=model,
        stroke_width_px=stroke_width_px,
        development_mode=development_mode,
    )
    thresholds = correction_thresholds(
        estimate,
        original_min=original_min,
        target_policy=target_policy,
        development_mode=development_mode,
    )
    if pairs_payload["data_origin"] != estimate.data_origin:
        raise ValueError("색쌍과 C80의 data_origin이 서로 다릅니다.")

    search = search_config or SearchConfig()

    rows = []
    for pair in pairs_payload["pairs"]:
        outcome = run_baseline(
            pair["rgb_a"],
            pair["rgb_b"],
            estimate.cvd_type,
            estimate.severity,
            thresholds,
            search_config=search,
            palette=pair["palette"],
        )
        rows.append(_result_row(pair, outcome))

    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("unchanged", "corrected", "search_failed")
    }
    payload = {
        "schema_version": "0.1-draft",
        "data_origin": estimate.data_origin,
        "generated_by": "run_personal_correction.py",
        "optimization_scope": "approximate_pairwise_not_global",
        "interpretation_scope": "model_constraint_only_not_human_readability",
        "inputs": {
            "validation_pairs_schema_version": pairs_payload["schema_version"],
            "participant_thresholds_schema_version": estimate.schema_version,
        },
        "search": {
            "step": search.step,
            "max_shift": search.max_shift,
            "direction_grid_radius": search.direction_grid_radius,
            "pair_chunk_size": search.pair_chunk_size,
        },
        "threshold": {
            "participant_id": estimate.participant_id,
            "cvd_type": estimate.cvd_type,
            "severity": estimate.severity,
            "severity_source": estimate.severity_source,
            "confirmed": estimate.confirmed,
            "model": estimate.model,
            "stroke_width_px": estimate.stroke_width_px,
            "stroke_width_deg": estimate.stroke_width_deg,
            "c80_sim": estimate.c80_sim,
            "ci_low": estimate.ci_low,
            "ci_high": estimate.ci_high,
            "value_type": estimate.value_type,
            "target_policy": target_policy,
            "selected_target": thresholds.target,
            "original_min": thresholds.original_min,
        },
        "summary": {
            "pair_count": len(rows),
            "unchanged_count": counts["unchanged"],
            "corrected_count": counts["corrected"],
            "search_failed_count": counts["search_failed"],
            "target_met_count": sum(row["target_met"] for row in rows),
        },
        "results": rows,
    }
    return validate_payload("correction_results", payload)


def write_correction_results(path: str | Path, payload: dict) -> Path:
    """검증을 통과한 결과만 임시 파일을 거쳐 원자적으로 저장한다."""
    validate_payload("correction_results", payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="개인 C80과 검증 색쌍을 최소 색이동 baseline에 연결합니다."
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument(
        "--model", required=True, choices=["B192", "B384", "C192", "D384"]
    )
    parser.add_argument("--stroke-width-px", required=True, type=int)
    parser.add_argument("--original-min", required=True, type=float)
    parser.add_argument("--target-policy", choices=["point", "ci_high"], default="point")
    parser.add_argument("--search-step", type=float, default=0.005)
    parser.add_argument("--max-shift", type=float, default=0.30)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--development-mode",
        action="store_true",
        help="synthetic/미확정 초안 자료 연결 시험에만 사용합니다.",
    )
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = _parser().parse_args()
    payload = run_personal_corrections(
        args.pairs,
        args.thresholds,
        participant_id=args.participant,
        model=args.model,
        stroke_width_px=args.stroke_width_px,
        original_min=args.original_min,
        target_policy=args.target_policy,
        development_mode=args.development_mode,
        search_config=SearchConfig(step=args.search_step, max_shift=args.max_shift),
    )
    output = write_correction_results(args.output, payload)
    summary = payload["summary"]
    print(
        f"저장 완료: {output} | {summary['pair_count']}쌍 중 "
        f"보정 {summary['corrected_count']}, 유지 {summary['unchanged_count']}, "
        f"실패 {summary['search_failed_count']}"
    )


if __name__ == "__main__":
    main()


__all__ = ["run_personal_corrections", "write_correction_results"]
