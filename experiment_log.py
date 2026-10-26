from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from contracts import (
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    MetricName,
    SandboxOutput,
    TaskType,
)


class ExperimentLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, result: ExperimentResult) -> None:
        record = {
            "hypothesis": {
                "id": result.hypothesis.id,
                "description": result.hypothesis.description,
                "config": dict(result.hypothesis.config),
                "code": result.hypothesis.code,
            },
            "task_type": result.task_type.value,
            "metric": result.metric.value,
            "status": result.status.value,
            "metric_value": result.metric_value,
            "elapsed_seconds": result.elapsed_seconds,
            "started_at": result.started_at,
            "error": result.error,
            "sandbox_output": (
                asdict(result.sandbox_output)
                if result.sandbox_output is not None
                else None
            ),
        }
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    def read(self) -> list[ExperimentResult]:
        if not self.path.exists():
            return []
        results: list[ExperimentResult] = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    results.append(_result_from_record(record))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid experiment log record at {self.path}:{line_number}"
                    ) from error
        return results


def _result_from_record(record: dict[str, Any]) -> ExperimentResult:
    hypothesis_data = record["hypothesis"]
    return ExperimentResult(
        hypothesis=Hypothesis(
            id=hypothesis_data["id"],
            description=hypothesis_data["description"],
            config=hypothesis_data.get("config", {}),
            code=hypothesis_data.get("code"),
        ),
        task_type=TaskType(record["task_type"]),
        metric=MetricName(record["metric"]),
        status=ExperimentStatus(record["status"]),
        metric_value=record.get("metric_value"),
        elapsed_seconds=record["elapsed_seconds"],
        started_at=record["started_at"],
        error=record.get("error"),
        sandbox_output=_sandbox_output_from_record(record.get("sandbox_output")),
    )


def _sandbox_output_from_record(
    record: dict[str, Any] | None,
) -> SandboxOutput | None:
    if record is None:
        return None
    return SandboxOutput(
        **{
            **record,
            "output_files": tuple(record.get("output_files", ())),
            "missing_outputs": tuple(record.get("missing_outputs", ())),
        }
    )
