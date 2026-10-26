from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.contracts import (
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    MetricName,
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
            },
            "task_type": result.task_type.value,
            "metric": result.metric.value,
            "status": result.status.value,
            "metric_value": result.metric_value,
            "elapsed_seconds": result.elapsed_seconds,
            "started_at": result.started_at,
            "error": result.error,
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
        ),
        task_type=TaskType(record["task_type"]),
        metric=MetricName(record["metric"]),
        status=ExperimentStatus(record["status"]),
        metric_value=record.get("metric_value"),
        elapsed_seconds=record["elapsed_seconds"],
        started_at=record["started_at"],
        error=record.get("error"),
    )
