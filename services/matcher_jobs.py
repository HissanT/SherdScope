"""In-process background jobs for contour-library builds and shape matching."""

from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from catalog.contours import build_reference_library
from catalog.matcher import run_match


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MatcherJobCoordinator:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _set(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(updates)
                job["updated_at"] = _now()

    def _start(
        self,
        kind: str,
        project_id: str,
        target: Callable[[str], dict[str, Any]],
        *,
        requested_id: str | None = None,
    ) -> dict[str, Any]:
        job_id = requested_id or uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "kind": kind,
            "project_id": project_id,
            "state": "queued",
            "message": "Queued",
            "current": 0,
            "total": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        def worker() -> None:
            self._set(job_id, state="running", message="Starting")
            try:
                result = target(job_id)
                self._set(job_id, state="complete", message="Complete", result=result)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                self._set(
                    job_id,
                    state="failed",
                    message=error_text,
                    error=error_text,
                    traceback=traceback.format_exc(),
                )

        threading.Thread(target=worker, daemon=True, name=f"sherdscope-{kind}-{job_id[:8]}").start()
        return dict(job)

    def start_library_build(self, project_id: str, project_path: Path, source_pdf: str | None = None) -> dict[str, Any]:
        def target(job_id: str):
            def progress(current: int, total: int, message: str):
                self._set(job_id, current=current, total=total, message=message)

            return build_reference_library(project_path, progress=progress, source_pdf=source_pdf)

        return self._start("contour_library", project_id, target)

    def start_match(
        self,
        project_id: str,
        project_path: Path,
        query_id: str,
    ) -> tuple[dict[str, Any], str]:
        run_id = uuid.uuid4().hex

        def target(job_id: str):
            def progress(level: int, current: int, total: int, message: str):
                self._set(
                    job_id,
                    level=level,
                    current=current,
                    total=total,
                    message=message,
                    run_id=run_id,
                )

            return run_match(
                project_path,
                query_id,
                run_id=run_id,
                progress=progress,
            )

        return self._start("match", project_id, target, requested_id=run_id), run_id

    def get(self, job_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or (project_id is not None and job.get("project_id") != project_id):
                return None
            return dict(job)
