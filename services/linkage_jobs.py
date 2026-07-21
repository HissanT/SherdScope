"""Persistent single-worker priority queue for linkage operations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid
from typing import Any, Callable


JOBS_NAME = "metadata_jobs.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty() -> dict[str, Any]:
    return {
        "schema_version": 1, "next_sequence": 1, "active": None,
        "paused": None, "queued": [], "recent": [], "updated_at": _now(),
    }


def _path(project_path: Path) -> Path:
    return Path(project_path) / "cards" / JOBS_NAME


def _read(project_path: Path) -> dict[str, Any]:
    path = _path(project_path)
    if not path.exists():
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        base = _empty()
        base.update(value if isinstance(value, dict) else {})
        return base
    except (OSError, ValueError, json.JSONDecodeError):
        return _empty()


def _write(project_path: Path, state: dict[str, Any]) -> None:
    path = _path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class LinkageJobCoordinator:
    def __init__(self, executor: Callable[..., dict[str, Any]]):
        self.executor = executor
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._private_payloads: dict[str, dict[str, Any]] = {}

    def enqueue(self, project_id: str, project_path: Path, kind: str,
                priority: int, payload: dict[str, Any] | None = None,
                dedupe_key: str | None = None,
                private_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(payload or {})
        dedupe_key = dedupe_key or f"{kind}:{json.dumps(payload, sort_keys=True)}"
        with self._lock:
            state = _read(project_path)
            candidates = ([state.get("active"), state.get("paused")] +
                          list(state.get("queued", [])))
            existing = next((item for item in candidates
                             if item and item.get("dedupe_key") == dedupe_key), None)
            if existing:
                return dict(existing)
            sequence = int(state.get("next_sequence", 1) or 1)
            state["next_sequence"] = sequence + 1
            job = {
                "id": uuid.uuid4().hex[:16], "kind": kind,
                "priority": int(priority), "sequence": sequence,
                "dedupe_key": dedupe_key, "payload": payload,
                "status": "queued", "created_at": _now(),
                "progress": {"current": 0, "total": 0, "message": "Queued"},
            }
            state.setdefault("queued", []).append(job)
            if private_payload:
                self._private_payloads[job["id"]] = dict(private_payload)
            _write(project_path, state)
            self._start_worker(project_id, Path(project_path))
            return dict(job)

    def snapshot(self, project_id: str, project_path: Path,
                 ensure_worker: bool = True) -> dict[str, Any]:
        with self._lock:
            state = _read(project_path)
            if ensure_worker and (state.get("active") or state.get("paused") or
                                  state.get("queued")):
                self._start_worker(project_id, Path(project_path), recover=True)
            return state

    def update_progress(self, project_path: Path, job_id: str, current: int,
                        total: int, message: str) -> None:
        with self._lock:
            state = _read(project_path)
            active = state.get("active")
            if not active or active.get("id") != job_id:
                return
            active["progress"] = {
                "current": int(current), "total": int(total), "message": str(message),
            }
            active["resume_cursor"] = {
                "current": int(current), "total": int(total), "message": str(message),
            }
            _write(project_path, state)

    def has_higher_priority(self, project_path: Path, priority: int) -> bool:
        with self._lock:
            state = _read(project_path)
            return any(int(job.get("priority", 100)) < int(priority)
                       for job in state.get("queued", []))

    def _start_worker(self, project_id: str, project_path: Path,
                      recover: bool = False) -> None:
        worker = self._workers.get(project_id)
        if worker and worker.is_alive():
            return
        if recover:
            state = _read(project_path)
            active = state.get("active")
            if active:
                active["status"] = "queued"
                active.setdefault("payload", {})["resume"] = True
                state.setdefault("queued", []).append(active)
                state["active"] = None
                _write(project_path, state)
        worker = threading.Thread(
            target=self._work, args=(project_id, project_path), daemon=True,
            name=f"linkage-{project_id[:12]}")
        self._workers[project_id] = worker
        worker.start()

    def _next_job(self, state: dict[str, Any]) -> dict[str, Any] | None:
        queued = list(state.get("queued", []))
        paused = state.get("paused")
        if paused:
            queued.append(paused)
        if not queued:
            return None
        selected = min(queued, key=lambda item: (
            int(item.get("priority", 100)), int(item.get("sequence", 0))))
        if paused and selected.get("id") == paused.get("id"):
            state["paused"] = None
            selected.setdefault("payload", {})["resume"] = True
        else:
            state["queued"] = [item for item in state.get("queued", [])
                               if item.get("id") != selected.get("id")]
        return selected

    def _work(self, project_id: str, project_path: Path) -> None:
        while True:
            with self._lock:
                state = _read(project_path)
                job = self._next_job(state)
                if not job:
                    state["active"] = None
                    _write(project_path, state)
                    return
                job["status"] = "running"
                job["started_at"] = job.get("started_at") or _now()
                state["active"] = job
                _write(project_path, state)
            try:
                execution_job = dict(job)
                execution_job["payload"] = {
                    **job.get("payload", {}),
                    **self._private_payloads.get(str(job.get("id")), {}),
                }
                result = self.executor(
                    project_id, project_path, execution_job,
                    lambda: self.has_higher_priority(project_path,
                                                     int(job.get("priority", 100)))) or {}
                paused = bool(result.get("paused"))
                with self._lock:
                    state = _read(project_path)
                    current = state.get("active") or job
                    state["active"] = None
                    if paused:
                        current["status"] = "paused"
                        current.setdefault("payload", {})["resume"] = True
                        current["progress"] = result.get(
                            "progress", current.get("progress", {}))
                        current["resume_cursor"] = result.get(
                            "resume_cursor", current.get("resume_cursor",
                                                         current.get("progress", {})))
                        state["paused"] = current
                    else:
                        current["status"] = "succeeded"
                        current["finished_at"] = _now()
                        current["result"] = result
                        state.setdefault("recent", []).insert(0, current)
                        state["recent"] = state["recent"][:20]
                    _write(project_path, state)
            except Exception as exc:
                with self._lock:
                    state = _read(project_path)
                    failed = state.get("active") or job
                    failed["status"] = "failed"
                    failed["finished_at"] = _now()
                    failed["error"] = str(exc)
                    state["active"] = None
                    state.setdefault("recent", []).insert(0, failed)
                    state["recent"] = state["recent"][:20]
                    _write(project_path, state)
            finally:
                with self._lock:
                    self._private_payloads.pop(str(job.get("id")), None)
