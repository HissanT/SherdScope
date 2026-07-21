import threading
import time

from services.linkage_jobs import LinkageJobCoordinator, _empty, _read, _write


def _wait_until(predicate, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("timed out waiting for linkage job")


def test_priority_job_preempts_and_bulk_resumes_from_checkpoint(tmp_path):
    project = tmp_path / "project"
    started = threading.Event()
    finish_unit = threading.Event()
    calls = []

    def execute(project_id, project_path, job, should_pause):
        resume = bool(job.get("payload", {}).get("resume"))
        calls.append((job["kind"], resume))
        if job["kind"] == "bulk_link" and not resume:
            started.set()
            assert finish_unit.wait(2)
            return {"paused": should_pause(), "progress": {"current": 1, "total": 3}}
        return {"paused": False}

    coordinator = LinkageJobCoordinator(execute)
    bulk = coordinator.enqueue("p", project, "bulk_link", 100,
                               dedupe_key="bulk")
    assert started.wait(2)
    priority = coordinator.enqueue("p", project, "figure_reread", 10,
                                   {"figure_key": "2.1"}, dedupe_key="figure:2.1")
    duplicate = coordinator.enqueue("p", project, "figure_reread", 10,
                                    {"figure_key": "2.1"}, dedupe_key="figure:2.1")
    assert duplicate["id"] == priority["id"]
    finish_unit.set()
    _wait_until(lambda: len(_read(project).get("recent", [])) == 2)

    assert calls == [("bulk_link", False), ("figure_reread", False),
                     ("bulk_link", True)]
    completed = {job["id"] for job in _read(project)["recent"]}
    assert completed == {bulk["id"], priority["id"]}


def test_stale_running_job_is_recovered_and_retries_only_active_unit(tmp_path):
    project = tmp_path / "project"
    state = _empty()
    state["active"] = {
        "id": "stale", "kind": "bulk_link", "priority": 100,
        "sequence": 1, "dedupe_key": "bulk", "payload": {},
        "status": "running", "progress": {"current": 7, "total": 20},
    }
    _write(project, state)
    observed = []

    def execute(project_id, project_path, job, should_pause):
        observed.append(job)
        return {"paused": False}

    coordinator = LinkageJobCoordinator(execute)
    coordinator.snapshot("p", project, ensure_worker=True)
    _wait_until(lambda: bool(_read(project).get("recent")))

    assert observed[0]["id"] == "stale"
    assert observed[0]["payload"]["resume"] is True
    assert _read(project)["recent"][0]["status"] == "succeeded"


def test_failed_job_does_not_stop_the_next_queued_job(tmp_path):
    project = tmp_path / "project"
    release = threading.Event()
    calls = []

    def execute(project_id, project_path, job, should_pause):
        calls.append(job["kind"])
        if job["kind"] == "bad":
            assert release.wait(2)
            raise RuntimeError("isolated failure")
        return {"paused": False}

    coordinator = LinkageJobCoordinator(execute)
    bad = coordinator.enqueue("p", project, "bad", 10)
    good = coordinator.enqueue("p", project, "good", 10)
    release.set()
    _wait_until(lambda: len(_read(project).get("recent", [])) == 2)

    results = {job["id"]: job for job in _read(project)["recent"]}
    assert results[bad["id"]]["status"] == "failed"
    assert results[good["id"]]["status"] == "succeeded"
    assert calls == ["bad", "good"]
