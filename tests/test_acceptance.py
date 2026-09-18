"""Machine-checkable acceptance criteria (harness/acceptance.py) and the
verifier's deterministic pre-pass.

The point of the pre-pass: criteria that are mechanical should be decided in
code, not handed to a model. A failing check must fail the ticket without a
model call at all, and a ticket whose criteria are entirely mechanical must
be able to pass without one. The model is still asked whenever the ticket
carries free-text criteria that need judgment.
"""

import importlib.util
import json
import pathlib

from harness import acceptance, beads, providers, reflection, verifier, workspaces


def _issue(checks=None, criteria=None, extra=None):
    metadata = dict(extra or {})
    if checks is not None:
        metadata[beads.CHECKS_KEY] = json.dumps(checks)
    if criteria is not None:
        metadata["acceptance_criteria"] = criteria
    return {"id": "p.1.1", "status": "closed", "title": "t", "description": "d",
            "metadata": metadata}


# -- acceptance.evaluate -------------------------------------------------


def test_evaluate_passes_mechanical_checks(tmp_path, monkeypatch):
    (tmp_path / "project.godot").write_text("config")
    (tmp_path / "README.md").write_text("# Title\n\n## Test\nrun the suite\n")
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([
        {"type": "file_exists", "path": "project.godot"},
        {"type": "file_absent", "path": "old.txt"},
        {"type": "contains", "path": "README.md", "text": "## Test"},
        {"type": "matches", "path": "README.md", "pattern": r"run the suite"},
    ])

    results = acceptance.evaluate(issue, tests=None)

    assert acceptance.failed(results) == []
    assert all(r["ok"] for r in results)


def test_evaluate_reports_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([
        {"type": "file_exists", "path": "missing.godot"},
        {"type": "contains", "path": "README.md", "text": "nope"},
    ])

    results = acceptance.evaluate(issue, tests=None)

    assert len(acceptance.failed(results)) == 2


def test_tests_at_least_check(tmp_path, monkeypatch):
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([{"type": "tests_at_least", "count": 5}])

    assert acceptance.evaluate(issue, {"exit": 0, "ran": 5, "failed": 0})[0]["ok"]
    assert not acceptance.evaluate(issue, {"exit": 0, "ran": 4, "failed": 0})[0]["ok"]
    assert not acceptance.evaluate(issue, {"exit": 1, "ran": 5, "failed": 1})[0]["ok"]
    assert not acceptance.evaluate(issue, None)[0]["ok"]


def test_a_check_cannot_read_outside_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([{"type": "file_exists", "path": "../outside"}])

    result = acceptance.evaluate(issue, tests=None)[0]

    assert not result["ok"]
    assert "escapes" in result["detail"]


def test_an_unknown_check_type_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([{"type": "launch_missiles"}])

    assert not acceptance.evaluate(issue, tests=None)[0]["ok"]


def test_malformed_checks_metadata_reads_as_no_checks():
    issue = {"id": "p.1.1", "metadata": {beads.CHECKS_KEY: "{not json"}}
    assert beads.acceptance_checks(issue) == []


# -- verifier pre-pass ---------------------------------------------------


class _MustNotRun:
    def invoke(self, *a, **k):
        raise AssertionError("the model must not be called for a mechanical verdict")


def _stub_verifier(monkeypatch, issue, tests, rec):
    monkeypatch.setattr(verifier, "awaiting_verdict", lambda conn, iss: True)
    monkeypatch.setattr(verifier.beads, "show", lambda tid: issue)
    monkeypatch.setattr(verifier, "_tests_for", lambda iss: tests)
    monkeypatch.setattr(
        verifier.verifications, "record", lambda *a, **k: rec.update(verdict=a[3], reasoning=a[4])
    )
    monkeypatch.setattr(verifier, "land", lambda tid: rec.update(landed=True))
    monkeypatch.setattr(
        verifier, "requeue_for_rework", lambda *a, **k: rec.update(requeued=True)
    )
    monkeypatch.setattr(verifier.beads, "flag_for_human", lambda *a, **k: rec.update(flagged=True))


def test_a_failed_check_fails_without_calling_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([{"type": "file_exists", "path": "MISSING"}])
    rec = {}
    _stub_verifier(monkeypatch, issue, tests=None, rec=rec)

    result = verifier.verify_ticket(None, "p.1.1", _MustNotRun())

    assert result["verdict"] == "fail"
    assert rec["verdict"] == "fail"
    assert rec.get("requeued") is True
    assert "MISSING" in result["reasoning"]


def test_all_checks_passing_with_no_prose_passes_without_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([{"type": "tests_at_least", "count": 1}])
    rec = {}
    _stub_verifier(monkeypatch, issue, tests={"exit": 0, "ran": 3, "passed": 3, "failed": 0}, rec=rec)

    result = verifier.verify_ticket(None, "p.1.1", _MustNotRun())

    assert result["verdict"] == "pass"
    assert rec.get("landed") is True


def test_prose_criteria_still_go_to_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(workspaces, "tree_for_ticket", lambda tid: str(tmp_path))
    issue = _issue([{"type": "tests_at_least", "count": 1}], criteria="the code is elegant")
    rec = {}
    _stub_verifier(monkeypatch, issue, tests={"exit": 0, "ran": 3, "passed": 3, "failed": 0}, rec=rec)

    class Model:
        def __init__(self):
            self.called = False

        def invoke(self, prompt):
            self.called = True
            assert "elegant" in prompt

            class R:
                content = json.dumps({"verdict": "pass", "reasoning": "fine"})

            return R()

    model = Model()
    result = verifier.verify_ticket(None, "p.1.1", model)

    assert model.called
    assert result["verdict"] == "pass"


# -- role-specific model env --------------------------------------------


def test_reflection_model_honours_reflection_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        providers, "build_chat_model", lambda cfg: captured.setdefault("cfg", cfg)
    )
    monkeypatch.setenv("REFLECTION_MODEL_BASE_URL", "http://local:8080/v1")
    monkeypatch.setenv("REFLECTION_MODEL_NAME", "local-model")

    reflection.build_model()

    assert captured["cfg"].base_url == "http://local:8080/v1"
    assert captured["cfg"].model == "local-model"


def _load_scheduler():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_scheduler.py"
    spec = importlib.util.spec_from_file_location("run_scheduler_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scheduler_provider_honours_role_env(monkeypatch):
    monkeypatch.setenv("PROGRESS_MODEL_BASE_URL", "http://local:8080/v1")
    monkeypatch.setenv("PROGRESS_MODEL_NAME", "local-model")
    mod = _load_scheduler()

    progress = mod._provider("progress", 2000)
    assert progress.base_url == "http://local:8080/v1"
    assert progress.model == "local-model"

    # A role with no override still falls through to the shared chain.
    product_owner = mod._provider("product-owner", 4000)
    assert product_owner.base_url != "http://local:8080/v1"
