"""The /metrics renderer (harness/metrics.py): Prometheus text from the
project tree, the running agents and the verdicts -- pure, no Beads or DB."""

from harness import metrics

TREE = [
    {
        "id": "proj-a", "title": "Game \"A\"", "status": "open",
        "epics": [
            {"id": "proj-a.1", "title": "Epic one", "status": "open", "stories": [
                {"id": "proj-a.1.1", "title": "done", "status": "closed"},
                {"id": "proj-a.1.2", "title": "working", "status": "in_progress"},
                {"id": "proj-a.1.3", "title": "stuck", "status": "in_progress", "labels": ["human"]},
                {"id": "proj-a.1.4", "title": "todo", "status": "open"},
            ]},
        ],
    },
    {"id": "proj-b", "title": "Held", "status": "open", "epics": []},
]


def _series(text, name):
    return [line for line in text.splitlines() if line.startswith(name + "{") or line.startswith(name + " ")]


def test_tickets_progress_and_human_flags():
    text = metrics.render(TREE, [], {"proj-b": "on hold"}, {}, [], None)
    tickets = _series(text, "custos_tickets")
    assert 'custos_tickets{project="proj-a",title="Game \\"A\\"",status="closed"} 1' in tickets
    assert 'custos_tickets{project="proj-a",title="Game \\"A\\"",status="in_progress"} 2' in tickets
    assert 'custos_tickets{project="proj-a",title="Game \\"A\\"",status="human"} 1' in tickets
    assert any(l.startswith('custos_project_progress_ratio{project="proj-a"') and l.endswith(" 0.2500") for l in _series(text, "custos_project_progress_ratio"))
    assert any('ticket="proj-a.1.3"' in l for l in _series(text, "custos_human_ticket"))
    held = [l for l in _series(text, "custos_project_info") if 'project="proj-b"' in l]
    assert held and held[0].endswith(" 1") and 'hold="on hold"' in held[0]


def test_agents_and_verdicts():
    agents = [{"seat_id": "renderer", "ticket_id": "proj-a.1.2", "title": "working", "started_at": "t0", "idle_seconds": 42}]
    recent = [{"issue_id": "proj-a.1.1", "verdict": "pass", "seat_id": "renderer", "reasoning": "line one\nline two", "at": 1700000000.0}]
    text = metrics.render(TREE, agents, {}, {"pass": 3, "fail": 1}, recent, 5)
    assert "custos_agents_running 1" in text
    assert 'custos_agent_idle_seconds{seat="renderer",ticket="proj-a.1.2",title="working",started="t0"} 42' in text
    assert 'custos_verifications{verdict="fail"} 1' in text
    assert "custos_tickets_ready 5" in text
    recent_line = _series(text, "custos_verdict_recent")[0]
    assert "\n" not in recent_line and recent_line.endswith(" 1700000000")
    assert text.endswith("\n")
