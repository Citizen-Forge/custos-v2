"""Prometheus metrics for Custos: projects, tickets, agents, verdicts.

Served at GET /metrics by the API (no auth, like /health) so a Prometheus on
the LAN can scrape it and Grafana can draw the board, the agents and their
work beside the GPUs that run them. Plain text exposition format -- no client
library: every value here is a gauge read fresh from the state the API
already holds (the cached project tree, Beads' in-progress list, the
verifications table), so there is nothing to accumulate in-process.

`render()` is pure: give it the gathered state, get the text. The route does
the gathering, so the format is testable without Beads or a database.
"""

STATUSES = ("open", "in_progress", "blocked", "closed")


def _esc(value) -> str:
    return str(value if value is not None else "").replace("\\", "\\\\").replace("\n", " ").replace('"', '\\"')


def _labels(**kv) -> str:
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in kv.items()) + "}"


def _is_human(issue: dict) -> bool:
    return "human" in (issue.get("labels") or [])


def _stories(project: dict):
    for epic in project.get("epics", []):
        for story in epic.get("stories", []):
            yield epic, story
            for sub in story.get("subtasks", []):
                yield epic, sub


def render(tree: list[dict], agents: list[dict], held: dict[str, str],
           verdicts: dict[str, int], recent: list[dict], ready_count: int | None) -> str:
    lines: list[str] = []

    def metric(name: str, kind: str, help_text: str):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")

    metric("custos_up", "gauge", "1 while the Custos API answers.")
    lines.append("custos_up 1")

    metric("custos_project_info", "gauge", "One series per project; 1 when on dispatch hold, else 0.")
    for p in tree:
        lines.append("custos_project_info" + _labels(project=p["id"], title=p.get("title", ""),
                     status=p.get("status", ""), hold=held.get(p["id"], "")) + f" {int(p['id'] in held)}")

    metric("custos_tickets", "gauge", "Stories per project by status.")
    metric_rows = []
    for p in tree:
        counts = {s: 0 for s in STATUSES}
        human = 0
        for _epic, story in _stories(p):
            st = story.get("status", "open")
            counts[st if st in counts else "open"] += 1
            human += int(_is_human(story))
        for st, n in counts.items():
            metric_rows.append("custos_tickets" + _labels(project=p["id"], title=p.get("title", ""), status=st) + f" {n}")
        metric_rows.append("custos_tickets" + _labels(project=p["id"], title=p.get("title", ""), status="human") + f" {human}")
    lines.extend(metric_rows)

    metric("custos_project_progress_ratio", "gauge", "Closed stories over all stories, per project.")
    for p in tree:
        stories = [s for _e, s in _stories(p)]
        done = sum(1 for s in stories if s.get("status") == "closed")
        lines.append("custos_project_progress_ratio" + _labels(project=p["id"], title=p.get("title", ""))
                     + f" {done / len(stories) if stories else 0:.4f}")

    metric("custos_epic_progress_ratio", "gauge", "Closed stories over all stories, per epic.")
    for p in tree:
        for epic in p.get("epics", []):
            st = epic.get("stories", [])
            done = sum(1 for s in st if s.get("status") == "closed")
            lines.append("custos_epic_progress_ratio" + _labels(project=p["id"], epic=epic["id"],
                         title=epic.get("title", ""), status=epic.get("status", ""))
                         + f" {done / len(st) if st else 0:.4f}")

    metric("custos_human_ticket", "gauge", "1 per ticket flagged for a person.")
    for p in tree:
        for _epic, story in _stories(p):
            if _is_human(story):
                lines.append("custos_human_ticket" + _labels(project=p["id"], ticket=story["id"],
                             title=story.get("title", ""), status=story.get("status", "")) + " 1")

    metric("custos_agents_running", "gauge", "Agents working a ticket now.")
    lines.append(f"custos_agents_running {len(agents)}")
    metric("custos_agent_idle_seconds", "gauge", "Seconds since the agent last took a graph step, per running agent.")
    for a in agents:
        idle = a.get("idle_seconds")
        lines.append("custos_agent_idle_seconds" + _labels(seat=a.get("seat_id") or "", ticket=a.get("ticket_id"),
                     title=a.get("title") or "", started=a.get("started_at") or "")
                     + f" {idle if idle is not None else -1}")

    if ready_count is not None:
        metric("custos_tickets_ready", "gauge", "Tickets ready to dispatch now.")
        lines.append(f"custos_tickets_ready {ready_count}")

    metric("custos_verifications", "gauge", "Recorded verdicts by outcome.")
    for verdict, n in sorted(verdicts.items()):
        lines.append("custos_verifications" + _labels(verdict=verdict) + f" {n}")

    metric("custos_verdict_recent", "gauge", "Recent verdicts; the value is when, as a Unix timestamp.")
    for v in recent:
        lines.append("custos_verdict_recent" + _labels(ticket=v["issue_id"], verdict=v["verdict"], seat=v["seat_id"],
                     reasoning=(v.get("reasoning") or "")[:160]) + f" {v['at']:.0f}")
    return "\n".join(lines) + "\n"
