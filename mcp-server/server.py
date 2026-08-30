"""
MCP server for the "interface agent" design (PLAN.md, 2026-08-30): lets
a Claude Code sidecar session -- run wherever the user runs Claude Code,
not inside this project's own docker-compose stack -- talk to a live
Custos v2 deployment over plain HTTP, using the exact same api.py this
project already ships and tests. Deliberately a thin translation layer
with no logic of its own: every tool here is a direct call to one
api.py endpoint, so the API stays the single source of truth for what
"the harness's current state" means, and this server can't drift from
it or duplicate its logic.

Config via env vars, not hardcoded, since this runs on whatever machine
the user's Claude Code session runs on, separate from the deployment:
    CUSTOS_API_URL   -- base URL of a running api.py, e.g.
                        http://192.168.100.231:8000 for the Unraid
                        deployment. Defaults to localhost for local dev.
    CUSTOS_API_TOKEN -- matches the deployment's API_AUTH_TOKEN if one
                        is set (see auth.py) -- omit if that deployment
                        left auth open.

Every tool fails soft: a network error, timeout, or non-2xx response
comes back as a readable string the agent can react to and relay to the
user, never an unhandled exception that would kill the MCP connection --
same posture as slack.py/avatar.py elsewhere in this project for any
call to something outside the harness's own control.

Register with Claude Code:
    claude mcp add --transport stdio custos \
        -- python /path/to/mcp-server/server.py

Run standalone to sanity-check it starts:
    CUSTOS_API_URL=http://localhost:8000 python mcp-server/server.py
"""

import json
import os

import httpx
from mcp.server.mcpserver import MCPServer

API_URL = os.environ.get("CUSTOS_API_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("CUSTOS_API_TOKEN")

mcp = MCPServer("custos-v2")


def _client() -> httpx.Client:
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    # 30s, not a smaller "should be instant" default: /projects walks
    # Beads' hierarchy with one `bd` subprocess call per epic (N+1, see
    # api.py's list_projects), and each individual `bd` invocation was
    # measured live at ~3-5s against this project's real (Dolt-backed)
    # workspace -- inherent to `bd`, not this server, but real enough
    # that a naive short timeout here would fail on real data.
    return httpx.Client(base_url=API_URL, headers=headers, timeout=30)


def _call(method: str, path: str, **kwargs) -> str:
    """Every tool below goes through this -- one place that turns a
    network failure or an API-level error into a string the calling
    agent can read and act on, rather than a crash. Response bodies are
    returned as compact JSON text: good enough for an LLM to read
    directly, and keeps this file from needing a bespoke formatter per
    endpoint."""
    try:
        response = _client().request(method, path, **kwargs)
    except httpx.HTTPError as e:
        return f"error reaching {API_URL}: {e}"

    if response.status_code == 401:
        return "error: 401 unauthorized -- check CUSTOS_API_TOKEN matches this deployment's API_AUTH_TOKEN"
    if not response.is_success:
        return f"error: {response.status_code} -- {response.text}"

    try:
        return json.dumps(response.json())
    except ValueError:
        return response.text


@mcp.tool()
def list_projects() -> str:
    """The full project -> epic -> story tree, sorted by priority. Check this before deciding
    whether a new idea belongs in an existing project or needs its own."""
    return _call("GET", "/projects")


@mcp.tool()
def list_tickets(status: str = "ready") -> str:
    """List tickets by status: 'ready' (unclaimed work), 'in_progress', or 'human' (flagged
    for a human decision -- check this to see what's waiting on you)."""
    return _call("GET", "/tickets", params={"status": status})


@mcp.tool()
def get_ticket(issue_id: str) -> str:
    """Full detail on one ticket by id, including its notes and close reason if closed."""
    return _call("GET", f"/tickets/{issue_id}")


@mcp.tool()
def respond_to_ticket(issue_id: str, response: str) -> str:
    """Answer a ticket that's flagged for human review, closing it with your response recorded."""
    return _call("POST", f"/tickets/{issue_id}/respond", json={"response": response})


@mcp.tool()
def dismiss_ticket(issue_id: str, reason: str | None = None) -> str:
    """Dismiss a ticket flagged for human review without answering it, closing it with an
    optional reason."""
    return _call("POST", f"/tickets/{issue_id}/dismiss", json={"reason": reason})


@mcp.tool()
def list_seats() -> str:
    """The current specialist seat roster, each with its outcomes (closed/refused/still-open),
    verification pass/fail record, and queue depth."""
    return _call("GET", "/seats")


@mcp.tool()
def get_outcomes(actor: str) -> str:
    """Closed/refused/still-open ticket counts for one seat or role by id."""
    return _call("GET", f"/outcomes/{actor}")


@mcp.tool()
def create_project(name: str, description: str, priority: int) -> str:
    """Create a new top-level project for a genuinely new body of work. priority is 0-4
    (0=highest) -- set it relative to what list_projects already shows. Epics (create_epic)
    and stories (create_story) nest under this."""
    return _call("POST", "/projects", json={"name": name, "description": description, "priority": priority})


@mcp.tool()
def create_epic(project_id: str, title: str, description: str) -> str:
    """Create an epic under an existing project (see list_projects/create_project) -- one
    coherent slice of that project's work. Use once per epic, not per story."""
    return _call("POST", f"/projects/{project_id}/epics", json={"title": title, "description": description})


@mcp.tool()
def create_story(epic_id: str, title: str, description: str) -> str:
    """Add one concrete, individually-workable story under an epic (created via create_epic) --
    a real task a seat could actually pick up and finish, not a vague restatement of the epic."""
    return _call("POST", f"/epics/{epic_id}/stories", json={"title": title, "description": description})


if __name__ == "__main__":
    mcp.run()
