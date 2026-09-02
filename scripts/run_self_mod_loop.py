"""
The trusted control plane for unattended self-modification.

The pipeline already existed and worked; what it was not was reachable.
Every stage was a manual `docker compose run` a human issued, so a ticket
asking the harness to improve itself could never actually be worked --
demonstrated expensively on 2026-09-01, when a seat spent ~12 hours
searching for src/harness/tools.py, a file agents cannot open at all
(permissions.check_within_workspace confines them to their workspace).

This service closes that gap without weakening containment. It runs
inside sandbox-runner, the one service that legitimately holds the Docker
socket (PLAN.md Phase 7), and polls the proposals table:

    pending        -> sandbox it (needs Docker: builds an image, runs tests)
    reviewed       -> reviewer.py records a verdict; a favourable one
                      parks the proposal at awaiting_human
    awaiting_human -> nothing happens here. A person approves or rejects
                      it on the dashboard.
    approved       -> deploy it (needs Docker: rebuild + restart services)

`approved` is reachable only by a human saying yes (2026-09-01). The
reviewing agent's verdict and the sandbox evidence inform that decision;
they no longer make it.

An agent's only interaction is creating a proposal row. It never gains
Docker access, never invokes these stages directly, and cannot reach the
host daemon -- the trusted side reaches out and picks the work up. That
is the same posture the project already takes for tool proposals: the
agent proposes, a separate trusted path decides and acts.

Both existing hard gates are preserved exactly:

- A proposal whose sandbox run was not clean never deploys, regardless of
  what any reviewer said (run_self_mod_deploy._has_clean_sandbox_result).
- Deployment refuses to run onto a dirty tree, and rolls the applied diff
  back if the final real-tree test run fails.

Two rails remain, even with a human in the path:

- SELF_MOD_MAX_DEPLOYS_PER_DAY caps how many changes land in a rolling
  24h, so a run of quick approvals cannot stack changes faster than
  anyone can notice one going wrong.
- SELF_MOD_AUTOPILOT=off disables the loop entirely without redeploying
  anything -- the kill switch for the one loop that can disable the thing
  that would fix it.
"""

import logging
import os
import time

import psycopg

from harness import self_mod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("self-mod-loop")

POLL_SECONDS = int(os.environ.get("SELF_MOD_POLL_INTERVAL", "60"))
MAX_DEPLOYS_PER_DAY = int(os.environ.get("SELF_MOD_MAX_DEPLOYS_PER_DAY", "3"))


def autopilot_enabled() -> bool:
    return os.environ.get("SELF_MOD_AUTOPILOT", "on").lower() not in ("off", "0", "false", "no")


def _sandbox_pending(conn) -> bool:
    """Sandbox one pending proposal. Imported lazily: the sandbox module
    pulls in Docker-dependent code that only makes sense in this service."""
    pending = self_mod.list_by_status(conn, "pending")
    if not pending:
        return False
    from run_self_mod_sandbox import sandbox_proposal

    proposal = pending[0]
    log.info("sandboxing proposal #%s", proposal["id"])
    sandbox_proposal(conn, proposal)
    return True


def _deploy_approved(conn) -> bool:
    approved = self_mod.list_by_status(conn, "approved")
    if not approved:
        return False

    # Everything here was explicitly approved by a person; the cap is a
    # brake on how fast approvals can stack up, not a substitute for the
    # decision itself.
    landed = self_mod.deployed_since(conn, 24 * 3600)
    if landed >= MAX_DEPLOYS_PER_DAY:
        log.warning(
            "holding %s approved proposal(s): %s already deployed in the last 24h "
            "(SELF_MOD_MAX_DEPLOYS_PER_DAY=%s)",
            len(approved), landed, MAX_DEPLOYS_PER_DAY,
        )
        return False

    from run_self_mod_deploy import deploy_proposal

    proposal = approved[0]
    log.info("deploying proposal #%s", proposal["id"])
    deploy_proposal(conn, proposal)

    # Report back onto the ticket that asked for this, if any. Read the
    # row again: deploy_proposal may have declined to deploy (a proposal
    # without a clean sandbox result is refused there, not here), and a
    # ticket must not be closed for a deployment that did not happen.
    after = self_mod.get(conn, proposal["id"])
    if after and after.get("status") == "deployed":
        from harness import self_mod_ticket

        self_mod_ticket.report_deployment(conn, after)
    return True


def run_one_cycle(conn) -> str:
    """One pass. Returns what it did, so the loop is testable without
    running forever."""
    if not autopilot_enabled():
        return "autopilot off"
    if _sandbox_pending(conn):
        return "sandboxed"
    if _deploy_approved(conn):
        return "deployed"
    return "idle"


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]
    log.info(
        "self-mod loop started: polling every %ss, max %s deploy(s)/24h, autopilot %s",
        POLL_SECONDS, MAX_DEPLOYS_PER_DAY, "on" if autopilot_enabled() else "OFF",
    )
    while True:
        try:
            with psycopg.connect(conn_string, autocommit=True) as conn:
                self_mod.init_table(conn)
                outcome = run_one_cycle(conn)
        except Exception:
            log.exception("self-mod cycle failed, continuing")
            outcome = "error"
        if outcome in ("idle", "error", "autopilot off"):
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
