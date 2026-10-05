"""
The acceptance-criteria verification loop's judgment: given a closed
ticket that had explicit acceptance criteria (beads.acceptance_criteria),
decide whether the actual work meets them. Same single-shot judgment
shape as reviewer.py -- gather real evidence (the ticket's own notes and
close reason, not a self-report from the seat that did the work), ask a
model, record a verdict. This is deliberately a SEPARATE agent's call
from the seat that did the work, not a self-grade -- same reasoning as
Phase 7's reviewer being separate from the overwatch agent that proposes
a tool.

This is the mechanism the user asked for in place of "Laurels" (a
human-feedback surface, deferred): automated positive/negative signal a
seat can actually earn without a human rating every ticket, and real
quality data for meta_agent.py to reason about beyond "was this ticket
refused."
"""

import json
import logging
import os
import re
import shlex
from contextlib import contextmanager

import psycopg

from . import acceptance, beads, toolchain, verifications, workspaces

log = logging.getLogger(__name__)

# How many times a failed verification sends a ticket back to an agent
# before it is parked for a human instead. The point of re-queuing is that
# the verifier's finding gets acted on, not merely recorded -- but a
# reopened ticket re-blocks its dependents (see the comment on the old
# fail branch), so the loop has to be finite. Two attempts rides out a
# verdict the agent can actually fix, then hands a genuine impasse to a
# person rather than churning forever.
MAX_VERIFIER_REWORKS = int(os.environ.get("MAX_VERIFIER_REWORKS", "2"))


ROLE = "verifier"


def build_model(routing=None, gate=None):
    """The verifier's model, from VERIFIER_* falling back to LOCAL_*.

    Shared by the standalone entry point and by the dispatcher's
    verify-on-close step, so the two cannot drift into judging tickets
    with differently-configured models.

    With no VERIFIER_* model configured it is the workers' own LOCAL chain,
    primary AND fallback, routed. It used to be the primary alone, so when
    DeepSeek ran out of balance (2026-09-30) nothing could be verified or
    landed even though the fallback was serving every agent. Pass the
    dispatcher's routing table and gate to share their cooldowns and the
    fallback's concurrency limit."""
    from .providers import ProviderConfig, build_chat_model

    if not os.environ.get("VERIFIER_MODEL_BASE_URL"):
        from .routing import ConcurrencyGate, RoutedModel, RoutingTable
        from .worker import _chain_from_env

        chain = _chain_from_env(
            "LOCAL",
            "http://host.docker.internal:11434/v1",
            "qwen2.5:7b-instruct",
            max_tokens=int(os.environ.get("VERIFIER_MAX_TOKENS", "6000")),
        )
        routing = routing or RoutingTable({})
        routing._chains.setdefault(ROLE, chain)
        return RoutedModel(ROLE, routing, gate or ConcurrencyGate())

    provider_cfg = ProviderConfig(
        name="verifier",
        base_url=os.environ.get(
            "VERIFIER_MODEL_BASE_URL",
            os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        ),
        model=os.environ.get(
            "VERIFIER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")
        ),
        # Falls back to LOCAL_MODEL_API_KEY for the same reason base_url
        # and model already fall back to their LOCAL_* counterparts: a
        # VERIFIER_* override is opt-out, so an unset key must not mean
        # "no key" when the inherited base_url is an authenticated one.
        api_key=os.environ.get("VERIFIER_MODEL_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY")),
        max_tokens=int(os.environ.get("VERIFIER_MAX_TOKENS", "6000")),
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if os.environ.get(
                "VERIFIER_MODEL_DISABLE_THINKING", os.environ.get("LOCAL_MODEL_DISABLE_THINKING")
            )
            else None
        ),
    )
    return build_chat_model(provider_cfg)


PROMPT = """You are verifying whether completed work actually meets its stated acceptance \
criteria. You are a SEPARATE reviewer, not the agent that did the work -- judge honestly \
from the evidence, don't assume good faith just because the work was marked complete.

Ticket: {title}
Description: {description}
Acceptance criteria: {acceptance_criteria}

How the assigned agent says it was completed (close reason): {close_reason}
Accumulated notes from the work: {notes}

Earlier verifier verdicts on previous attempts are deliberately left out of these notes: judge \
this attempt from its own evidence. Notes describing what was WRONG BEFORE the change (in the \
past tense) are history, not the current state -- the diff and the measured results say what \
holds now.

The actual code change this ticket produced, as a unified diff:
{diff}

The file list at the top of each commit (the --stat) is COMPLETE. A very large change has some \
patches trimmed or not shown, and that is said in the diff where it happens. A file in the file \
list IS part of this change even when its patch is not shown: never fail a ticket because a file \
or function is "missing from the diff" when the file list names it or a trimmed-patch note \
covers it -- check the measured test result instead.

Read the diff as a diff. A line starting with `-` was REMOVED by this ticket and is NOT in \
the code any more; a line starting with `+` was ADDED and IS the current state. Never report \
a `-` line as a present-day problem -- if the ticket deleted a bad line, that is the fix, not \
the fault. When both a `-` and a `+` version of the same setting appear, only the `+` one is live.

The project's configuration and readme files AS THEY STAND RIGHT NOW:
{current_files}

Those are the live contents, read from disk just now -- not the diff, not a claim. They are \
read from the ticket's OWN working tree, which is the state this ticket's change produces and \
is not yet on the project's integration branch (that happens only if you pass it). If a \
criterion asks whether a file exists or what it contains, answer from this section; do not say \
a thing cannot be verified when its current contents are printed above.

Result of actually running the project's own test suite just now:
{test_result}

That test result is MEASURED, not claimed -- the suite was executed to produce it. Where it \
speaks to a criterion (do the tests run, do they pass), believe it over anything you infer \
from reading the diff. Do not assert that tests fail, or that the project does not build, when \
the measured result above says otherwise.

Result of running THIS TICKET'S OWN verifier command (the "VERIFIER COMMAND" its acceptance \
criteria name) just now, against the ticket's own tree:
{criterion_result}

That is also MEASURED. It is the suite the criteria were written against: where it passed, the \
criteria about its tests running and passing are met, and its assertions are the evidence for \
the behaviour the criteria describe.

Mechanical acceptance checks declared on the ticket, evaluated in code just now:
{mechanical_checks}

Those are computed, not judged. If every declared criterion is already covered above and all \
of them are OK, say pass without second-guessing them; a FAILED mechanics line is always a \
fail. Judge the remaining criteria yourself.

Weigh the diff above all else. It is what the ticket actually changed; the close reason and \
notes are the agent's own account of it and may be generous. If the diff is empty or does not \
plausibly implement the criteria, that is a fail no matter how confident the description sounds.

Decide pass or fail against the acceptance criteria specifically -- not whether the work is \
impressive, not whether you'd have done it differently, just whether the stated criteria are \
actually met based on the evidence given. If the evidence is too thin to tell either way, \
fail rather than assume -- an unverifiable claim of success is not the same as success.

Respond with strict JSON and nothing else: \
{{"verdict": "pass"|"fail", "reasoning": "<one or two sentences>"}}
"""


def _diff_for(issue: dict) -> str:
    """The diff this ticket produced.

    Prefers the commits the harness made for this ticket (found by their
    `<ticket-id>: ` subject), falling back to the recorded work_commit.
    Absent for a ticket that genuinely changed no files -- which is a
    meaningful signal to the verifier, so this returns empty rather than
    raising."""
    sha = (issue.get("metadata") or {}).get("work_commit")
    try:
        return workspaces.diff_for_ticket(
            workspaces.project_id_for(issue["id"]), issue["id"], sha
        )
    except Exception:
        return ""


def _tests_for(issue: dict) -> dict | None:
    """Run the project's own suite against the ticket's own tree.

    The ticket's tree, not the integration checkout: the merge into the
    integration branch is gated on this verdict, so the checkout does not
    yet contain the change under review and the suite would be measured
    without it."""
    try:
        return workspaces.run_tests_for_ticket(issue["id"])
    except Exception:
        log.exception("could not run tests for %s", issue.get("id"))
        return None


def _current_files_for(issue: dict) -> str:
    """Live contents of the project's criteria-bearing files, from the
    ticket's own tree, for the same reason _tests_for uses it."""
    try:
        return workspaces.criteria_file_snapshot_for_ticket(issue["id"])
    except Exception:
        log.exception("could not snapshot files for %s", issue.get("id"))
        return ""


def _is_merge_conflict(reason: str) -> bool:
    # "local changes would be overwritten" is the integration checkout holding
    # uncommitted strays, not a content collision -- but it is still a retry,
    # not a human's problem (merge_to_integration now clears the strays first,
    # so the retry usually just works).
    return (
        "CONFLICT" in reason
        or "Merge conflict" in reason
        or "would be overwritten" in reason
    )


@contextmanager
def _conn(conn):
    """Reuse a caller's connection, or open one from DATABASE_URL. The
    conflict requeue needs a connection for the graph-thread reset, and the
    worker's land call has none to hand."""
    if conn is not None:
        yield conn
    else:
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as opened:
            yield opened


def land(ticket_id: str, conn=None) -> bool:
    """Merge a ticket's approved work into the integration branch.

    The merge used to happen when the ticket was committed, which meant the
    integration branch carried work the verifier had not judged yet -- and
    when a verdict came back fail, work that had already been rejected.
    Landing it here instead, on a pass, is what makes "actioned from the
    state of the integration branch" mean something: what a ticket starts
    from is only ever work that passed review.

    A CONFLICT is ordinary contention, not a broken ticket: the integration
    tip moved under this ticket (another ticket landed first) and both
    changed the same lines. Parking that behind a human strands approved
    work; instead the ticket is requeued so the agent redoes it on the new
    tip. Any other merge failure (or a conflict that will not clear after
    the rework budget) is parked for a person, with the work left safely on
    its own branch."""
    ok, reason = workspaces.merge_to_integration(ticket_id)
    if ok:
        log.info("landed %s on the integration branch", ticket_id)
        return True

    if _is_merge_conflict(reason):
        issue = beads.show(ticket_id)
        attempt = int((issue.get("metadata") or {}).get("rework_count") or 0) + 1
        if attempt <= MAX_VERIFIER_REWORKS:
            try:
                with _conn(conn) as active:
                    requeue_for_rework(
                        active,
                        ticket_id,
                        "the work passed review but the integration branch moved "
                        f"under it (another ticket landed first) and the merge "
                        f"conflicted: {reason}",
                        attempt,
                    )
                log.warning(
                    "requeued %s after an integration merge conflict (attempt %s)",
                    ticket_id,
                    attempt,
                )
                return False
            except Exception:
                log.exception("could not requeue %s after a merge conflict", ticket_id)
        else:
            reason = (
                f"the merge kept conflicting after {MAX_VERIFIER_REWORKS} rework "
                f"attempt(s): {reason}"
            )

    log.error("could not land %s: %s", ticket_id, reason)
    beads.flag_for_human(
        ticket_id,
        f"the work was approved but could not be merged into the integration branch: "
        f"{reason}. It is committed on branch {workspaces.ticket_branch(ticket_id)}; "
        f"someone has to resolve that.",
    )
    return False


def accept_and_merge(ticket_id: str, response: str, conn=None) -> dict:
    """A person accepts a ticket's work: record it, close it, and merge its
    branch into the integration branch.

    A plain respond closes a ticket and records the decision, but nothing ever
    lands it -- landing only follows a verifier pass, and the verifier defers
    to a human decision. workspace-o0n.16.1 was accepted that way and its work
    sat on its ticket branch, unmerged, until a later ticket carried it in.

    A ticket that is already closed (accepted before this existed) is merged
    without being reopened. A merge that cannot happen is NOT requeued for an
    agent the way a verifier-passed conflict is -- a person decided this
    ticket, so the conflict goes back to a person: the ticket is flagged with
    the reason and the work stays on its branch.

    Returns {"merged": bool, "reason": str}."""
    project_id = workspaces.project_id_for(ticket_id)
    issue = beads.show(ticket_id)
    if issue.get("status") == "closed":
        beads.append_note(ticket_id, f"human response: {response}")
        beads.remove_human_flag(ticket_id)
        beads.set_metadata(ticket_id, beads.DECISION_KEY, "answered")
    else:
        beads.respond_to_human(ticket_id, response)

    repo = workspaces.path_for(project_id)
    branch = workspaces.ticket_branch(ticket_id)
    has_branch = os.path.isdir(repo) and workspaces._branch_exists(repo, branch)
    ref, source = _work_to_accept(ticket_id, project_id, branch if has_branch else None, conn)
    if ref is None:
        if not has_branch:
            reason = f"there is no branch {branch} and no judged commit to merge, so nothing landed"
        else:
            reason = (
                f"nothing to merge: {branch} holds no work beyond the integration branch "
                f"(a rework resets it there) and no judged commit with work is recorded"
            )
        beads.append_note(ticket_id, f"accepted; {reason}")
        return {"merged": False, "reason": reason}

    ok, reason = workspaces.merge_to_integration(ticket_id, ref)
    if not ok:
        log.error("accepted %s but could not merge it: %s", ticket_id, reason)
        beads.flag_for_human(
            ticket_id,
            f"accepted, but the merge into the integration branch failed: {reason}. "
            f"The work is at {source}; resolve the merge, then accept again.",
        )
        return {"merged": False, "reason": reason}
    beads.append_note(ticket_id, f"accepted and merged {source} into the integration branch")
    log.info("accepted %s and merged %s", ticket_id, source)
    return {"merged": True, "reason": ""}


def _work_to_accept(ticket_id: str, project_id: str, branch: str | None, conn) -> tuple:
    """(ref, description) of the work a person is accepting, or (None, "").

    The ticket's branch, when it holds work beyond the integration branch.
    Otherwise the commit that was JUDGED: picking a ticket up for rework puts
    its branch back on the integration tip (workspaces.reset_for_attempt), so
    accepting work the verifier REJECTED -- the very case accept exists for --
    finds the branch empty. Found live 2026-10-03: workspace-o0n.1.5 was
    accepted, "merged" an empty branch and reported merged=true, while its
    work sat on a commit only the verification record still named."""
    if branch and workspaces.commits_ahead(project_id, branch) > 0:
        return branch, f"branch {branch}"
    for recorded in _judged_commits(ticket_id, conn):
        head = recorded.split("..")[-1]
        if head and workspaces.commits_ahead(project_id, head) > 0:
            return head, f"the judged commit {head[:12]}"
    return None, ""


def _judged_commits(ticket_id: str, conn) -> list[str]:
    """The ticket's work commit as currently recorded, then as last judged."""
    out: list[str] = []
    current = (beads.show(ticket_id).get("metadata") or {}).get("work_commit")
    if current:
        out.append(current)
    try:
        with _conn(conn) as active:
            record = verifications.get_for_issue(active, ticket_id)
        if record and record.get("work_commit"):
            out.append(record["work_commit"])
    except Exception:
        log.exception("could not read the last verdict for %s", ticket_id)
    return out


def _describe_tests(tests: dict | None) -> str:
    if tests is None:
        return "(no runnable test script in this project)"
    if tests["exit"] == 0 and tests["ran"] == 0:
        return (
            "THE TEST COMMAND EXITED 0 BUT RAN ZERO TESTS. This is not a passing suite -- it is "
            "a command that verified nothing (e.g. a runner whose file glob matched no files). "
            f"Treat it as no test coverage at all.\n{tests['tail']}"
        )
    headline = ""
    if tests["exit"] == 0 and tests["ran"] > 0 and tests["failed"] == 0:
        # Said in words, not only as numbers: a fallback model failed
        # workspace-o0n.1.5 twice on claims about code that 10,644 passing
        # tests had just exercised, reading straight past exit=0 failed=0.
        headline = (
            f"THE SUITE PASSED: all {tests['ran']} tests ran and passed (exit 0). A claim that "
            "the code under test is broken has to be squared with this.\n"
        )
    return (
        headline
        + f"exit={tests['exit']}  tests_run={tests['ran']}  passed={tests['passed']}  "
        f"failed={tests['failed']}\n{tests['tail']}"
    )


# -- the ticket's own verifier command ----------------------------------
#
# Stories carry their own "VERIFIER COMMAND: <cmd>" -- the suite the criteria
# were written against. The verifier used to run only the project's whole
# suite and hand the model its headline, so a model could not see the one
# result the criteria name. Found live 2026-10-04: workspace-o0n.18.14 was
# failed twice by the fallback model for "the test suite result does not
# verify the specific acceptance criteria" while its named suite passed 47/0.

_VERIFIER_COMMAND = re.compile(r"VERIFIER COMMAND:\s*(.+)")
# An earlier verdict, as requeue_for_rework writes it into the notes.
_EARLIER_VERDICT = re.compile(r"^\s*verifier (fail|pass) #\d+:")


def verifier_command_in(criteria: str | None) -> str | None:
    """The command named on a criteria's `VERIFIER COMMAND:` line, or None."""
    m = _VERIFIER_COMMAND.search(criteria or "")
    if not m:
        return None
    return m.group(1).strip().strip("`").strip() or None


def _criterion_run_for(issue: dict, criteria: str | None) -> dict | None:
    """Run the criteria's verifier command in the ticket's own tree.

    ONLY the project's own test executable is run: the command must start
    with the same program as the project's declared test command, so a line
    of ticket text cannot run anything else on the harness. None when there
    is no command to run; a refused command is reported, not run."""
    command = verifier_command_in(criteria)
    if not command:
        return None
    declared = toolchain.test_command_for(workspaces.project_id_for(issue["id"]))
    try:
        program = shlex.split(command)[0]
        allowed = shlex.split(declared)[0] if declared else None
    except (ValueError, IndexError):
        return {"command": command, "refused": "the command cannot be parsed"}
    if program != allowed:
        return {
            "command": command,
            "refused": f"it does not run the project's test program ({allowed or 'none declared'})",
        }
    try:
        result = workspaces.run_command_for_ticket(issue["id"], command)
    except Exception:
        log.exception("could not run the verifier command for %s", issue.get("id"))
        return None
    result["command"] = command
    return result


def _criterion_failure(run: dict | None) -> str | None:
    """The reason the verifier command's run is a mechanical fail, or None."""
    if not run or run.get("refused"):
        return None
    if run["exit"] != 0 or run["failed"] > 0:
        return (
            f"The ticket's own verifier command failed when run just now ({run['command']}): "
            f"exit={run['exit']} tests_run={run['ran']} failed={run['failed']}. {run['tail'][-600:]}"
        )
    if run["ran"] == 0:
        return (
            f"The ticket's own verifier command ran zero tests ({run['command']}), so it "
            f"verified nothing. {run['tail'][-600:]}"
        )
    return None


def _describe_criterion_run(run: dict | None) -> str:
    if run is None:
        return "(the acceptance criteria name no verifier command)"
    if run.get("refused"):
        return f"NOT RUN: {run['command']} -- {run['refused']}."
    headline = ""
    if run["exit"] == 0 and run["ran"] > 0 and run["failed"] == 0:
        headline = f"THE TICKET'S SUITE PASSED: {run['ran']} tests ran and passed (exit 0).\n"
    return (
        headline + f"command: {run['command']}\nexit={run['exit']}  tests_run={run['ran']}  "
        f"passed={run['passed']}  failed={run['failed']}\n{run['tail']}"
    )


def verdict_json(content) -> dict:
    """The verdict object in a model's reply: the whole reply when it is bare
    JSON, else the first JSON object in it -- inside a ```json fence or after
    a line of prose.

    Found live 2026-10-05: workspace-o0n.19.1 was failed "verifier response
    unparseable" on a reply the local model had wrapped in a fence -- a parse
    failure, not a judgement, on the ticket's last rework attempt. Raises
    json.JSONDecodeError when there is no JSON object at all, which the caller
    fails closed on as before."""
    text = str(content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    raise json.JSONDecodeError("no JSON object in the reply", text, 0)


def _notes_for_model(notes: str | None) -> str:
    """The ticket's notes without earlier verifier verdicts.

    A weaker model repeats a verdict it is shown, however the prompt asks it
    to judge afresh: workspace-o0n.18.14's second fail quoted its first word
    for word ("...as noted in the verifier's accumulated notes"). The notes
    stay on the ticket for people and for the reworking agent."""
    kept = [line for line in (notes or "").splitlines() if not _EARLIER_VERDICT.match(line)]
    return "\n".join(kept).strip() or "(none)"


def _reset_thread(conn, issue_id: str) -> None:
    """Drop the ticket's LangGraph thread so a requeued attempt genuinely
    starts over.

    Without this, re-queuing is a no-op that lies: the graph for that
    ticket has already reached END, so worker.work_one_ticket's resume
    returns immediately, and (before completion_summary is cleared) the
    ticket would just be re-closed with its old claim. A fresh thread with
    the verifier's finding in the opening prompt is what makes the agent
    re-attempt. Best-effort: the checkpointer tables are absent in some
    tests, and a failed reset must not abort the verdict itself."""
    try:
        with conn.cursor() as cur:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (issue_id,))
    except psycopg.errors.UndefinedTable:
        # Test databases have no checkpointer tables until a worker test
        # creates them. Absent tables mean nothing to reset, not a failure.
        log.debug("no checkpointer tables to reset for %s", issue_id)
    except Exception:
        log.exception("could not reset the graph thread for %s", issue_id)


def requeue_for_rework(conn, issue_id: str, reasoning: str, attempt: int) -> None:
    """Send a verification-failed ticket back to an agent, carrying the
    verifier's actual finding.

    Clearing completion_summary is load-bearing: worker.work_one_ticket
    closes a ticket as soon as that key is set, so leaving it would let the
    old claim close the ticket again the moment capacity frees, with the
    verifier's finding never read. work_commit is cleared so the verifier's
    idempotency (which is per-commit) judges the next attempt's diff
    instead of skipping it."""
    beads.set_metadata(issue_id, "rework_count", str(attempt))
    beads.set_metadata(issue_id, "rework_reason", reasoning[:4000])
    # human_decision goes with them: requeueing means the decision that
    # stood is superseded, and the next attempt's work has to be judged on
    # its own merits rather than skipped as already-answered.
    for key in ("completion_summary", "work_commit", beads.DECISION_KEY):
        try:
            beads.unset_metadata(issue_id, key)
        except Exception:
            log.exception("could not clear %s on %s", key, issue_id)
    try:
        beads.remove_human_flag(issue_id)
    except Exception:
        log.exception("could not clear the human flag on %s", issue_id)
    beads.reopen(issue_id, f"verifier fail #{attempt}: {reasoning}")
    _reset_thread(conn, issue_id)


def awaiting_verdict(conn, issue: dict) -> bool:
    """Whether `verify_ticket` would judge this issue now: closed, carrying
    acceptance criteria, and with no verdict standing for the commit
    currently under judgement.

    Split out of verify_ticket so the scheduler can ask the same question
    WITHOUT building a session -- it asks once per cycle to detect a cycle
    with nothing to do (run_scheduler._nothing_to_do, 2026-09-16). A second
    copy of this rule would let the two drift, and the failure that causes
    is the bad direction: an idle-guard that believes nothing awaits a
    verdict silently stops verifying.

    The idempotency is per-commit, not per-issue: a verdict already recorded
    for the commit under judgement means the question is answered, but a
    ticket whose work_commit has changed since (i.e. it was requeued after a
    fail and produced new work) is judged again. A verdict recorded before
    work_commit existed (NULL) is trusted as-is: re-judging every
    already-verified legacy ticket would reopen settled work and re-block
    its dependents for no reason. The one exception is a ticket explicitly
    requeued for rework -- its old verdict must not stand in the way of
    judging the new work."""
    if beads.human_decision(issue):
        # A person has judged it. Re-weighing a decision is not judging
        # work, and doing it here reopened tickets that were already
        # accepted on the strength of a contaminated commit record -- see
        # beads.DECISION_KEY for the live case.
        return False
    if not beads.acceptance_criteria(issue) and not beads.acceptance_checks(issue):
        return False
    if issue.get("status") != "closed":
        return False
    current_commit = (issue.get("metadata") or {}).get("work_commit")
    existing = verifications.get_for_issue(conn, issue["id"])
    if not existing:
        return True
    if existing.get("work_commit") == current_commit:
        return False
    if existing.get("work_commit") is None and not (issue.get("metadata") or {}).get("rework_count"):
        return False
    return True


def verify_ticket(conn, issue_id: str, model) -> dict | None:
    """The recorded verdict dict, or None when the ticket isn't a candidate
    -- see awaiting_verdict for what counts as one.

    Structured `acceptance_checks` are evaluated in code FIRST. A failing
    check decides a fail with no model call at all, and a ticket whose
    criteria are entirely machine-checkable can pass without one; the model
    is asked only about the free-text criteria that genuinely need judgment.
    An unparseable model response fails closed to "fail", same posture as
    reviewer.py -- an unverifiable verdict is not a pass."""
    issue = beads.show(issue_id)
    if not awaiting_verdict(conn, issue):
        return None
    criteria = beads.acceptance_criteria(issue)
    checks = beads.acceptance_checks(issue)
    current_commit = (issue.get("metadata") or {}).get("work_commit")

    seat_id = beads.assigned_seat(issue) or issue.get("assignee") or "unknown"
    tests = _tests_for(issue)
    check_results = acceptance.evaluate(issue, tests) if checks else []
    failed_checks = acceptance.failed(check_results)
    criterion_run = _criterion_run_for(issue, criteria)
    criterion_failure = _criterion_failure(criterion_run)

    if failed_checks:
        # Deterministic: no model call can override a failed check.
        verdict = "fail"
        reasoning = (
            "Machine-checked acceptance criteria failed: "
            + "; ".join(r["detail"] for r in failed_checks)
        )
    elif criterion_failure:
        # The ticket's own verifier command, run just now, failed: no model
        # can read past that.
        verdict = "fail"
        reasoning = criterion_failure
    elif check_results and not criteria:
        # Every criterion the ticket stated is machine-checkable and passed.
        verdict = "pass"
        reasoning = (
            "Machine-checked acceptance criteria all passed: "
            + "; ".join(r["detail"] for r in check_results)
        )
    else:
        response = model.invoke(
            PROMPT.format(
                title=issue.get("title", ""),
                description=issue.get("description", ""),
                acceptance_criteria=criteria or "(none -- judged by the mechanical checks)",
                close_reason=issue.get("close_reason") or "(none recorded)",
                notes=_notes_for_model(issue.get("notes")),
                diff=_diff_for(issue) or "(no code change recorded for this ticket)",
                current_files=(
                    _current_files_for(issue) or "(no configuration or readme files found)"
                ),
                test_result=_describe_tests(tests),
                criterion_result=_describe_criterion_run(criterion_run),
                mechanical_checks=acceptance.describe(check_results),
            )
        )
        content = getattr(response, "content", response)

        try:
            data = verdict_json(content)
            verdict = data["verdict"]
            reasoning = data.get("reasoning", "")
            if verdict not in ("pass", "fail"):
                raise ValueError(f"unexpected verdict: {verdict!r}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            verdict = "fail"
            reasoning = f"verifier response unparseable: {e}"

        # Mechanical override, applied only to the one case that admits no
        # judgement: the suite exited 0 while running nothing. A model
        # reading a well-written test file has no way to see that the file is
        # never executed, so this is checked rather than asked.
        if verdict == "pass" and tests and tests["exit"] == 0 and tests["ran"] == 0:
            verdict = "fail"
            reasoning = (
                "Mechanical check overrode a pass: the project's test command exited 0 but ran "
                "zero tests, so it is not evidence of anything. " + reasoning
            )

    verifications.record(conn, issue_id, seat_id, verdict, reasoning, work_commit=current_commit)

    # A pass is what lands the work. Until this runs the ticket's commit is
    # only on its own branch, so the integration branch never holds work
    # that has not been judged -- which is what lets the next ticket start
    # from a tree of accepted work only.
    if verdict == "pass":
        try:
            land(issue_id, conn=conn)
        except Exception:
            log.exception("approved %s but could not land it", issue_id)

    # A fail must be actioned, not just recorded. The ticket goes back to
    # an agent with the verifier's finding attached (requeue_for_rework),
    # and only after MAX_VERIFIER_REWORKS attempts is it parked for a
    # human. This replaced the old split -- reopen when the test suite was
    # objectively broken, otherwise only flag -- which left judgement-call
    # fails sitting unactioned on the board forever. The cost of reopening
    # is real (it re-blocks dependents; a false fail once froze 73 of them
    # for sixteen hours, see the git history), which is exactly why the
    # loop is bounded rather than unbounded.
    if verdict == "fail":
        attempt = int((issue.get("metadata") or {}).get("rework_count") or 0) + 1
        try:
            if attempt <= MAX_VERIFIER_REWORKS:
                log.info("requeuing %s for rework attempt %s: %s", issue_id, attempt, reasoning[:200])
                requeue_for_rework(conn, issue_id, reasoning, attempt)
            else:
                beads.flag_for_human(
                    issue_id,
                    f"verification failed after {MAX_VERIFIER_REWORKS} rework attempt(s): {reasoning}",
                )
        except Exception:
            log.exception("could not act on failed verification for %s", issue_id)

    return {"issue_id": issue_id, "seat_id": seat_id, "verdict": verdict, "reasoning": reasoning}
