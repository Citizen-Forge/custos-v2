"""
Tool-call permission layer. Three deliberately separate concerns:

1. `is_statically_safe` -- a fast-path allow-list so the ordinary
   build/test/VCS commands every project needs (npm, node, tsc, the test
   runner, local git, workspace-scoped file shuffling) skip the LLM
   classifier (classifier.py) entirely. The principle (user's call,
   2026-09-05): as long as a command's effects stay inside the project
   workspace, an agent should just be able to run it -- it was being
   starved by a classifier that reflexively denied `shell_exec` as
   "potentially executes arbitrary shell commands", which is every shell
   command. This is a *speed and reliability* path, NOT a sandbox:
   shell_exec still runs an un-jailed process (cwd=workspace_root, but
   nothing stops `cat /etc/shadow`). So the list is curated to
   workspace-scoped dev tooling, obvious escapes are rejected (absolute
   paths outside the workspace, `..`, `~`, command/process substitution,
   redirects out of tree, network tools, privilege escalation), and
   anything not matched falls through to the classifier -- which stays as
   the backstop for the long tail. It fundamentally cannot stop malicious
   code run *through* an allowed runtime (`npm test` runs whatever
   package.json says); that is a supply-chain / real-sandbox problem, not
   one a verb list solves.

2. `check_within_workspace` -- a hard invariant enforced *inside* the file
   tools themselves (tools.py), independent of whatever the static
   allow-list or classifier decided. This one is not classifier-
   overridable on purpose: workspace containment is a sandbox boundary,
   not a task-semantic judgment call, so it doesn't belong to the same
   layer that's reasoning about intent.

3. `is_infrastructure` -- names injected into every project workspace by
   the harness or its tooling (.git, .beads, .claude, .codex, .agents),
   not the project's own content. Unlike (2) this is NOT a security
   boundary and never raises: an agent poking at one of these is
   ordinary orientation behavior, not an escape attempt, so callers
   return an explanatory string the agent can read and move on from --
   same "ordinary, not a security event" treatment read_file already
   gives a missing file. The actual security boundary for a sensitive
   file like .claude/settings.json is still the classifier (or workspace
   containment for an escape); this only keeps an agent from wasting a
   turn finding that out the hard way. Found live 2026-09-04: an agent
   listed its workspace, tried to read .claude/settings.json because it
   was sitting right there in the listing, got denied, and its very next
   turn came back completely empty -- no text, no tool call, no further
   progress for hours.
"""

import os
import shlex


class PermissionDenied(Exception):
    pass


HIDDEN_FROM_LISTING = {".git", ".beads", ".claude", ".codex", ".agents"}


# Read-only infrastructure an agent may inspect even though it lives outside
# its workspace: installed toolchains. The default is the Godot 4 data dir the
# image installs -- export templates live there, and an agent building or
# exporting a Godot project legitimately needs to read them. Found in the
# 2026-09-19 denial audit: `ls /root/.local/share/godot/export_templates` was
# denied as an out-of-workspace read. os.pathsep-separated, overridable.
READONLY_ROOTS = tuple(
    p for p in os.environ.get("TOOLCHAIN_READ_PATHS", "/root/.local/share/godot").split(os.pathsep) if p
)

# Verbs whose path arguments may also reach a READONLY_ROOT. Everything else
# stays workspace-scoped. `find` is screened for its mutating flags below, and
# redirects are checked separately (and stay workspace-only), so this cannot
# be used to write into a read-only root.
_READONLY_VERBS = {
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ag",
    "find", "fd", "wc", "which", "type", "file", "stat", "tree", "du",
    "realpath", "readlink", "dirname", "basename", "strings", "less", "more",
    "xxd", "od", "md5sum", "sha1sum", "sha256sum", "cmp", "diff", "jq", "yq",
}
_FIND_MUTATORS = {
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprint", "-fprint0", "-fprintf", "-fls",
}


# -- shell command allow-list ------------------------------------------------

# Leading verbs an agent is expected to run to build and test a project.
# basename() is applied first, so `./node_modules/.bin/tsc` and
# `/usr/bin/node` reduce to `tsc` / `node`.
_ALLOWED_VERBS = {
    # package managers / task runners
    "npm", "npx", "pnpm", "pnpx", "yarn", "bun", "bunx", "corepack",
    # JS/TS runtimes and build tools
    "node", "nodejs", "deno", "tsx", "ts-node", "tsc", "vite", "esbuild",
    "rollup", "webpack", "swc", "parcel", "turbo", "nx", "gulp", "grunt",
    # other language toolchains a project might legitimately use
    "python", "python3", "pip", "pip3", "pytest", "ruff", "mypy", "poetry",
    "go", "cargo", "rustc", "make", "cmake", "gradle", "mvn",
    # test runners / linters / formatters
    "jest", "vitest", "mocha", "ava", "tap", "playwright", "cypress",
    "eslint", "prettier", "biome", "standard", "tsd", "c8", "nyc",
    # read-only inspection / text processing
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ag",
    "find", "fd", "wc", "pwd", "which", "type", "file", "stat", "tree",
    "du", "df", "env", "printenv", "date", "whoami", "id", "uname",
    "hostname", "realpath", "readlink", "dirname", "basename", "seq",
    "true", "false", "test", "[", "echo", "printf", "yes", "sleep", "jq",
    "yq", "column", "base64", "md5sum", "sha1sum", "sha256sum", "cksum",
    "cmp", "diff", "sort", "uniq", "cut", "tr", "nl", "tac", "rev",
    "fold", "expand", "comm", "join", "paste", "xxd", "od", "strings",
    "less", "more",
    # workspace-scoped mutation (path args are escape-checked below)
    "mkdir", "rmdir", "touch", "cp", "mv", "rm", "ln", "chmod", "sed",
    "awk", "tee", "patch", "install", "mktemp", "unzip", "tar",
    # harmless shell builtins
    "cd", "export", "set", "unset", "read", "shift", "xargs", "time",
    # the beads issue CLI: bd prime tells every agent to run `bd ready`,
    # `bd show`, `bd close` ... it operates on the local .beads DB. Only
    # `bd dolt` reaches a remote, and that's screened below like git.
    "bd",
}

# Never fast-pathed. Privilege escalation, disk/process/system control,
# and anything that reaches the network -- none of it is "build and test
# this project", and none of it is workspace-scoped.
_DENIED_VERBS = {
    "sudo", "su", "doas", "pkexec", "chown", "chgrp", "chroot", "nsenter",
    "mount", "umount", "dd", "mkfs", "fdisk", "parted", "shutdown",
    "reboot", "halt", "poweroff", "systemctl", "service", "initctl",
    "kill", "killall", "pkill", "crontab", "at", "batch",
    "ssh", "scp", "sftp", "rsync", "nc", "ncat", "netcat", "telnet",
    "curl", "wget", "ftp", "aria2c", "socat",
    "apt", "apt-get", "aptitude", "dpkg", "yum", "dnf", "rpm", "apk",
    "brew", "pacman", "snap", "flatpak",
    "docker", "podman", "nerdctl", "kubectl", "helm", "vagrant",
    "setcap", "setfacl", "iptables", "nft", "ifconfig",
    "eval", "exec", "source", "trap",
    "bash", "sh", "zsh", "fish", "dash", "ksh",
}

# git is fine for local history work; anything that talks to (or
# reconfigures) a remote is not workspace-scoped.
_GIT_REMOTE_SUBCOMMANDS = {
    "push", "fetch", "pull", "clone", "remote", "submodule", "archive",
    "bundle", "request-pull", "send-email", "svn", "p4", "daemon",
    "credential", "http-fetch", "http-push", "imap-send",
}

# Commands that break the harness's own bookkeeping if an agent runs them,
# regardless of what the classifier thinks. Unlike the allow-list filters
# above (which only decide whether to SKIP the classifier), these are hard
# denials in the same class as check_within_workspace: an integrity
# boundary, not a task-semantic judgment, so they are not the classifier's
# call to override.
#
# Found live 2026-09-17: an agent closed its own ticket and ran
# `git update-ref refs/heads/master HEAD` from its worktree, putting
# unverified work on the integration branch and leaving the integration
# checkout's working tree stale (HEAD had the files, the index did not).
# The classifier allowed both -- neither is a remote, network or privilege
# operation, and the completion gate assumes the agent reports completion
# through complete_ticket rather than running the raw mutation. A verb
# blacklist is bypassable (sh -c, python -c), which is why
# workspaces.revert_unexpected_integration_move exists as the mechanical
# backstop; this closes the direct path.
_FORBIDDEN_GIT_SUBCOMMANDS = {
    # move/create/delete refs directly -- the integration branch is the
    # harness's to advance (verifier.land), never an agent's
    "update-ref", "symbolic-ref", "pack-refs",
    # change HEAD, the index or the working tree: the harness owns the trees
    "reset", "checkout", "switch", "rebase", "cherry-pick", "revert",
    "stash", "worktree", "filter-branch", "replace", "gc", "prune",
}

# A few of the subcommands above have genuinely read-only forms that touch
# neither refs nor the working tree, and denying those made a harmless
# inspection look like an integrity-boundary violation. Found live
# 2026-09-18: an agent ran `git worktree list` to understand the layout and
# the gate refused it, because the whole `worktree` verb is blacklisted.
# Only the verbs named here are let through; every mutation form
# (`worktree add/remove/prune/move`, `stash push/pop/...`) stays forbidden.
_READONLY_GIT_FORMS = {
    "worktree": {"list"},
    "stash": {"list", "show"},
}

# `git branch`/`git tag` are read-only without these; with them they move
# or delete refs.
_FORBIDDEN_GIT_FLAGS = {
    "branch": {"-f", "-D", "-d", "-m", "-M", "--force", "--delete", "--move"},
    "tag": {"-f", "-d", "--force", "--delete"},
}

# git global options that consume the next token, so the subcommand is the
# token after their value (e.g. `git -C /repo update-ref ...`).
_GIT_VALUE_FLAGS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}

# bd subcommands that mutate the board. An agent reports completion through
# the harness's own tools (complete_ticket/refuse_ticket/decline_ticket),
# which record it correctly; the raw mutation bypasses the completion gate
# and the verifier.
_FORBIDDEN_BD_SUBCOMMANDS = {
    "close", "delete", "reopen", "update", "assign", "set-state", "label",
    "tag", "dep", "link", "duplicate", "supersede", "promote", "undo",
    "restore", "compact", "dolt", "sync", "init", "bootstrap", "hooks",
    "import", "gate", "merge-slot", "swarm", "federation", "branch", "vc",
}

# Inline-code flags for interpreters: `node -e "..."`, `python -c "..."`.
# The verb is allowed (running the project's code IS the job), but an
# ad-hoc snippet that bypasses the project's own scripts is exactly the
# kind of thing the classifier should still get to look at.
_EVAL_FLAGS = {"-e", "--eval", "-c", "--command", "-p", "--print"}
_INTERPRETER_VERBS = {
    "node", "nodejs", "deno", "bun", "python", "python3", "ruby", "php", "perl",
}

_DANGEROUS_ENV_PREFIXES = ("LD_", "DYLD_")
_DANGEROUS_ENV = {"PATH", "IFS", "BASH_ENV", "ENV", "SHELLOPTS", "PS4", "PROMPT_COMMAND"}

_SUBSTITUTION_MARKERS = ("$(", "`", "<(", ">(")
_SEGMENT_SEPARATORS = {"&&", "||", ";", ";;", "|", "|&", "&", "\n"}
_REDIRECT_OPS = {">", ">>", "<", "&>", "&>>", ">&", "<>"}
_GROUPING = {"(", ")", "{", "}"}
# Redirect targets that are fine even though they're absolute / outside
# the tree: the standard devices and any /dev/fd/N.
_OK_REDIRECT_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/stdin", "/dev/tty"}


def _is_within_workspace(path: str, workspace_root: str) -> bool:
    """Containment by path components, NOT by string prefix.

    A plain `startswith` was wrong in a way that only bites once
    workspaces have siblings: from /projects/proj-a the path
    ../proj-abc/x resolves to /projects/proj-abc/x, which starts with
    /projects/proj-a and was therefore allowed. With one workspace that
    was a latent bug (/workspace-evil escaped /workspace); with a
    workspace per project it would have meant no isolation between
    projects at all, which is the whole point of having them."""
    root = os.path.abspath(workspace_root)
    resolved = os.path.abspath(os.path.join(root, path))
    if resolved == root:
        return True
    return resolved.startswith(root + os.sep)


def _looks_like_path(token: str) -> bool:
    return "/" in token or token in (".", "..") or token.startswith("~")


def _within_any(path: str, roots) -> bool:
    resolved = os.path.abspath(path)
    for root in roots:
        root = os.path.abspath(root)
        if resolved == root or resolved.startswith(root + os.sep):
            return True
    return False


def _readable_path(path: str, workspace_root: str | None) -> bool:
    """Within the workspace, or within a declared read-only toolchain root."""
    if _within_any(path, READONLY_ROOTS):
        return True
    return workspace_root is not None and _is_within_workspace(path, workspace_root)


def _arg_stays_within(token: str, within) -> bool:
    """A single command argument stays inside whatever `within` permits.

    Non-path arguments (`express`, `-rf`, `s/a/b/`, a commit message) pass
    trivially -- only tokens that actually look like a filesystem path get
    resolved and checked."""
    if token.startswith("-"):
        return True
    # strip a `--flag=value` wrapper and check the value
    if token.startswith("--") and "=" in token:
        token = token.split("=", 1)[1]
    if not token or not _looks_like_path(token):
        return True
    if token.startswith("~"):
        return False
    if ".." in token.split("/") or token.startswith("/"):
        return within(token)
    return True


def _arg_stays_in_workspace(token: str, workspace_root: str) -> bool:
    return _arg_stays_within(token, lambda p: _is_within_workspace(p, workspace_root))


def _arg_readable(token: str, workspace_root: str) -> bool:
    return _arg_stays_within(token, lambda p: _readable_path(p, workspace_root))


def _is_env_assignment(token: str) -> bool:
    name, sep, _ = token.partition("=")
    return bool(sep) and name.isidentifier()


def _segment_ok(seg: list[str], workspace_root: str | None) -> bool:
    i = 0
    while i < len(seg) and _is_env_assignment(seg[i]):
        name = seg[i].split("=", 1)[0]
        if name in _DANGEROUS_ENV or name.startswith(_DANGEROUS_ENV_PREFIXES):
            return False
        i += 1
    if i >= len(seg):
        return True  # nothing but assignments
    verb = os.path.basename(seg[i])
    rest = seg[i + 1:]

    if verb in _DENIED_VERBS or verb == "":
        return False

    if verb == "git":
        sub = next((a for a in rest if not a.startswith("-")), None)
        if sub in _GIT_REMOTE_SUBCOMMANDS:
            return False
    elif verb == "bd":
        sub = next((a for a in rest if not a.startswith("-")), None)
        if sub == "dolt":  # `bd dolt push/pull/...` -- remote sync
            return False
    elif verb not in _ALLOWED_VERBS:
        return False

    if verb in _INTERPRETER_VERBS and any(a in _EVAL_FLAGS for a in rest):
        return False

    if workspace_root is not None:
        # Read-only verbs may also reach a declared toolchain root; everything
        # else (and any `find` that mutates) stays workspace-only.
        readonly = verb in _READONLY_VERBS and not (
            verb == "find" and any(a in _FIND_MUTATORS for a in rest)
        )
        checker = _arg_readable if readonly else _arg_stays_in_workspace
        if not all(checker(a, workspace_root) for a in rest):
            return False
    return True


def _tokenize(command: str) -> list[str] | None:
    """Operator-aware tokenization. `shlex.split` is only a *word*
    splitter -- it leaves `pwd;` as one token -- so use the lexer in
    punctuation-chars mode, which emits `;` `&&` `|` `>` `(` ... as their
    own tokens while still respecting quotes. Returns None if the command
    can't be lexed (unbalanced quotes)."""
    lex = shlex.shlex(command, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        return list(lex)
    except ValueError:
        return None


def _ok_redirect_target(target: str, workspace_root: str | None) -> bool:
    if target.isdigit():
        return True  # `2>&1` style fd dup
    if target in _OK_REDIRECT_TARGETS or target.startswith("/dev/fd/"):
        return True
    if workspace_root is None:
        return True
    return _arg_stays_in_workspace(target, workspace_root)


def _shell_is_allowlisted(command: str, workspace_root: str | None) -> bool:
    command = command.strip()
    if not command:
        return False
    if any(m in command for m in _SUBSTITUTION_MARKERS):
        return False  # command / process substitution -- let the classifier look

    tokens = _tokenize(command)
    if not tokens:
        return False

    seg: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _GROUPING:
            return False  # subshell / brace group -- too much for the fast path
        if tok in _SEGMENT_SEPARATORS:
            if not _segment_ok(seg, workspace_root):
                return False
            seg = []
        elif tok in _REDIRECT_OPS or (tok.rstrip("0123456789") in _REDIRECT_OPS and any(ch.isdigit() for ch in tok)):
            # `>`, `2>`, `>>`, `&>` ...  -- drop the fd number that lexed
            # onto the previous token, and workspace-check the target.
            if seg and seg[-1].isdigit():
                seg.pop()
            target = tokens[i + 1] if i + 1 < len(tokens) else None
            if target is None or not _ok_redirect_target(target, workspace_root):
                return False
            i += 2
            continue
        else:
            seg.append(tok)
        i += 1
    return _segment_ok(seg, workspace_root)


def _git_after_subcommand(rest: list[str]) -> tuple[str | None, list[str]]:
    """The git subcommand and the tokens after it, skipping global options
    and their values (so `git -C /repo worktree list` resolves the same
    way whether you want the verb or its arguments)."""
    i = 0
    while i < len(rest):
        a = rest[i]
        if a in _GIT_VALUE_FLAGS:
            i += 2
            continue
        if a.startswith("--") and "=" in a:
            i += 1
            continue
        if a.startswith("-"):
            i += 1
            continue
        return a, rest[i + 1:]
    return None, []


def _git_subcommand(rest: list[str]) -> str | None:
    """The git subcommand, skipping global options and their values."""
    return _git_after_subcommand(rest)[0]


def _segment_forbidden(seg: list[str]) -> str | None:
    i = 0
    while i < len(seg) and _is_env_assignment(seg[i]):
        i += 1
    if i >= len(seg):
        return None
    verb = os.path.basename(seg[i])
    rest = seg[i + 1:]

    if verb == "git":
        sub, after = _git_after_subcommand(rest)
        if sub in _FORBIDDEN_GIT_SUBCOMMANDS:
            readonly = _READONLY_GIT_FORMS.get(sub or "", set())
            action = next((a for a in after if not a.startswith("-")), None)
            if action not in readonly:
                return (
                    f"`git {sub}` edits the repository's own refs or working tree, which the "
                    "harness owns; ticket work is landed by the verifier-gated merge. Use the "
                    "file tools and report completion with complete_ticket."
                )
        flags = _FORBIDDEN_GIT_FLAGS.get(sub or "", set())
        if flags and any(a in flags for a in rest):
            return f"`git {sub}` with a force/delete/move flag rewrites refs, which is not allowed."
    elif verb == "bd":
        sub = next((a for a in rest if not a.startswith("-")), None)
        if sub in _FORBIDDEN_BD_SUBCOMMANDS:
            return (
                "`bd` board mutations are not run directly by agents; report completion with "
                "complete_ticket/refuse_ticket/decline_ticket and let the harness record it."
            )
    return None


def forbidden_reason(tool_name: str, tool_args: dict) -> str | None:
    """A hard, non-classifier-overridable reason this call must not run, or None.

    Applied before `is_statically_safe`/the classifier in the graph's
    permission_gate. Deliberately narrow: only the commands that would move
    the integration branch or mutate the board outside the harness's own
    pathways (see _FORBIDDEN_GIT_SUBCOMMANDS / _FORBIDDEN_BD_SUBCOMMANDS).
    Compound commands are split, so `cd x && git update-ref ...` is caught
    too."""
    if tool_name != "shell_exec":
        return None
    try:
        command = tool_args.get("command", "")
    except Exception:
        return None
    tokens = _tokenize(command)
    if not tokens:
        return None
    segments: list[list[str]] = [[]]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _SEGMENT_SEPARATORS or tok in _GROUPING:
            segments.append([])
        elif tok in _REDIRECT_OPS or (
            tok.rstrip("0123456789") in _REDIRECT_OPS and any(c.isdigit() for c in tok)
        ):
            if segments[-1] and segments[-1][-1].isdigit():
                segments[-1].pop()
            i += 2
            continue
        else:
            segments[-1].append(tok)
        i += 1
    for seg in segments:
        reason = _segment_forbidden(seg)
        if reason:
            return reason
    return None


# Harness-control tools. They touch neither the shell, the filesystem nor the
# network -- they record a verdict, note or subtask on the board, and they are
# how an agent finishes a ticket. A safety classifier has no business gating
# them, and a denial (or an unparseable response, which fails closed) turns
# straight into a loop: the agent literally cannot report completion. Found in
# the 2026-09-19 denial audit -- a `refuse_ticket` had been denied.
_CONTROL_TOOLS = {
    "remember_fact", "create_subtask", "complete_subtask", "complete_ticket",
    "refuse_ticket", "decline_ticket", "write_handoff_note",
}


def is_statically_safe(tool_name: str, tool_args: dict, workspace_root: str | None) -> bool:
    """Fast-path allow: True means this call may skip the classifier.

    Fails safe -- any surprise returns False and the call is classified
    as normal."""
    try:
        if tool_name in _CONTROL_TOOLS:
            return True  # board/control only; no shell, filesystem or network
        if tool_name in ("read_file", "list_directory"):
            # the tools' own check_within_workspace / is_infrastructure
            # guards still run regardless; this just skips the LLM call.
            if workspace_root is None:
                return False
            return _readable_path(tool_args.get("path", ""), workspace_root)
        if tool_name == "shell_exec":
            return _shell_is_allowlisted(tool_args.get("command", ""), workspace_root)
    except Exception:
        return False
    return False


def check_within_workspace(path: str, workspace_root: str) -> None:
    if not _is_within_workspace(path, workspace_root):
        raise PermissionDenied(f"path escapes workspace: {path!r}")


def check_readable(path: str, workspace_root: str) -> None:
    """Containment for READ tools: the workspace, or a declared read-only
    toolchain root. Writes keep using check_within_workspace."""
    if not _readable_path(path, workspace_root):
        raise PermissionDenied(f"path escapes workspace: {path!r}")


def is_infrastructure(path: str) -> bool:
    """True if any component of `path` names harness/tool infrastructure
    (see module docstring) rather than the project's own content. Checks
    every component, not just the first, so a nested reference like
    `src/../.claude/settings.json` or a deeper `sub/.beads/x` still
    matches."""
    parts = path.replace("\\", "/").strip("/").split("/")
    return any(p in HIDDEN_FROM_LISTING for p in parts)
