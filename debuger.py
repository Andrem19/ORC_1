"""
Autonomous supervisor for the ORC_1 orchestrator.

Runs `conda run -n env6 python main.py`, tails logs/orchestrator.log,
detects the first fallback on a slice, kills the orchestrator, invokes
`claude --dangerously-skip-permissions --model opus` as a diagnostic
engineer with a strict VERDICT contract, prints the Russian report,
then either whitelists the slice (verdict 1) or just restarts after
Claude's in-place repair (verdict 2/3). On a clean run without any
fallback it bumps start_from = "vN" -> "vN+1" in config.toml.

Launch: conda run -n env6 python debuger.py
Stop:   Ctrl+C (supervisor stops the orchestrator first, then exits).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_SYMLINK = LOGS_DIR / "orchestrator.log"
CONFIG_PATH = PROJECT_ROOT / "config.toml"
STATE_DIR = PROJECT_ROOT / "state"
ALLOWED_FILE = STATE_DIR / "debugger_allowed_fallback.json"

CONDA_BIN = "/home/jupiter/miniconda3/bin/conda"
CONDA_ENV = "env6"

FALLBACK_PAT = re.compile(
    r"orchestrator\.direct\.fallback - Fallback #\d+ \(.+?\) also failed for slice (\S+?):"
)
FALLBACK_ATTEMPT_PAT = re.compile(
    r"orchestrator\.direct\.fallback - Attempting fallback #\d+ with provider .+? for slice (\S+)"
)
VERDICT_PAT = re.compile(r"^\s*VERDICT\s*[:=]\s*([123])\b", re.MULTILINE)
START_FROM_PAT = re.compile(r'^(\s*start_from\s*=\s*")v(\d+)(")', re.MULTILINE)

MAX_PLAN_VERSION = 12

KILL_WAIT = 5.0
POLL_SLEEP = 0.5
RESTART_COOLDOWN = 3.0

# ANSI: show cursor + reset attributes — orchestrator's rich console may
# have hidden the cursor / set styles; once we SIGKILL it, those escape
# sequences never got their closing partner, so we restore the terminal.
_TERM_RESET = "\x1b[?25h\x1b[0m\n"


def log(msg: str) -> None:
    # Write supervisor status messages to stderr so they do not interleave
    # with the orchestrator's rich live dashboard on stdout.
    sys.stderr.write(f"[debuger] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# State: slices known to be beyond minimax's capability (verdict 1 whitelist)
# ---------------------------------------------------------------------------
class AllowedFallbackStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._slices: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log(f"warning: could not parse {self._path} ({exc}); starting empty")
            return
        entries = data.get("allowed_fallback_slices", [])
        if isinstance(entries, list):
            self._slices = {str(x) for x in entries if isinstance(x, str)}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"allowed_fallback_slices": sorted(self._slices)}
        self._path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def contains(self, slice_id: str) -> bool:
        return slice_id in self._slices

    def add(self, slice_id: str) -> None:
        if slice_id in self._slices:
            return
        self._slices.add(slice_id)
        self._save()

    def list(self) -> list[str]:
        return sorted(self._slices)


# ---------------------------------------------------------------------------
# Log tailer — follows the logs/orchestrator.log symlink across rotations
# ---------------------------------------------------------------------------
class LogTailer:
    def __init__(self, symlink_path: Path) -> None:
        self._symlink_path = symlink_path
        self._fh = None
        self._resolved: Path | None = None
        self._buf = ""

    def _resolve(self) -> Path | None:
        try:
            if not self._symlink_path.exists():
                return None
            return self._symlink_path.resolve(strict=True)
        except (OSError, RuntimeError):
            return None

    def seek_to_current_eof(self) -> None:
        """Open the current target and seek to EOF — skip any stale lines."""
        target = self._resolve()
        if target is None:
            return
        try:
            fh = open(target, "r", encoding="utf-8", errors="replace")
            fh.seek(0, os.SEEK_END)
        except OSError as exc:
            log(f"warning: cannot open {target}: {exc}")
            return
        self._close()
        self._fh = fh
        self._resolved = target
        self._buf = ""

    def _close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None

    def poll(self) -> Iterator[str]:
        """Yield newly written, complete lines (silently — do NOT write to stdout).

        The orchestrator owns stdout and renders its own rich console
        dashboard there. We only read the log file to detect patterns
        (fallback) — duplicating lines to stdout would corrupt rich's
        live-refresh cursor control and cause flicker.
        """
        target = self._resolve()
        if target is None:
            return
        if target != self._resolved:
            # symlink rotated → new run file, start from beginning
            try:
                fh = open(target, "r", encoding="utf-8", errors="replace")
            except OSError as exc:
                log(f"warning: cannot open {target}: {exc}")
                return
            self._close()
            self._fh = fh
            self._resolved = target
            self._buf = ""
        if self._fh is None:
            return
        try:
            chunk = self._fh.read()
        except OSError:
            return
        if not chunk:
            return
        self._buf += chunk
        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1 :]
            yield line

    def close(self) -> None:
        self._close()
        self._resolved = None
        self._buf = ""


# ---------------------------------------------------------------------------
# Orchestrator process control
# ---------------------------------------------------------------------------
def start_orchestrator() -> subprocess.Popen:
    cmd = [
        CONDA_BIN,
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "-u",
        "main.py",
    ]
    log(f"starting orchestrator: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # own process group for clean group-kill
    )
    log(f"orchestrator pid={proc.pid} pgid={os.getpgid(proc.pid)}")
    return proc


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def kill_orchestrator(proc: subprocess.Popen) -> None:
    """Kill the orchestrator and its whole process group immediately.

    We do NOT send SIGINT/SIGTERM first — those trigger orchestrator's
    drain mode and let it finish the current slice, which is exactly the
    behaviour we want to avoid on fallback detection. SIGKILL on the
    process group takes down orchestrator + claude/qwen/lmstudio workers
    in one shot.
    """
    if proc.poll() is not None:
        _restore_terminal()
        return
    log("killing orchestrator (SIGKILL process group — no drain)")
    _signal_group(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=KILL_WAIT)
    except subprocess.TimeoutExpired:
        log(f"warning: orchestrator pg did not die within {KILL_WAIT}s after SIGKILL")
        _restore_terminal()
        return
    log(f"orchestrator exited rc={proc.returncode}")
    _restore_terminal()


def _restore_terminal() -> None:
    """Restore cursor/styles the orchestrator's rich console may have left set."""
    try:
        sys.stdout.write(_TERM_RESET)
        sys.stdout.flush()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Claude investigator
# ---------------------------------------------------------------------------
CLAUDE_PROMPT_TEMPLATE = """\
You are invoked as a diagnostic engineer for the ORC_1 orchestrator.

Working directory: /home/jupiter/PYTHON/ORC_1
MCP codebase:     /home/jupiter/PYTHON/FOUNDATION
Failing slice:    {slice_id}
Already-whitelisted-as-beyond-minimax slices: {allowed_slices_list}

Primary evidence:
  - logs/orchestrator.log (symlink to the latest run log)
  - logs/runs/<latest>/orchestrator.events.jsonl
  - state/, artifacts/, compiled_plans/ for the last run
  - git log in both repos for recent architectural changes

Task — in this exact order:

1. Read the orchestrator log around the failure for slice {slice_id}.
   Read its fallback attempt details, the verifier output, worker
   transcript, and any incidents. Establish the concrete cause.

2. Classify the cause. VERDICT PRIORITY IS STRICT: you must try
   VERDICT 3 first, then VERDICT 2, and only fall back to VERDICT 1
   if 3 and 2 are genuinely impossible without weakening any
   acceptance criterion. VERDICT 1 is the emergency exit — do not
   pick it just because investigation is hard. Before choosing it you
   must be able to show concretely why no universal, smart,
   criteria-preserving improvement to the orchestrator OR the MCP
   would make minimax succeed on this slice.

   VERDICT 3 (PREFERRED) — root cause is in the orchestrator
     architecture (/home/jupiter/PYTHON/ORC_1) and can be fixed so
     minimax succeeds on this slice WITHOUT weakening any acceptance
     criterion. Prefer universal, generalizable improvements that
     make the algorithm smarter / more robust for many slices, not
     just this one. Do the change. Follow the repository's CLAUDE.md
     (env6, 800-line files, unit + integration tests required). This
     is the strongly preferred outcome.

   VERDICT 2 (PREFERRED OVER 1) — root cause is a bug / architectural
     hole in our MCP (/home/jupiter/PYTHON/FOUNDATION). Fix it there.
     RULES:
       (a) do not add new MCP tools,
       (b) do not remove or reduce functionality of existing tools,
       (c) preserve full backward compatibility.
     After patching run `conda run -n env6 pytest -n 12` in the
     FOUNDATION repo; only if ALL tests pass run
     `git add -A && git commit -m "<msg>" && git push` there.

   VERDICT 1 (EMERGENCY ONLY — LAST RESORT) — choose this only after
     you have seriously considered and ruled out both 3 and 2.
     Justify rigorously in the Russian report WHY no
     criteria-preserving orchestrator or mcp change could make
     minimax succeed. Do not use VERDICT 1 as a shortcut. If there
     is any plausible path to a universal fix in orchestrator or
     mcp, take it as 3 or 2 instead. Only when minimax is
     demonstrably under-capable and every fix would require
     weakening acceptance criteria, state that and stop.

At the very end of your response, write the report in RUSSIAN. It
must include: what you found, what you did, and why. The final line
of the entire response MUST be exactly one of:

VERDICT: 1
VERDICT: 2
VERDICT: 3

Do not add text after that line. The supervisor parses it to decide
whether to whitelist the slice or just restart the orchestrator.
"""


def _resolve_claude_bin() -> str:
    found = shutil.which("claude")
    if found:
        return found
    candidate = Path("/home/jupiter/.local/bin/claude")
    if candidate.exists():
        return str(candidate)
    return "claude"  # last resort — let exec fail with a clear message


def build_claude_prompt(slice_id: str, allowed_slices: list[str]) -> str:
    allowed_repr = json.dumps(allowed_slices, ensure_ascii=False) if allowed_slices else "[]"
    return CLAUDE_PROMPT_TEMPLATE.format(
        slice_id=slice_id,
        allowed_slices_list=allowed_repr,
    )


def _print_stream_event(evt: dict, final_report_parts: list[str]) -> None:
    """Pretty-print one stream-json event and accumulate final text."""
    etype = evt.get("type")
    if etype == "assistant":
        msg = evt.get("message") or {}
        for block in msg.get("content", []) or []:
            btype = block.get("type")
            if btype == "text":
                text = block.get("text") or ""
                if text:
                    sys.stdout.write(text)
                    if not text.endswith("\n"):
                        sys.stdout.write("\n")
                    sys.stdout.flush()
            elif btype == "tool_use":
                name = block.get("name") or "?"
                sys.stdout.write(f"[claude tool_use: {name}]\n")
                sys.stdout.flush()
    elif etype == "result":
        # Final aggregated result — authoritative report text
        result_text = evt.get("result")
        if isinstance(result_text, str) and result_text:
            final_report_parts.append(result_text)
    elif etype == "system":
        sub = evt.get("subtype") or ""
        sys.stdout.write(f"[claude system: {sub}]\n")
        sys.stdout.flush()


def run_claude_investigator(prompt: str) -> tuple[str, int]:
    claude_bin = _resolve_claude_bin()
    cmd = [
        claude_bin,
        "--dangerously-skip-permissions",
        "--model",
        "opus",
        "--output-format",
        "stream-json",
        "--verbose",
        "--print",
        prompt,
    ]
    log(f"running claude investigator: {claude_bin} --model opus (bypass permissions)")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        log(f"error: claude CLI not found: {exc}")
        return "", 127

    final_report_parts: list[str] = []
    raw_buffer: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            raw_buffer.append(line)
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                try:
                    evt = json.loads(stripped)
                except json.JSONDecodeError:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                _print_stream_event(evt, final_report_parts)
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
    except KeyboardInterrupt:
        log("Ctrl+C during claude run — terminating claude")
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        raise
    rc = proc.wait()
    if final_report_parts:
        report = "\n".join(final_report_parts)
    else:
        report = "".join(raw_buffer)
    return report, rc


def parse_verdict(report_text: str) -> int | None:
    if not report_text:
        return None
    matches = VERDICT_PAT.findall(report_text)
    if not matches:
        return None
    # last VERDICT line wins — Claude's spec says it must be the final line
    try:
        return int(matches[-1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# config.toml start_from bump
# ---------------------------------------------------------------------------
def bump_start_from(config_path: Path) -> tuple[str, str] | None:
    text = config_path.read_text(encoding="utf-8")
    match = START_FROM_PAT.search(text)
    if not match:
        log(f"warning: no `start_from = \"vN\"` key found in {config_path}")
        return None
    current_n = int(match.group(2))
    if current_n >= MAX_PLAN_VERSION:
        log(f"start_from already at v{current_n} (MAX_PLAN_VERSION={MAX_PLAN_VERSION}) — not bumping")
        return None
    new_n = current_n + 1
    new_text = START_FROM_PAT.sub(
        lambda m: f'{m.group(1)}v{new_n}{m.group(3)}',
        text,
        count=1,
    )
    config_path.write_text(new_text, encoding="utf-8")
    return (f"v{current_n}", f"v{new_n}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
class Supervisor:
    def __init__(self) -> None:
        self._stop = False
        self._current_proc: subprocess.Popen | None = None

    def request_stop(self) -> None:
        self._stop = True
        proc = self._current_proc
        if proc is not None and proc.poll() is None:
            log("Ctrl+C received — stopping orchestrator and exiting")
            try:
                kill_orchestrator(proc)
            except Exception as exc:  # best-effort shutdown
                log(f"warning: error stopping orchestrator on shutdown: {exc}")

    def run(self) -> int:
        allowed = AllowedFallbackStore(ALLOWED_FILE)
        while not self._stop:
            cycle_rc = self._run_cycle(allowed)
            if self._stop:
                break
            if cycle_rc == "exhausted":
                log("all plan versions exhausted — sleeping; Ctrl+C to exit")
                while not self._stop:
                    time.sleep(1.0)
                break
            time.sleep(RESTART_COOLDOWN)
        log("supervisor exit")
        return 0

    def _run_cycle(self, allowed: AllowedFallbackStore) -> str:
        tailer = LogTailer(LOG_SYMLINK)
        # Skip any pre-existing content in the current symlink target — the
        # orchestrator restart will rotate the symlink anyway, but this keeps
        # us from printing stale lines if rotation is delayed.
        tailer.seek_to_current_eof()

        proc = start_orchestrator()
        self._current_proc = proc

        fallback_slice: str | None = None
        saw_any_fallback_attempt = False

        try:
            while proc.poll() is None and not self._stop:
                for line in tailer.poll():
                    m = FALLBACK_PAT.search(line)
                    if m is None:
                        m = FALLBACK_ATTEMPT_PAT.search(line)
                    if m is None:
                        continue
                    slice_id = m.group(1)
                    saw_any_fallback_attempt = True
                    if allowed.contains(slice_id):
                        log(f"fallback on whitelisted slice {slice_id} — ignoring")
                        continue
                    fallback_slice = slice_id
                    break
                if fallback_slice is not None:
                    break
                time.sleep(POLL_SLEEP)
        except KeyboardInterrupt:
            self.request_stop()
            tailer.close()
            return "stopped"

        if self._stop:
            tailer.close()
            return "stopped"

        if fallback_slice is not None:
            log(f"fallback detected on slice {fallback_slice} — killing orchestrator")
            kill_orchestrator(proc)
            self._current_proc = None
            # drain any final lines for context
            for _ in tailer.poll():
                pass
            tailer.close()

            prompt = build_claude_prompt(fallback_slice, allowed.list())
            sys.stdout.write("\n" + "=" * 72 + "\n")
            sys.stdout.write(f"[debuger] invoking Claude (opus, bypass) for slice {fallback_slice}\n")
            sys.stdout.write("=" * 72 + "\n")
            sys.stdout.flush()
            try:
                report, rc = run_claude_investigator(prompt)
            except KeyboardInterrupt:
                self._stop = True
                return "stopped"

            sys.stdout.write("\n" + "=" * 72 + "\n")
            sys.stdout.write(f"[debuger] Claude finished (rc={rc}) — final report:\n")
            sys.stdout.write("=" * 72 + "\n")
            sys.stdout.write(report.rstrip() + "\n")
            sys.stdout.write("=" * 72 + "\n\n")
            sys.stdout.flush()

            verdict = parse_verdict(report)
            if verdict == 1:
                allowed.add(fallback_slice)
                log(f"verdict 1 — slice {fallback_slice} whitelisted (future fallbacks on it ignored)")
            elif verdict == 2:
                log("verdict 2 — mcp repair done by Claude (tests + commit + push assumed) — restarting")
            elif verdict == 3:
                log("verdict 3 — orchestrator repair done by Claude — restarting")
            else:
                log("warning: no VERDICT: N found in Claude's report — restarting without whitelisting")
            return "fallback"

        # Orchestrator exited by itself
        rc = proc.returncode
        self._current_proc = None
        # drain trailing lines
        for _ in tailer.poll():
            pass
        tailer.close()

        if rc == 0 and not saw_any_fallback_attempt:
            bumped = bump_start_from(CONFIG_PATH)
            if bumped is None:
                log("clean completion but could not bump start_from — stopping loop")
                return "exhausted"
            old, new = bumped
            log(f"clean completion — config.toml start_from {old} → {new}")
            return "bumped"

        if rc == 0:
            log("orchestrator exited rc=0 but fallback was seen earlier this run — restarting")
        else:
            log(f"orchestrator exited rc={rc} — restarting")
        return "restart"


def _install_sigint(supervisor: Supervisor) -> None:
    def _handler(signum, frame):
        supervisor.request_stop()
        # do not re-raise — main loop checks _stop and exits cleanly
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    supervisor = Supervisor()
    _install_sigint(supervisor)
    try:
        return supervisor.run()
    except KeyboardInterrupt:
        supervisor.request_stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())
