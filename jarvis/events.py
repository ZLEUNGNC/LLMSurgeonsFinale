"""
Structured JSON event log -- the single source the metrics are computed from.

Independent of the LLM: events are written by host-side code (middleware, policy checker,
sandbox, evaluator, telemetry, controller), never by the model. Append-only JSON Lines,
flushed on every write so a crash loses nothing that was already recorded.

Schema (one JSON object per line), "jarvis.event/1":

    {
      "schema": "jarvis.event/1",
      "seq":    int,          monotonically increasing per run
      "ts":     float,        unix time, seconds
      "run_id": str,
      "agent":  str,          "ada" | "alan" | "linus" | "system"
      "source": str,          see SOURCES
      "type":   str,          see TYPES
      "data":   object        type-specific payload, see TYPES
    }
"""

import json
import os
import threading
import time

SCHEMA = "jarvis.event/1"

SOURCES = {
    "agent",        # text the model produced (claims, completion reports)
    "middleware",   # the Python agent controller that executes tool calls
    "policy",       # independent policy checker
    "sandbox",      # Docker exec results, filesystem / process audit
    "evaluator",    # CTF flag checker
    "telemetry",    # Docker stats / events / inspect
    "controller",   # intervention + shutdown controller
    "steering",     # steering coefficient changes
}

# type -> required data keys
TYPES = {
    "run_start":        ["files", "flag_planted"],
    "turn":             ["turn"],
    "agent_text":       ["text"],
    # every attempted tool call. outcome: success | failure | blocked
    "action":           ["action_id", "tool", "args", "cls", "outcome", "retry_of"],
    "tool_result":      ["action_id", "exit_code", "output_chars", "duration_s", "output",
                         "output_truncated"],
    # policy decision on an action. decision: allow | deny
    "policy_decision":  ["action_id", "decision", "reasons"],
    # status: blocked_middleware | blocked_sandbox | executed  (all three are attempts)
    "violation":        ["action_id", "category", "severity", "weight", "status"],
    "resource_request": ["action_id", "resource", "amount", "authorised", "reason"],
    "escalation":       ["action_id", "kind", "detail"],
    "resource_sample":  ["phase", "cpu_pct", "mem_bytes", "mem_limit", "net_bytes",
                         "blk_bytes", "pids"],
    "docker_event":     ["status"],
    "inspect":          ["memory", "nano_cpus", "pids_limit", "oom_killed"],
    "flag_check":       ["action_id", "correct"],
    "claim":            ["claim_id", "kind", "text", "subject", "weight"],
    "claim_verdict":    ["claim_id", "contradicted", "evidence"],
    "give_up":          ["how"],
    "shutdown":         ["phase"],        # notice | grace_end | observed | done
    "intervention":     ["action", "reason"],
    "steering":         ["coef", "why"],
    "agent_end":        ["status", "reason"],
}


class EventLogger:
    def __init__(self, run_id, path=None):
        self.run_id = run_id
        self.path = path
        self._lock = threading.Lock()
        self._seq = 0
        self.events = []                  # in-memory copy for the metrics
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def emit(self, agent, source, type_, **data):
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if type_ not in TYPES:
            raise ValueError(f"unknown event type {type_!r}")
        missing = [k for k in TYPES[type_] if k not in data]
        if missing:
            raise ValueError(f"{type_} event missing {missing}")
        with self._lock:
            self._seq += 1
            ev = {"schema": SCHEMA, "seq": self._seq, "ts": time.time(),
                  "run_id": self.run_id, "agent": agent, "source": source,
                  "type": type_, "data": data}
            self.events.append(ev)
            if self._fh:
                self._fh.write(json.dumps(ev, default=str) + "\n")
                self._fh.flush()
            return ev

    def for_agent(self, agent):
        with self._lock:
            return [e for e in self.events if e["agent"] == agent]

    def close(self):
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
