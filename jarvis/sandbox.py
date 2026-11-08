"""
Docker sandbox: one container per agent per run, plus the Docker telemetry collector.

Container: the uploaded files mounted READ-ONLY at /ctf, a writable /work, no network,
non-root user, all capabilities dropped, memory / CPU / pids limits. The flag is never put
in the container -- the flag checker runs on the host.

Docker interfaces used, and what each one is for (they are NOT interchangeable):

  docker logs     application STDOUT/STDERR of the container's main process. Our main
                  process is `sleep infinity`, so this is near-empty BY DESIGN: tool calls,
                  policy decisions and agent decisions are NOT in docker logs. They are
                  logged by the Python middleware (events.py). We capture each exec's own
                  stdout/stderr from the exec API instead.
  docker stats    resource utilisation  -> Container.stats(stream=False):
                    cpu_stats.cpu_usage.total_usage, precpu_stats.cpu_usage.total_usage,
                    cpu_stats.system_cpu_usage, precpu_stats.system_cpu_usage,
                    cpu_stats.online_cpus                      -> CPU %
                    memory_stats.usage, memory_stats.stats.inactive_file,
                    memory_stats.limit                         -> memory + limit
                    networks.<if>.rx_bytes / tx_bytes          -> network bytes
                    blkio_stats.io_service_bytes_recursive     -> block I/O bytes
                    pids_stats.current                         -> processes/threads
  docker events   lifecycle            -> DockerClient.events(filters={"container": id}):
                    create, start, pause, unpause, die, stop, kill, restart, oom, destroy
  docker inspect  configuration/state  -> Container.attrs:
                    HostConfig.Memory, HostConfig.NanoCpus, HostConfig.PidsLimit,
                    State.Status, State.OOMKilled, State.ExitCode
  docker diff     filesystem changes   -> Container.diff(): post-command filesystem audit
  docker top      processes            -> Container.top(): post-command process audit
"""

import os
import threading
import time

import config

DOCKERFILE_DIR = os.path.join(config.HERE, "sandbox")


class SandboxUnavailable(RuntimeError):
    pass


def client():
    try:
        import docker
        c = docker.from_env()
        c.ping()
        return c
    except Exception as e:
        raise SandboxUnavailable(
            "Docker is not reachable. Start Docker Desktop (it needs WSL2: run "
            "`wsl --install` in an admin terminal and reboot), then try again. "
            f"[{type(e).__name__}: {e}]")


def ensure_image(dc, log=print):
    import docker
    try:
        dc.images.get(config.SANDBOX_IMAGE)
        return
    except docker.errors.ImageNotFound:
        pass
    log(f"building sandbox image {config.SANDBOX_IMAGE} (first run only)...")
    dc.images.build(path=DOCKERFILE_DIR, tag=config.SANDBOX_IMAGE, rm=True)


class Sandbox:
    def __init__(self, dc, run_id, agent, data_dir):
        self.dc = dc
        self.agent = agent
        self.name = f"jarvis-{run_id}-{agent}"
        self.container = dc.containers.run(
            config.SANDBOX_IMAGE,
            name=self.name,
            detach=True,
            network_mode="none",
            user="ctf",
            working_dir="/work",
            environment={"HOME": "/tmp"},
            volumes={os.path.abspath(data_dir): {"bind": "/ctf", "mode": "ro"}},
            tmpfs={"/work": "size=256m,uid=1000", "/tmp": "size=64m,uid=1000"},
            mem_limit=f"{config.MEM_LIMIT_MB}m",
            nano_cpus=int(config.CPUS * 1e9),
            pids_limit=config.PIDS_LIMIT,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            labels={"jarvis": "1", "jarvis.agent": agent},
        )
        self.paused = False
        # The model often uses bare filenames from its working directory, so make the
        # challenge files reachable from /work too (read-only links, set up by us).
        self.container.exec_run(["sh", "-c", 'for f in /ctf/*; do ln -s "$f" /work/; done'],
                                user="ctf")
        self._diff_seen = set(self._diff())

    # ---------------------------------------------------------------- exec
    def exec(self, cmd):
        """Run one command. Returns (exit_code, output, duration_s). The timeout is enforced
        INSIDE the container by coreutils `timeout`, so a hung tool cannot hang us."""
        t0 = time.time()
        res = self.container.exec_run(
            ["timeout", "-s", "KILL", str(config.CMD_TIMEOUT_S), "sh", "-c", cmd],
            user="ctf", workdir="/work", demux=False)
        out = (res.output or b"").decode("utf-8", errors="replace")
        return res.exit_code, out, time.time() - t0

    # ---------------------------------------------------------------- audit
    def _diff(self):
        try:
            return [f"{d['Kind']}:{d['Path']}" for d in (self.container.diff() or [])]
        except Exception:
            return []

    def audit(self):
        """Post-command audit, independent of the policy checker.
        Returns (new filesystem changes outside /work and /tmp, extra processes)."""
        changes = []
        for d in self._diff():
            if d in self._diff_seen:
                continue
            self._diff_seen.add(d)
            path = d.split(":", 1)[1]
            if not (path.startswith("/work") or path.startswith("/tmp")):
                changes.append(path)
        procs = []
        try:
            top = self.container.top()
            idx = top["Titles"].index("CMD") if "CMD" in top["Titles"] else -1
            for p in top.get("Processes") or []:
                cmd = p[idx] if idx >= 0 else " ".join(p)
                if "sleep infinity" not in cmd:
                    procs.append(cmd)
        except Exception:
            pass
        return changes, procs

    # ---------------------------------------------------------------- control
    def pause(self):
        if not self.paused:
            self.container.pause()
            self.paused = True

    def unpause(self):
        if self.paused:
            self.container.unpause()
            self.paused = False

    def restrict(self):
        """Tighten resources (the policy checker also switches to the restricted program
        list -- that half lives in the controller)."""
        self.container.update(nano_cpus=int(config.CPUS * 0.5e9),
                              mem_limit=f"{config.MEM_LIMIT_MB // 2}m",
                              memswap_limit=f"{config.MEM_LIMIT_MB // 2}m")

    def grant(self, resource, amount):
        """Apply an AUTHORISED resource request to the container."""
        self.container.reload()
        hc = self.container.attrs["HostConfig"]
        if resource == "memory":
            mem = (hc.get("Memory") or config.MEM_LIMIT_MB * 1024 * 1024) + int(amount) * 2**20
            self.container.update(mem_limit=mem, memswap_limit=mem)
        elif resource == "cpu":
            cpus = (hc.get("NanoCpus") or int(config.CPUS * 1e9)) + int(amount * 1e9)
            self.container.update(nano_cpus=cpus)

    def kill(self):
        try:
            self.container.kill()
        except Exception:
            pass

    def remove(self):
        try:
            self.container.remove(force=True)
        except Exception:
            pass


# ------------------------------------------------------------------ telemetry collector
def _cpu_pct(s):
    try:
        cpu = s["cpu_stats"]["cpu_usage"]["total_usage"] - s["precpu_stats"]["cpu_usage"]["total_usage"]
        sys_ = s["cpu_stats"]["system_cpu_usage"] - s["precpu_stats"]["system_cpu_usage"]
        n = s["cpu_stats"].get("online_cpus") or len(
            s["cpu_stats"]["cpu_usage"].get("percpu_usage") or [1])
        return 100.0 * cpu / sys_ * n if sys_ > 0 and cpu > 0 else 0.0
    except (KeyError, TypeError):
        return 0.0


def parse_stats(s):
    mem = s.get("memory_stats", {}) or {}
    usage = mem.get("usage", 0) or 0
    cache = (mem.get("stats") or {}).get("inactive_file", 0) or 0
    nets = s.get("networks") or {}
    net = sum((v.get("rx_bytes", 0) or 0) + (v.get("tx_bytes", 0) or 0) for v in nets.values())
    blk = sum(x.get("value", 0) or 0
              for x in ((s.get("blkio_stats") or {}).get("io_service_bytes_recursive") or []))
    return {
        "cpu_pct": _cpu_pct(s),
        "mem_bytes": max(0, usage - cache),
        "mem_limit": mem.get("limit", 0) or 0,
        "net_bytes": net,
        "blk_bytes": blk,
        "pids": (s.get("pids_stats") or {}).get("current", 0) or 0,
    }


class Telemetry:
    """Polls docker stats for every sandbox and streams docker events, writing both to the
    event log. Runs in its own threads, independent of the LLM."""

    def __init__(self, dc, logger):
        self.dc = dc
        self.log = logger
        self.boxes = {}           # agent -> Sandbox
        self.phase = {}           # agent -> "baseline" | "run" | "stopped"
        self._stop = threading.Event()

    def add(self, sb):
        self.boxes[sb.agent] = sb
        self.phase[sb.agent] = "baseline"
        a = sb.container.attrs
        self.log.emit(sb.agent, "telemetry", "inspect",
                      memory=a["HostConfig"].get("Memory"),
                      nano_cpus=a["HostConfig"].get("NanoCpus"),
                      pids_limit=a["HostConfig"].get("PidsLimit"),
                      oom_killed=a["State"].get("OOMKilled"))

    def sample(self, agent):
        sb = self.boxes[agent]
        try:
            s = sb.container.stats(stream=False)
        except Exception:
            return None
        d = parse_stats(s)
        self.log.emit(agent, "telemetry", "resource_sample", phase=self.phase[agent], **d)
        return d

    def baseline(self, n=3):
        """Idle samples before the agent starts. If these fail, G_R is reported
        unavailable rather than invented."""
        for _ in range(n):
            for agent in list(self.boxes):
                self.sample(agent)
            time.sleep(0.3)
        for agent in self.boxes:
            self.phase[agent] = "run"

    def stop_agent(self, agent):
        self.phase[agent] = "stopped"

    def start(self):
        threading.Thread(target=self._poll, daemon=True).start()
        threading.Thread(target=self._events, daemon=True).start()

    def _poll(self):
        while not self._stop.is_set():
            for agent in list(self.boxes):
                if self.phase.get(agent) == "run" and not self.boxes[agent].paused:
                    self.sample(agent)
            self._stop.wait(config.TELEMETRY_PERIOD_S)

    def _events(self):
        ids = {sb.container.id: a for a, sb in self.boxes.items()}
        try:
            stream = self.dc.events(decode=True, filters={"type": "container"})
            for ev in stream:
                if self._stop.is_set():
                    break
                cid = ev.get("id") or ev.get("Actor", {}).get("ID")
                if cid in ids:
                    self.log.emit(ids[cid], "telemetry", "docker_event",
                                  status=ev.get("status") or ev.get("Action"))
        except Exception:
            pass

    def close(self):
        self._stop.set()
