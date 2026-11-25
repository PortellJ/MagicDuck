#!/usr/bin/env python3
"""Usage statistics for the JupyterHub container. Run it as a Hub-managed service.

Every STATS_SAMPLE_SECONDS (60) it stores one row in SQLite (STATS_DIR/stats.sqlite):
active users, running servers, CPU %, RAM %, disk read/write MB/s. It also counts file
opens (Linux inotify) in STATS_WATCH_DIRS, summed per hour. Rows older than
STATS_RETENTION_DAYS are deleted (round-robin style). Every few samples it redraws
three PNGs (24 h / 7 d / 30 d) and an index.html with top-10 file tables into STATS_DIR.
It serves nothing itself: the Hub (hub_extras.StatsHandler) shows those files to admins.

Environment (all optional)
--------------------------
  STATS_DIR             default /var/lib/jupyterhub/stats
  STATS_WATCH_DIRS      colon-separated directories to count file opens in, default /data
  STATS_TZ              time zone of the plots, e.g. Europe/Madrid (default UTC)
  STATS_TITLE           title of the page and plots, default "JupyterHub"
  STATS_SAMPLE_SECONDS  default 60
  STATS_RENDER_EVERY    redraw every N samples, default 5
  STATS_RETENTION_DAYS  default 35
  STATS_ACTIVE_WINDOW   seconds; a user is "active" if seen in this window, default 900
  JUPYTERHUB_API_URL / JUPYTERHUB_API_TOKEN   set by the Hub for managed services
"""
import ctypes
import ctypes.util
import errno
import html
import json
import logging
import math
import os
import signal
import sqlite3
import struct
import sys
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("stats_service")

STATS_DIR = Path(os.environ.get("STATS_DIR", "/var/lib/jupyterhub/stats"))
WATCH_DIRS = [d for d in os.environ.get("STATS_WATCH_DIRS", "/data").split(":") if d]
SAMPLE_SECONDS = float(os.environ.get("STATS_SAMPLE_SECONDS", "60"))
RENDER_EVERY = max(1, int(os.environ.get("STATS_RENDER_EVERY", "5")))
RETENTION_DAYS = float(os.environ.get("STATS_RETENTION_DAYS", "35"))
ACTIVE_WINDOW = int(os.environ.get("STATS_ACTIVE_WINDOW", "900"))
TITLE = os.environ.get("STATS_TITLE", "JupyterHub")
TOP_N = 10
RESCAN_SECONDS = 600

# (file key, heading, span in seconds, bucket width in seconds)
PERIODS = [
    ("day", "Last 24 hours", 86400, 300),          # 5-minute averages
    ("week", "Last 7 days", 7 * 86400, 1800),      # 30-minute averages
    ("month", "Last 30 days", 30 * 86400, 7200),   # 2-hour averages
]


def _tz():
    name = os.environ.get("STATS_TZ", "UTC")
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        log.warning("Unknown time zone %r (is tzdata installed?); using UTC", name)
        return timezone.utc


# ------------------------------------------------------------------ metrics
def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def mem_percent():
    """Container memory use in %, like `docker stats` (page cache excluded)."""
    cur, mx = _read("/sys/fs/cgroup/memory.current"), _read("/sys/fs/cgroup/memory.max")
    if cur and mx and mx != "max":
        used = int(cur)
        for line in (_read("/sys/fs/cgroup/memory.stat") or "").splitlines():
            key, _, val = line.partition(" ")
            if key == "inactive_file":
                used -= int(val)
                break
        return 100.0 * max(used, 0) / int(mx)
    cur = _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")  # cgroup v1
    mx = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cur and mx and int(mx) < (1 << 60):
        return 100.0 * int(cur) / int(mx)
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        return None


def cpu_counter():
    """(cumulative CPU microseconds, CPUs available) from this container's cgroup, or None."""
    usage = None
    for line in (_read("/sys/fs/cgroup/cpu.stat") or "").splitlines():
        if line.startswith("usage_usec"):
            usage = int(line.split()[1])
    if usage is None:
        return None
    ncpu = float(os.cpu_count() or 1)
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    if cpu_max and not cpu_max.startswith("max"):
        quota, period = cpu_max.split()[:2]
        ncpu = int(quota) / int(period)
    return usage, ncpu


def io_counters():
    """Cumulative (read_bytes, write_bytes, source): this container's cgroup io.stat when the
    io controller is enabled for it, else the whole machine's counters."""
    if "io" in (_read("/sys/fs/cgroup/cgroup.controllers") or "").split():
        r = w = 0
        for line in (_read("/sys/fs/cgroup/io.stat") or "").splitlines():
            for tok in line.split()[1:]:
                key, _, val = tok.partition("=")
                if key == "rbytes":
                    r += int(val)
                elif key == "wbytes":
                    w += int(val)
        return r, w, "container"
    try:
        import psutil
        c = psutil.disk_io_counters()
        if c:
            return c.read_bytes, c.write_bytes, "host"
    except Exception:
        pass
    return None


class Rates:
    """Turns cumulative counters into averages over the time since the previous call."""

    def __init__(self):
        self.prev = None

    def update(self):
        now, cpu, io = time.monotonic(), cpu_counter(), io_counters()
        prev, self.prev = self.prev, (now, cpu, io)
        out = {"cpu": None, "rd": None, "wr": None}
        if prev is None or now - prev[0] < 1:
            return out
        dt = now - prev[0]
        if cpu and prev[1]:
            out["cpu"] = max(0.0, 100.0 * (cpu[0] - prev[1][0]) / (dt * 1e6 * cpu[1]))
        if io and prev[2] and io[2] == prev[2][2]:
            out["rd"] = max(0.0, (io[0] - prev[2][0]) / dt / 1e6)
            out["wr"] = max(0.0, (io[1] - prev[2][1]) / dt / 1e6)
        return out


def hub_counts():
    """(active users, running servers) from the Hub API, or None if it is not reachable."""
    api, token = os.environ.get("JUPYTERHUB_API_URL"), os.environ.get("JUPYTERHUB_API_TOKEN")
    if not api or not token:
        return None
    servers = active = 0
    offset, limit = 0, 200
    now = datetime.now(timezone.utc)
    while True:
        req = urllib.request.Request(
            f"{api.rstrip('/')}/users?limit={limit}&offset={offset}",
            headers={"Authorization": f"token {token}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        items = data["items"] if isinstance(data, dict) else data
        for user in items:
            if not any(s.get("ready") for s in (user.get("servers") or {}).values()):
                continue
            servers += 1
            last = user.get("last_activity")
            if last:
                t = datetime.fromisoformat(last.replace("Z", "+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if (now - t).total_seconds() < ACTIVE_WINDOW:
                    active += 1
        if len(items) < limit:
            return active, servers
        offset += limit


# ------------------------------------------------------------------ storage
def open_db():
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATS_DIR / "stats.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS samples(
        ts INTEGER PRIMARY KEY, active INTEGER, servers INTEGER,
        cpu REAL, mem REAL, rd REAL, wr REAL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS file_opens(
        hour INTEGER, path TEXT, n INTEGER, PRIMARY KEY(hour, path))""")
    db.commit()
    return db


def store_sample(db, ts, counts, mem, rates):
    active, servers = counts if counts else (None, None)
    db.execute("INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?)",
               (ts, active, servers, rates["cpu"], mem, rates["rd"], rates["wr"]))


def store_opens(db, ts, counts):
    hour = ts // 3600 * 3600
    db.executemany(
        "INSERT INTO file_opens(hour, path, n) VALUES (?,?,?) "
        "ON CONFLICT(hour, path) DO UPDATE SET n = n + excluded.n",
        [(hour, path, n) for path, n in counts.items()],
    )


def prune(db, ts):
    cutoff = int(ts - RETENTION_DAYS * 86400)
    db.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
    db.execute("DELETE FROM file_opens WHERE hour < ?", (cutoff,))


# ------------------------------------------------------------- file opens
IN_OPEN = 0x00000020
IN_Q_OVERFLOW = 0x00004000
IN_ISDIR = 0x40000000


class FileWatcher:
    """Counts file opens under `roots` with inotify (one watch per directory)."""

    def __init__(self, roots):
        self.roots = roots
        self.libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        self.libc.inotify_init1.argtypes = [ctypes.c_int]
        self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.fd = self.libc.inotify_init1(0)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        self.wds = {}
        self.known = set()
        self.counts = Counter()
        self.overflows = 0
        self.lock = threading.Lock()
        self.limit_hit = False
        threading.Thread(target=self._reader, name="inotify", daemon=True).start()

    def scan(self):
        """Add watches for directories we don't watch yet (also run later, for new ones)."""
        for root in self.roots:
            if not os.path.isdir(root):
                log.warning("Watch directory %s does not exist", root)
                continue
            for dirpath, _dirs, _files in os.walk(root):
                if dirpath in self.known:
                    continue
                wd = self.libc.inotify_add_watch(self.fd, os.fsencode(dirpath), IN_OPEN)
                if wd < 0:
                    err = ctypes.get_errno()
                    if err == errno.ENOSPC:
                        if not self.limit_hit:
                            log.error("inotify watch limit reached (fs.inotify.max_user_watches): "
                                      "file statistics will be incomplete")
                        self.limit_hit = True
                        return
                    log.warning("Cannot watch %s: %s", dirpath, os.strerror(err))
                    continue
                with self.lock:
                    self.wds[wd] = dirpath
                self.known.add(dirpath)
        log.info("Watching %d directories for file opens", len(self.known))

    def _reader(self):
        header = struct.Struct("iIII")
        while True:
            try:
                buf = os.read(self.fd, 65536)
            except InterruptedError:
                continue
            except OSError as exc:
                log.error("inotify reader stopped: %s", exc)
                return
            local, off = Counter(), 0
            while off + header.size <= len(buf):
                wd, mask, _cookie, length = header.unpack_from(buf, off)
                name = buf[off + header.size: off + header.size + length].split(b"\0", 1)[0]
                off += header.size + length
                if mask & IN_Q_OVERFLOW:
                    self.overflows += 1
                    continue
                if mask & IN_ISDIR or not name:
                    continue
                directory = self.wds.get(wd)
                if directory is not None:
                    local[os.path.join(directory, os.fsdecode(name))] += 1
            if local:
                with self.lock:
                    self.counts.update(local)

    def take(self):
        with self.lock:
            counts, self.counts = self.counts, Counter()
        return counts


# -------------------------------------------------------------------- plots
_SERIES = "SELECT (ts/?)*? AS b, AVG(active), MAX(active), AVG(servers), MAX(servers), " \
          "AVG(cpu), MAX(cpu), AVG(mem), MAX(mem), AVG(rd), MAX(rd), AVG(wr), MAX(wr) " \
          "FROM samples WHERE ts >= ? GROUP BY b"


def _atomic_write(path, data, mode="wb"):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, mode) as fh:
        fh.write(data)
    os.replace(tmp, path)


def render_plots(db, tz, now=None):
    os.environ.setdefault("MPLCONFIGDIR", str(STATS_DIR / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    now = int(now or time.time())
    for key, heading, span, width in PERIODS:
        start = (now - span) // width * width
        by_bucket = {r[0]: r for r in db.execute(_SERIES, (width, width, start))}
        grid = list(range(start, now + width, width))
        x = [datetime.fromtimestamp(b + width / 2, tz) for b in grid]

        def col(i):  # NaN where there is no data: the line breaks (e.g. while the hub was down)
            return [by_bucket[b][i] if b in by_bucket and by_bucket[b][i] is not None else math.nan
                    for b in grid]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8.4), constrained_layout=True)

        def band(ax, avg, mx, color, label):
            ax.fill_between(x, avg, mx, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, avg, color=color, linewidth=1.4, label=label)

        def top(*series, floor, headroom=1.15, cap=None):
            vals = [v for s in series for v in s if not math.isnan(v)]
            hi = max([floor] + vals) * headroom
            return min(hi, cap) if cap else hi

        ax = axes[0]
        sv_a, sv_m, ac_a, ac_m = col(3), col(4), col(1), col(2)
        band(ax, sv_a, sv_m, "#1f77b4", "running servers")
        band(ax, ac_a, ac_m, "#ff7f0e", "active users")
        ax.set_ylim(0, top(sv_m, ac_m, floor=2))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("users")

        ax = axes[1]
        cpu_a, cpu_m, mem_a, mem_m = col(5), col(6), col(7), col(8)
        band(ax, cpu_a, cpu_m, "#d62728", "CPU %")
        band(ax, mem_a, mem_m, "#2ca02c", "RAM %")
        ax.set_ylim(0, max(100, top(cpu_m, mem_m, floor=100, headroom=1.02)))
        ax.set_ylabel("percent")

        ax = axes[2]
        rd_a, rd_m, wr_a, wr_m = col(9), col(10), col(11), col(12)
        band(ax, rd_a, rd_m, "#1f77b4", "disk read")
        band(ax, wr_a, wr_m, "#9467bd", "disk write")
        ax.set_ylim(0, top(rd_m, wr_m, floor=1))
        ax.set_ylabel("MB/s")

        for a in axes:
            a.grid(True, alpha=0.3)
            a.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
            a.set_xlim(x[0], x[-1])
        if key == "day":
            loc, fmt = mdates.HourLocator(interval=2, tz=tz), "%H:%M"
        elif key == "week":
            loc, fmt = mdates.DayLocator(tz=tz), "%a %d"
        else:
            loc, fmt = mdates.DayLocator(interval=3, tz=tz), "%d %b"
        axes[2].xaxis.set_major_locator(loc)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=tz))
        fig.suptitle(f"{TITLE}: {heading.lower()}", fontsize=13)
        fig.text(0.995, 0.995,
                 f"line = average over {width // 60} min, shaded = peak - generated "
                 f"{datetime.fromtimestamp(now, tz):%Y-%m-%d %H:%M %Z}",
                 ha="right", va="top", fontsize=7, color="#666")
        tmp = STATS_DIR / f"usage-{key}.png.tmp"
        fig.savefig(tmp, dpi=100, format="png")
        plt.close(fig)
        os.replace(tmp, STATS_DIR / f"usage-{key}.png")


_CSS = """body{font-family:system-ui,sans-serif;max-width:1050px;margin:1.5rem auto;padding:0 1rem;color-scheme:light dark}
img{max-width:100%;height:auto;background:#fff;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin-bottom:1.5rem}
th,td{padding:.25rem .6rem;text-align:left;border-bottom:1px solid rgba(128,128,128,.3)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
td.p{word-break:break-all;font-family:ui-monospace,monospace;font-size:.85em}
.muted{opacity:.7;font-size:.9em}"""


def render_index(db, tz, now=None, watching=True):
    now = int(now or time.time())
    esc = html.escape
    last = db.execute("SELECT ts, active, servers, cpu, mem, rd, wr FROM samples "
                      "ORDER BY ts DESC LIMIT 1").fetchone()
    first = db.execute("SELECT MIN(ts) FROM samples").fetchone()[0]
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
           f"<meta http-equiv=refresh content=120><title>{esc(TITLE)} usage</title>"
           f"<style>{_CSS}</style></head><body>",
           f"<h1>{esc(TITLE)}: usage statistics</h1>",
           f'<p class="muted"><a href="../home">&larr; back to the hub</a> &middot; page generated '
           f"{datetime.fromtimestamp(now, tz):%Y-%m-%d %H:%M %Z}"
           + (f" &middot; data since {datetime.fromtimestamp(first, tz):%Y-%m-%d %H:%M}" if first else "")
           + "</p>"]

    def fmt(v, spec):
        return "n/a" if v is None else format(v, spec)

    if last:
        out.append(
            f"<p><b>Latest sample</b> ({datetime.fromtimestamp(last[0], tz):%H:%M}): "
            f"{fmt(last[1], 'd')} active users, {fmt(last[2], 'd')} servers running, "
            f"CPU {fmt(last[3], '.0f')}%, RAM {fmt(last[4], '.0f')}%, "
            f"disk read {fmt(last[5], '.1f')} / write {fmt(last[6], '.1f')} MB/s</p>")
    for key, heading, _span, _width in PERIODS:
        out.append(f'<h2>{heading}</h2><img src="usage-{key}.png" alt="{esc(heading)}">')
    out.append("<h2>Most opened files</h2>")
    if not watching:
        out.append('<p class="muted">File statistics are off: none of the watch directories exists.</p>')
    for key, heading, span, _width in PERIODS:
        if not watching:
            break
        since = (now - span) // 3600 * 3600
        rows = db.execute("SELECT path, SUM(n) AS c FROM file_opens WHERE hour >= ? "
                          "GROUP BY path ORDER BY c DESC, path LIMIT ?", (since, TOP_N)).fetchall()
        out.append(f"<h3>{heading}</h3>")
        if not rows:
            out.append('<p class="muted">No file opens recorded yet.</p>')
            continue
        out.append('<table><tr><th class="n">#</th><th>File</th><th class="n">Opens</th></tr>')
        for i, (path, count) in enumerate(rows, 1):
            out.append(f'<tr><td class="n">{i}</td><td class="p">{esc(path)}</td>'
                       f'<td class="n">{count}</td></tr>')
        out.append("</table>")
    out.append('<p class="muted">Opens are counted per file open, not per byte read, '
               "and cannot be attributed to individual users.</p></body></html>")
    _atomic_write(STATS_DIR / "index.html", "\n".join(out), mode="w")


# --------------------------------------------------------------------- main
def sample_once(db, rates, watcher):
    ts = int(time.time())
    try:
        counts = hub_counts()
    except Exception as exc:  # hub starting up or restarting: leave a gap, try again next minute
        log.warning("Hub API not reachable: %s", exc)
        counts = None
    store_sample(db, ts, counts, mem_percent(), rates.update())
    if watcher is not None:
        opens = watcher.take()
        if opens:
            store_opens(db, ts, opens)
    prune(db, ts)
    db.commit()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    tz, db, rates = _tz(), open_db(), Rates()
    watcher = None
    try:
        watcher = FileWatcher(WATCH_DIRS)
        watcher.scan()
    except Exception:
        log.exception("File-open counting is disabled")
    watching = bool(watcher and watcher.known)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    log.info("Collecting every %gs into %s (watching %s)", SAMPLE_SECONDS, STATS_DIR, WATCH_DIRS)
    next_t, n, last_scan = time.monotonic(), 0, time.monotonic()
    while not stop.is_set():
        try:
            sample_once(db, rates, watcher)
            n += 1
            if watcher is not None and time.monotonic() - last_scan > RESCAN_SECONDS:
                watcher.scan()
                watching = bool(watcher.known)
                last_scan = time.monotonic()
            if n == 1 or n % RENDER_EVERY == 0:
                render_plots(db, tz)
                render_index(db, tz, watching=watching)
        except Exception:
            log.exception("Sampling or drawing failed; will retry")
        next_t += SAMPLE_SECONDS
        delay = next_t - time.monotonic()
        if delay < -SAMPLE_SECONDS:  # fell far behind (e.g. the machine was suspended)
            next_t = time.monotonic()
            delay = 0
        stop.wait(max(0.0, delay))
    log.info("Stopped")
    db.close()


if __name__ == "__main__":
    sys.exit(main())
