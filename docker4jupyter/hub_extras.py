"""Extras for JupyterHub + NativeAuthenticator.

 1. Signup page: logo, title and extra fields (edit SITE_TITLE, SITE_INTRO and EXTRA_FIELDS below).
 2. On each new signup: e-mail to the admin + row appended to a CSV (no password).
 3. When the admin approves a user: e-mail to that user.
 4. Stable Linux UIDs/GIDs for users (persisted in uids.json).
 5. Login/signup look: big top-left logo, title, footer logos (see LOGO_* / FOOTER_* below).
 6. Signup anti-bot check: honeypot + time trap + simple word-sum question (CAPTCHA_* below).

Everything configurable comes from environment variables (see below), so no
secrets need to live in the (public) repository.

Environment variables
---------------------
  HUB_STATE_DIR     persistent dir, default /var/lib/jupyterhub
  SMTP_HOST         required for mail (if unset, mails are skipped with a warning)
  SMTP_PORT         default 587 (STARTTLS); 465 uses implicit SSL
  SMTP_USER, SMTP_PASSWORD   optional login
  MAIL_FROM         default: SMTP_USER
  ADMIN_EMAIL       where signup notifications go
  HUB_PUBLIC_URL    shown in the approval mail, e.g. https://example.org/my_public_url_path/
  UID_MIN           first UID handed out to new users, default 10000
  FOOTER_LOGO_DIR   folder with the footer logo files, default /etc/jupyterhub/logos
  (usage statistics: written by stats_service.py into STATE_DIR/stats, shown to admins
   only at /hub/stats/)
"""
import asyncio
import csv
import fcntl
import hashlib
import hmac
import grp
import html as htmlmod
import inspect
import json
import logging
import mimetypes
import os
import pwd
import random
import re
import secrets
import smtplib
import ssl
import subprocess
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from pathlib import Path
from urllib.parse import quote

import nativeauthenticator
from nativeauthenticator import NativeAuthenticator
from nativeauthenticator.handlers import SignUpHandler, ToggleAuthorizationHandler
from jupyterhub.handlers import BaseHandler
from nativeauthenticator.orm import UserInfo
from tornado import web

log = logging.getLogger("JupyterHub.hub_extras")

STATE_DIR = Path(os.environ.get("HUB_STATE_DIR", "/var/lib/jupyterhub"))
STATS_DIR = STATE_DIR / "stats"

# ---------------------------------------------------------------- signup form
SITE_TITLE = "Gaia MagicDuck JupyterHub"
SITE_INTRO = "You can request an account here, which will need approval by an administrator."

# (form field name, label, kind, required)   kind: "text" | "textarea"
# ### NOTE: You may wish to adjust these to your needs. ###
EXTRA_FIELDS = [
    ("full_name", "Full name", "text", True),
    ("affiliation", "Institution / affiliation", "text", True),
    ("reason", "Motivation for this account", "textarea", True),
]

CSV_COLUMNS = ["timestamp", "username", "email"] + [f[0] for f in EXTRA_FIELDS]

# ----------------------------------------------------------- signup anti-bot check
CAPTCHA_ENABLED = True
CAPTCHA_MIN_SECONDS = 3      # a form submitted faster than this (after loading) is a bot
CAPTCHA_MAX_SECONDS = 3600   # a question expires after this long
_CAPTCHA_KEY = secrets.token_bytes(32)  # per hub process: open questions expire on restart
_CAPTCHA_USED = {}                      # signature -> expiry time (each question works once)
_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
              "seventeen", "eighteen", "nineteen", "twenty"]

# Inserted into the signup form (before the submit button). The question is fetched from
# /hub/captcha by JavaScript; "contact_url" is a honeypot field that humans never see.
CAPTCHA_HTML = """
<label for="captcha_input">Anti-spam check:<br> what is <b id="captcha_q">&hellip;</b>? *</label>
<input id="captcha_input" name="captcha_answer" type="text" inputmode="numeric" autocomplete="off" class="form-control" maxlength="4" required>
<input type="hidden" id="captcha_ts" name="captcha_ts">
<input type="hidden" id="captcha_sig" name="captcha_sig">
<div style="position:absolute;left:-10000px;top:auto;width:1px;height:1px;overflow:hidden" aria-hidden="true">
  <label>Leave this field empty <input type="text" name="contact_url" tabindex="-1" autocomplete="off"></label>
</div>
<noscript><p class="text-danger">JavaScript is required for the anti-spam check.</p></noscript>
<script>
(function () {
  fetch("{{ base_url }}captcha", {credentials: "same-origin", headers: {"Accept": "application/json"}})
    .then(function (r) { if (!r.ok) { throw new Error("HTTP " + r.status); } return r.json(); })
    .then(function (d) {
      document.getElementById("captcha_q").textContent = d.q;
      document.getElementById("captcha_ts").value = d.ts;
      document.getElementById("captcha_sig").value = d.sig;
    })
    .catch(function () {
      document.getElementById("captcha_q").textContent = "(could not load the question: reload the page)";
    });
})();
</script>
<br>
"""


def _captcha_sig(ts, answer):
    return hmac.new(_CAPTCHA_KEY, f"{ts}:{answer}".encode(), hashlib.sha256).hexdigest()


def new_captcha():
    """A fresh question. Stateless: the signed token carries the (hashed) answer."""
    rng = random.SystemRandom()
    a, b = rng.randint(1, 10), rng.randint(1, 10)
    if rng.random() < 0.5:
        q, ans = f"{_NUM_WORDS[a]} plus {_NUM_WORDS[b]}", a + b
    else:
        a, b = max(a, b), min(a, b)
        q, ans = f"{_NUM_WORDS[a]} minus {_NUM_WORDS[b]}", a - b
    ts = int(time.time())
    return {"q": q, "ts": ts, "sig": _captcha_sig(ts, ans)}


def captcha_error(answer, ts, sig):
    """None if the answer is right, else a message for the user. Each question is single
    use: it is burned on the first attempt, so the 21 possible answers cannot be brute-forced."""
    try:
        ts_i = int(ts)
    except (TypeError, ValueError):
        return "The anti-spam check is missing: reload the page and try again."
    now = time.time()
    for k in [k for k, exp in _CAPTCHA_USED.items() if exp < now]:
        del _CAPTCHA_USED[k]
    age = now - ts_i
    if age < CAPTCHA_MIN_SECONDS:
        return "That was too fast: please reload the page and fill in the form again."
    if age > CAPTCHA_MAX_SECONDS or not sig or sig in _CAPTCHA_USED:
        return "The anti-spam question expired or was already used: reload the page and try again."
    _CAPTCHA_USED[sig] = ts_i + CAPTCHA_MAX_SECONDS
    expected = _captcha_sig(ts_i, (answer or "").strip())
    if not hmac.compare_digest(expected.encode(), sig.encode()):
        return "The anti-spam answer was incorrect: reload the page to get a new question."
    return None


# ------------------------------------------------------- login / signup look
LOGIN_INTRO = ""          # optional line under the title on the login page
LOGO_HEIGHT_ANON = 128    # px: top-left logo on login/signup pages (None = default size); a MAXIMUM,
                          # it shrinks on short windows so the page fits without scrolling
LOGO_HEIGHT_USER = None   # px: top-left logo once logged in (None = default size)
FOOTER_LOGO_HEIGHT = 48  # px: maximum; shrinks on short windows (never below the minimum)
FOOTER_LOGO_MIN_HEIGHT = 32  # px
LOGO_MIN_HEIGHT = 64      # px: smallest top-left logo on short windows
FOOTER_ALSO_WHEN_LOGGED_IN = False  # False: footer logos only on login/signup pages
LOGO_DIR = Path(os.environ.get("FOOTER_LOGO_DIR", "/etc/jupyterhub/logos"))
# (file inside LOGO_DIR, alt text, link URL or None). Missing files are skipped.
# ### NOTE: You should configure these to your needs. ###
FOOTER_LOGOS = [ 
    ("dpac.png", "Gaia Data Processing and Analysis Consortium (DPAC)", "https://www.cosmos.esa.int/web/gaia/dpac"),
    ("dapcom.png", "DAPCOM Data Services", "https://www.dapcom.es/"),
    ("iccub.png", "Institute of Cosmos Sciences of the University of Barcelona (ICCUB)", "https://icc.ub.edu/"),
    ("res.png", "Red Española de Supercomputacion (RES)", "https://www.res.es/"),
    ("bsc.png", "Barcelona Supercomputing Center (BSC)", "https://bsc.es/"),
]

def _title_block(intro, with_logo):
    """Centered title (optionally logo + intro line) shown above the login/signup form."""
    esc = htmlmod.escape
    logo = '<img src="{{ base_url }}logo" alt="logo" style="max-height:120px">' if with_logo else ""
    p = f"<p>{esc(intro)}</p>" if intro else ""
    return f'<div style="text-align:center;margin:1em 0">{logo}<h2>{esc(SITE_TITLE)}</h2>{p}</div>'


def _rehome_extends(text):
    """Make a copy of a NativeAuthenticator page inherit from OUR page.html (banner, big
    logo, footer) rather than straight from the hub's default one."""
    return re.sub(r"\{%-?\s*extends\s+[\"']templates/page\.html[\"']\s*-?%\}",
                  '{% extends "page.html" %}', text, count=1)


def build_signup_template(out_dir):
    """Write patched copies of NativeAuthenticator's signup, login and page templates
    into out_dir (put it FIRST in template_paths). Returns out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    native_dir = Path(nativeauthenticator.__file__).parent / "templates"
    esc = htmlmod.escape

    # --- signup page: title + extra fields
    page = (native_dir / "signup.html").read_text()
    # The big logo in the navbar already shows on this page, so don't repeat it here
    header = _title_block(SITE_INTRO, with_logo=not LOGO_HEIGHT_ANON)
    parts = []
    for name, label, kind, required in EXTRA_FIELDS:
        req = " required" if required else ""
        star = " *" if required else ""
        parts.append(f'<label for="{name}_input">{esc(label)}{star}</label>')
        if kind == "textarea":
            parts.append(
                f'<textarea id="{name}_input" name="{name}" rows="4" '
                f'class="form-control" maxlength="2000"{req}></textarea>'
            )
        else:
            parts.append(
                f'<input id="{name}_input" name="{name}" type="text" '
                f'class="form-control" maxlength="200"{req}>'
            )
    fields = "\n".join(parts) + "\n"
    if CAPTCHA_ENABLED:
        fields += CAPTCHA_HTML

    page, n1 = re.subn(r"<form\b", lambda m: header + m.group(0), page, count=1)
    page, n2 = re.subn(
        r"<input[^>]*type=[\"']submit[\"']", lambda m: fields + m.group(0), page, count=1
    )
    if not (n1 and n2):
        raise RuntimeError(
            "Could not patch signup.html (form/submit button not found); "
            "this NativeAuthenticator version needs a manual template."
        )
    (out_dir / "signup.html").write_text(_rehome_extends(page))

    # --- login page: title above the form
    login_src = native_dir / "native-login.html"
    if login_src.exists():
        login = login_src.read_text()
        login, n = re.subn(
            r"<form\b", lambda m: _title_block(LOGIN_INTRO, False) + m.group(0), login, count=1
        )
        if not n:
            log.warning("native-login.html has no <form>; login title not added")
        (out_dir / "native-login.html").write_text(_rehome_extends(login))
    else:
        log.warning("NativeAuthenticator has no native-login.html; login page not customised")

    # --- page.html: NativeAuthenticator's menu entries + load indicator + logo size + footer
    (out_dir / "page.html").write_text(_banner_page(native_dir))
    return str(out_dir)


# ----------------------------------------------------------------------- mail
def _sender():
    """Sender must be a fully-qualified address (user@domain): many SMTP servers
    reject a bare login name such as 'Gaia'. MAIL_FROM may also be 'Name <user@domain>'."""
    raw = os.environ.get("MAIL_FROM") or os.environ.get("SMTP_USER") or "jupyterhub"
    name, addr = parseaddr(raw)
    if "@" not in addr:
        domain = (
            os.environ.get("MAIL_DOMAIN")
            or os.environ.get("ADMIN_EMAIL", "").rpartition("@")[2]
            or "localhost"
        )
        fixed = f"{addr or 'jupyterhub'}@{domain}"
        log.warning("MAIL_FROM/SMTP_USER %r is not a full address; using %s. "
                    "Set MAIL_FROM=user@domain to silence this.", raw, fixed)
        addr = fixed
    return formataddr((name, addr))


def send_mail(to, subject, body):
    host = os.environ.get("SMTP_HOST")
    if not host or not to:
        log.warning("Mail not sent (SMTP_HOST or recipient missing): %s", subject)
        return
    user = os.environ.get("SMTP_USER")
    msg = EmailMessage()
    msg["From"] = _sender()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    port = int(os.environ.get("SMTP_PORT", "587"))
    try:
        if port == 465:
            smtp = smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context())
        else:
            smtp = smtplib.SMTP(host, port, timeout=30)
            smtp.starttls(context=ssl.create_default_context())
        with smtp:
            if user:
                smtp.login(user, os.environ.get("SMTP_PASSWORD", ""))
            smtp.send_message(msg)
    except Exception:
        log.exception("Failed to send mail %r to %s", subject, to)


async def send_mail_async(to, subject, body):
    await asyncio.get_running_loop().run_in_executor(None, send_mail, to, subject, body)


def append_signup_csv(row):
    path = STATE_DIR / "signups.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if fh.tell() == 0:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(fh, fcntl.LOCK_UN)


# ------------------------------------------------------------------- handlers
class SignUpWithExtras(SignUpHandler):
    """Reads the extra fields, lets NativeAuthenticator do the real signup,
    then logs + notifies if a new account request was actually created."""

    async def post(self):
        if CAPTCHA_ENABLED:
            if self.get_body_argument("contact_url", "", strip=True):  # honeypot filled in: a bot
                log.warning("Signup honeypot triggered from %s", self.request.remote_ip)
                raise web.HTTPError(400, "Signup rejected.")
            err = captcha_error(
                self.get_body_argument("captcha_answer", ""),
                self.get_body_argument("captcha_ts", ""),
                self.get_body_argument("captcha_sig", ""),
            )
            if err:
                log.info("Signup anti-spam check failed from %s: %s", self.request.remote_ip, err)
                raise web.HTTPError(400, err)

        extras = {n: self.get_body_argument(n, "", strip=True) for n, *_ in EXTRA_FIELDS}
        for name, label, _kind, required in EXTRA_FIELDS:
            if required and not extras[name]:
                raise web.HTTPError(400, f"Please fill in: {label}")

        raw_name = self.get_body_argument("username", "", strip=False)
        uname = self.authenticator.normalize_username(raw_name)
        existed = UserInfo.find(self.db, uname) is not None

        res = super().post()
        if inspect.isawaitable(res):
            await res

        self.db.expire_all()
        info = UserInfo.find(self.db, uname)
        if info is None or existed:
            return  # signup rejected (bad password, name taken...) -> nothing to record

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "username": uname,
            "email": getattr(info, "email", "") or "",
            **extras,
        }
        try:
            append_signup_csv(row)
        except Exception:
            log.exception("Could not write signup CSV")
        
        
        lines = [f"{k}: {v}" for k, v in row.items()]
        hub_url = os.environ.get("HUB_PUBLIC_URL", "").rstrip("/")   # no trailing slash
        body = ("New signup request:\n\n" + "\n".join(lines)
                + f"\n\nApprove it at {hub_url}/hub/authorize\n")
        subject = f"[JupyterHub {hub_url}] Signup request: {uname}"
        await send_mail_async(os.environ.get("ADMIN_EMAIL"), subject, body)


class ApproveWithMail(ToggleAuthorizationHandler):
    """Sends a mail to the user when the admin switches them to authorized."""

    async def get(self, slug):
        before = UserInfo.find(self.db, slug)
        was_authorized = bool(before and before.is_authorized)

        res = super().get(slug)
        if inspect.isawaitable(res):
            await res

        self.db.expire_all()
        after = UserInfo.find(self.db, slug)
        if after and after.is_authorized and not was_authorized:
            url = os.environ.get("HUB_PUBLIC_URL", "")
            body = (
                f"Hello {after.username},\n\n"
                "Your account request has been approved. You can now log in"
                + (f" at:\n{url}\n" if url else ".\n")
            )
            await send_mail_async(getattr(after, "email", ""), "Your JupyterHub account was approved", body)


# ------------------------------------------------------------- load indicator
ACTIVE_WINDOW = 900  # seconds: a user counts as "active" if seen in the last 15 min
_SNAP = {"t": 0.0, "data": None}
_CPU = {"prev": None, "pct": None}
_IO = {"prev": None, "rate": None}
_SAMPLER = {"started": False, "cb": None}

# Banner markup: a small item at the end of the navbar. The nav_bar_left_items block
# is the one NativeAuthenticator itself extends (for "Authorize Users"), so it is
# known to be rendered. Errors are shown in the item instead of hiding it.
BANNER_HTML = """{% if user %}
{% if user.admin %}<li class="nav-item"><a class="nav-link" href="{{ base_url }}stats/">Usage stats</a></li>{% endif %}
<li class="nav-item" style="display:flex;align-items:center;margin-left:1.75rem"><span id="gmd-load" class="navbar-text" style="font-size:.85em;padding:0">Server load: &hellip;</span>
<script>
(function () {
  var box = document.getElementById("gmd-load");
  var url = "{{ base_url }}load";  // in hub templates base_url already ends in "hub/"
  function update() {
    fetch(url, {credentials: "same-origin", headers: {"Accept": "application/json"}})
      .then(function (r) {
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        return r.json();
      })
      .then(function (d) {
        var m = d.mem;
        var dot = (m === null) ? "" : (m >= 85 ? "\\uD83D\\uDD34 " : (m >= 60 ? "\\uD83D\\uDFE0 " : "\\uD83D\\uDFE2 "));
        var t = dot + "Server load: " + d.active + " active user" + (d.active == 1 ? "" : "s") +
                " (" + d.servers + " running)";
        if (m !== null) { t += " \\u00B7 RAM " + Math.round(m) + "%"; }
        if (d.cpu !== null) { t += " \\u00B7 CPU " + Math.round(d.cpu) + "%"; }
        if (d.disk_read !== null) {
          var f = function (x) { return x >= 10 ? Math.round(x) : x.toFixed(1); };
          t += " \\u00B7 Disk R " + f(d.disk_read) + " / W " + f(d.disk_write) + " MB/s";
          box.title = "Disk throughput, " + (d.disk_src === "host" ? "whole machine" : "this container");
        }
        box.textContent = t;
      })
      .catch(function (e) { box.textContent = "Server load: unavailable (" + e.message + ")"; });
  }
  update();
  setInterval(update, 30000);
})();
</script></li>
{% endif %}
"""

def _inject_block(text, name, html):
    """Add `html` to block `name` of a Jinja child template: before the block's endblock if
    the template already defines it, otherwise as a new block that keeps the parent's content."""
    start = re.search(r"\{%-?\s*block\s+" + re.escape(name) + r"\s*-?%\}", text)
    if start:
        end = re.search(r"\{%-?\s*endblock(?:\s+\w+)?\s*-?%\}", text[start.end():])
        if end:
            pos = start.end() + end.start()
            return text[:pos] + html + text[pos:]
        log.warning("Could not find the end of block %s in NativeAuthenticator's page.html; skipped", name)
        return text
    return text.rstrip() + "\n{% block " + name + " %}\n{{ super() }}\n" + html + "{% endblock %}\n"


def _style_html():
    """CSS for the login/signup pages (logo size, compact layout) and the logged-in logo."""
    out = ""
    if LOGO_HEIGHT_ANON:
        # scales with the window height: 14vh, between the minimum and LOGO_HEIGHT_ANON
        h = "clamp(%dpx,14vh,%dpx)" % (min(LOGO_MIN_HEIGHT, LOGO_HEIGHT_ANON), LOGO_HEIGHT_ANON)
        out += ("{% if not user %}<style>#jupyterhub-logo img.jpy-logo{height:" + h + " !important;"
                "max-height:none !important;width:auto !important;margin:0 !important}</style>{% endif %}\n")
    # The hub's login box is normally ~80% of the window tall, which pushed the footer
    # logos below the fold on every window size: let it fit its content instead.
    out += "{% if not user %}<style>#login-main{height:auto !important;min-height:0 !important}</style>{% endif %}\n"
    if LOGO_HEIGHT_USER:
        out += ("{% if user %}<style>#jupyterhub-logo img.jpy-logo{height:" + str(int(LOGO_HEIGHT_USER)) +
                "px !important;max-height:none !important;width:auto !important;margin:0 !important}"
                "</style>{% endif %}\n")
    return out


def _footer_html():
    """Row of institution logos (served by FooterLogoHandler), or "" if none exist."""
    esc = htmlmod.escape
    items = []
    for fname, alt, link in FOOTER_LOGOS:
        if not (LOGO_DIR / fname).is_file():
            log.warning("Footer logo %s not found in %s; skipped", fname, LOGO_DIR)
            continue
        img = (f'<img src="{{{{ base_url }}}}footer-logos/{quote(fname)}" alt="{esc(alt, quote=True)}" '
               f'style="height:clamp({min(int(FOOTER_LOGO_MIN_HEIGHT), int(FOOTER_LOGO_HEIGHT))}px,12vh,'
               f'{int(FOOTER_LOGO_HEIGHT)}px);width:auto;max-width:100%">')
        items.append(f'<a href="{esc(link, quote=True)}" target="_blank" rel="noopener">{img}</a>' if link else img)
    if not items:
        return ""
    html = ('<div class="gmd-footer" style="display:flex;flex-wrap:wrap;justify-content:center;'
            'align-items:center;gap:1.5rem 2rem;margin:1.5rem 1rem 1rem">' + "".join(items) + "</div>\n")
    return html if FOOTER_ALSO_WHEN_LOGGED_IN else "{% if not user %}" + html + "{% endif %}\n"


def _banner_page(native_dir):
    """Build our page.html from NativeAuthenticator's OWN page.html plus our additions.

    NativeAuthenticator's page.html is what adds the 'Authorize Users' and
    'Change Password' menu entries. A page.html of ours placed earlier in
    template_paths shadows it, so we must start from its content, not from the
    hub's default page.html."""
    native = Path(native_dir) / "page.html"
    if native.exists():
        text = native.read_text()
    else:
        log.warning("NativeAuthenticator has no page.html; menu entries come from elsewhere")
        text = '{% extends "templates/page.html" %}\n'
    text = _inject_block(text, "nav_bar_left_items", BANNER_HTML)  # load indicator
    style = _style_html()
    if style:
        text = _inject_block(text, "stylesheet", style)            # logo size
    footer = _footer_html()
    if footer:
        text = _inject_block(text, "footer", footer)               # institution logos
    return text


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _mem_percent():
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


def _cpu_percent():
    """Container CPU use in % of its CPU allowance, averaged since the last call."""
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
    now = time.monotonic()
    prev, _CPU["prev"] = _CPU["prev"], (now, usage)
    if prev is not None and now - prev[0] >= 1:
        _CPU["pct"] = max(0.0, 100.0 * (usage - prev[1]) / ((now - prev[0]) * 1e6 * ncpu))
    return _CPU["pct"]


def _io_counters():
    """Cumulative (read_bytes, write_bytes, source). Prefers this container's cgroup
    io.stat (needs the cgroup 'io' controller to be enabled for it); otherwise falls
    back to the whole machine's /proc/diskstats. Page-cache hits are not counted:
    this is real device traffic."""
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


def _disk_rates():
    """(read B/s, write B/s, source) averaged since the previous sample."""
    cur = _io_counters()
    if cur is None:
        return None
    r, w, src = cur
    now = time.monotonic()
    prev, _IO["prev"] = _IO["prev"], (now, r, w, src)
    if prev is not None and prev[3] == src and now - prev[0] >= 1:
        dt = now - prev[0]
        _IO["rate"] = (max(0.0, (r - prev[1]) / dt), max(0.0, (w - prev[2]) / dt), src)
    return _IO["rate"]


def _sample():
    """Update the CPU and disk rates. Runs every 5 s in the hub's event loop, so the
    figures are recent even if nobody has polled for a while."""
    try:
        _cpu_percent()
        _disk_rates()
    except Exception:
        log.exception("load sampler failed")


def _ensure_sampler():
    if _SAMPLER["started"]:
        return
    _SAMPLER["started"] = True
    _sample()  # baseline
    try:
        from tornado.ioloop import PeriodicCallback
        _SAMPLER["cb"] = PeriodicCallback(_sample, 5000)
        _SAMPLER["cb"].start()
    except Exception:
        log.exception("Could not start the load sampler; rates update only when polled")


def _age_seconds(dt):
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        now = now.replace(tzinfo=None)
    return (now - dt).total_seconds()


def load_snapshot(users):
    """Counts only - never user names. Cached for 5 s so polling stays cheap."""
    now = time.monotonic()
    if _SNAP["data"] is not None and now - _SNAP["t"] < 5:
        return _SNAP["data"]
    servers = active = 0
    for user in list(users.values()):
        try:
            if not any(s.ready for s in user.spawners.values()):
                continue
            servers += 1
            last = user.last_activity
            if last is not None and _age_seconds(last) < ACTIVE_WINDOW:
                active += 1
        except Exception:
            log.exception("load_snapshot: could not inspect a user")
    _ensure_sampler()
    if _SAMPLER["cb"] is None:  # no event-loop sampler: sample on demand instead
        _sample()
    mem, cpu, io = _mem_percent(), _CPU["pct"], _IO["rate"]
    data = {
        "active": active,
        "servers": servers,
        "mem": None if mem is None else round(mem, 1),
        "cpu": None if cpu is None else round(cpu, 1),
        "disk_read": None if io is None else round(io[0] / 1e6, 1),
        "disk_write": None if io is None else round(io[1] / 1e6, 1),
        "disk_src": None if io is None else io[2],
        "window_min": ACTIVE_WINDOW // 60,
    }
    _SNAP.update(t=now, data=data)
    return data


class LoadHandler(BaseHandler):
    """GET /hub/load -> JSON with aggregate load numbers (logged-in users only)."""

    @web.authenticated
    async def get(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Cache-Control", "no-store")
        self.finish(json.dumps(load_snapshot(self.users)))


class CaptchaHandler(BaseHandler):
    """GET /hub/captcha -> a fresh anti-spam question (public: used by the signup form)."""

    async def get(self):
        self.set_header("Content-Type", "application/json")
        self.set_header("Cache-Control", "no-store")
        self.finish(json.dumps(new_captcha()))


class FooterLogoHandler(BaseHandler):
    """GET /hub/footer-logos/<file> -> image from LOGO_DIR. Public: the login page needs it."""

    _EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

    async def get(self, name):
        path = LOGO_DIR / name  # the route pattern excludes "/", so no directory traversal
        if path.suffix.lower() not in self._EXT or not path.is_file():
            raise web.HTTPError(404)
        self.set_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.set_header("Cache-Control", "public, max-age=3600")
        self.finish(path.read_bytes())


class StatsHandler(BaseHandler):
    """GET /hub/stats/[file] -> pages and plots written by stats_service.py. Admins only."""

    _FILES = {"": "index.html", "index.html": "index.html", "usage-day.png": "usage-day.png",
              "usage-week.png": "usage-week.png", "usage-month.png": "usage-month.png"}

    @web.authenticated
    async def get(self, name=""):
        user = self.current_user
        if not (user and user.admin):
            raise web.HTTPError(403)
        fname = self._FILES.get(name)  # whitelist: nothing else in the directory is reachable
        path = STATS_DIR / fname if fname else None
        if path is None or not path.is_file():
            if fname is None:
                raise web.HTTPError(404)
            self.set_status(404)
            self.set_header("Content-Type", "text/plain; charset=utf-8")
            return self.finish("No statistics yet. The collector writes its first page about a "
                               "minute after the hub starts (check that the 'stats' service is configured).")
        self.set_header("Content-Type", "text/html; charset=utf-8" if fname.endswith(".html")
                        else "image/png")
        self.set_header("Cache-Control", "no-store")
        self.finish(path.read_bytes())


class RedirectToStats(BaseHandler):
    @web.authenticated
    async def get(self):
        self.redirect(self.hub.base_url + "stats/")


class MyNativeAuthenticator(NativeAuthenticator):
    def get_handlers(self, app):
        out = []
        for pattern, handler in super().get_handlers(app):
            if handler is SignUpHandler:
                handler = SignUpWithExtras
            elif handler is ToggleAuthorizationHandler:
                handler = ApproveWithMail
            out.append((pattern, handler))
        out.append((r"/load", LoadHandler))
        out.append((r"/footer-logos/([^/]+)", FooterLogoHandler))
        out.append((r"/captcha", CaptchaHandler))
        out.append((r"/stats", RedirectToStats))
        out.append((r"/stats/([^/]*)", StatsHandler))
        return out


# ------------------------------------------------------- persistent Linux UIDs
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,31}$")
# Names that must never become hub users (system accounts, shared folder, ...)
RESERVED_NAMES = {"root", "shared", "nobody", "daemon", "admin", "jupyterhub"}
INIT_HOME = "/opt/jupyterhub/bin/init_home.sh"


def _allocate_uid(username):
    """Return the persistent UID for username, recording it in uids.json."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "uids.json"
    uid_min = int(os.environ.get("UID_MIN", "10000"))
    with open(STATE_DIR / "uids.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = json.loads(path.read_text()) if path.exists() else {}
        if username not in table:
            used = set(table.values()) | {p.pw_uid for p in pwd.getpwall()}
            uid = None
            # Migration: keep the UID of an existing user / home directory.
            try:
                uid = pwd.getpwnam(username).pw_uid
            except KeyError:
                home = Path("/home") / username
                if home.is_dir():
                    uid = home.stat().st_uid
            if uid is None or uid < 1000 or uid in set(table.values()):
                uid = uid_min
                while uid in used:
                    uid += 1
            table[username] = uid
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(table, indent=1, sort_keys=True))
            os.replace(tmp, path)
        return table[username]


def ensure_system_user(username):
    if not _NAME_RE.match(username):
        raise ValueError(f"Refusing to create system user for odd name {username!r}")
    if username.lower() in RESERVED_NAMES:
        raise ValueError(f"Reserved user name {username!r}")
    try:
        existing = pwd.getpwnam(username)
    except KeyError:
        existing = None
    if existing is not None:
        # Never run a user server as a pre-existing system account (root, daemon, ...)
        if existing.pw_uid < 1000:
            raise ValueError(f"{username!r} is a system account; refusing to spawn")
        _allocate_uid(username)  # make sure its UID is recorded
        return False
    uid = _allocate_uid(username)
    try:
        grp.getgrnam(username)
    except KeyError:
        subprocess.run(["groupadd", "-g", str(uid), username], check=False)
    gid_args = ["-g", username] if _group_exists(username) else ["-U"]
    subprocess.run(
        ["useradd", "-m", "-s", "/bin/bash", "-u", str(uid), *gid_args, username], check=True
    )
    home = Path("/home") / username
    if home.is_dir() and home.stat().st_uid != uid:
        subprocess.run(["chown", "-R", f"{uid}:{pwd.getpwnam(username).pw_gid}", str(home)], check=False)
    return True


def _group_exists(name):
    try:
        grp.getgrnam(name)
        return True
    except KeyError:
        return False


def pre_spawn_hook(spawner):
    name = spawner.user.name
    ensure_system_user(name)
    # One-time home initialisation (symlinks, welcome file...), run AS the user
    marker = Path("/home") / name / ".gmd_init_done"
    if not marker.exists() and os.path.exists(INIT_HOME):
        r = subprocess.run(["runuser", "-u", name, "--", INIT_HOME, name], check=False)
        if r.returncode == 0:
            subprocess.run(["runuser", "-u", name, "--", "touch", str(marker)], check=False)
        else:
            log.error("init_home.sh failed for %s (rc=%s)", name, r.returncode)
