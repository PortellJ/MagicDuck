"""Extras for JupyterHub + NativeAuthenticator.

 1. Signup page: logo, title and extra fields (edit EXTRA_FIELDS below).
 2. On each new signup: e-mail to the admin + row appended to a CSV (no password).
 3. When the admin approves a user: e-mail to that user.
 4. Stable Linux UIDs/GIDs for users (persisted in uids.json).

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
"""
import asyncio
import csv
import fcntl
import grp
import html as htmlmod
import inspect
import json
import logging
import os
import pwd
import re
import smtplib
import ssl
import subprocess
import time
from email.message import EmailMessage
from pathlib import Path

import nativeauthenticator
from nativeauthenticator import NativeAuthenticator
from nativeauthenticator.handlers import SignUpHandler, ToggleAuthorizationHandler
from nativeauthenticator.orm import UserInfo
from tornado import web

log = logging.getLogger("JupyterHub.hub_extras")

STATE_DIR = Path(os.environ.get("HUB_STATE_DIR", "/var/lib/jupyterhub"))

# ---------------------------------------------------------------- signup form
SITE_TITLE = "MagicDuck JupyterHub"
SITE_INTRO = "You can request an account here, which will need approval by an administrator."

# (form field name, label, kind, required)   kind: "text" | "textarea"
EXTRA_FIELDS = [
    ("full_name", "Full name", "text", True),
    ("affiliation", "Institution / affiliation", "text", True),
    ("reason", "Motivation for this account", "textarea", True),
]

CSV_COLUMNS = ["timestamp", "username", "email"] + [f[0] for f in EXTRA_FIELDS]


def build_signup_template(out_dir):
    """Copy NativeAuthenticator's signup.html to out_dir with logo, title and
    extra fields injected. Returns out_dir (put it FIRST in template_paths)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(nativeauthenticator.__file__).parent / "templates" / "signup.html"
    page = src.read_text()

    esc = htmlmod.escape
    header = (
        '<div style="text-align:center;margin:1em 0">'
        '<img src="{{ base_url }}logo" alt="logo" style="max-height:120px">'
        f'<h2>{esc(SITE_TITLE)}</h2><p>{esc(SITE_INTRO)}</p></div>'
    )
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

    page, n1 = re.subn(r"<form\b", lambda m: header + m.group(0), page, count=1)
    page, n2 = re.subn(
        r"<input[^>]*type=[\"']submit[\"']", lambda m: fields + m.group(0), page, count=1
    )
    if not (n1 and n2):
        raise RuntimeError(
            "Could not patch signup.html (form/submit button not found); "
            "this NativeAuthenticator version needs a manual template."
        )
    (out_dir / "signup.html").write_text(page)
    return str(out_dir)


# ----------------------------------------------------------------------- mail
def send_mail(to, subject, body):
    host = os.environ.get("SMTP_HOST")
    if not host or not to:
        log.warning("Mail not sent (SMTP_HOST or recipient missing): %s", subject)
        return
    user = os.environ.get("SMTP_USER")
    msg = EmailMessage()
    msg["From"] = os.environ.get("MAIL_FROM") or user or "jupyterhub@localhost"
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
        body = "New signup request:\n\n" + "\n".join(lines) + "\n\nApprove it in /hub/authorize.\n"
        await send_mail_async(os.environ.get("ADMIN_EMAIL"), f"[JupyterHub] Signup request: {uname}", body)


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


class MyNativeAuthenticator(NativeAuthenticator):
    def get_handlers(self, app):
        out = []
        for pattern, handler in super().get_handlers(app):
            if handler is SignUpHandler:
                handler = SignUpWithExtras
            elif handler is ToggleAuthorizationHandler:
                handler = ApproveWithMail
            out.append((pattern, handler))
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
