#!/bin/sh
# Gaia MagicDuck: entry point for the Docker container

# It runs as root on every container start (volumes are attached at runtime, so
# permission changes made in the Dockerfile don't apply to them).
set -e

# Users cannot access sensitive JupyterHub files
chmod 700 /var/lib/jupyterhub/
# Users cannot list /home but can still enter their own home
chmod 711 /home
# Shared folder (example notebooks etc.): readable by all, writable only by root
mkdir -p /home/shared
chmod -R go+rX /home/shared

# Note: /data is mounted read-only (see run.sh). Set its permissions once on the
# host instead, e.g.:  chmod -R a+rX /data/

# Start JupyterHub (it must stay root: it creates users and spawns servers as them)
exec "$@"
