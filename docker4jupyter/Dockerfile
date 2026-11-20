# Dockerfile for the Gaia MagicDuck

# Download base image
FROM python:3.14-slim

# Image information
LABEL version="0.2"
LABEL description="Gaia MagickDuck JupyterHub"

# Install required system packages
RUN apt-get update && \
   apt-get upgrade -y && \
   apt-get install -y python3 python3-pip python3-venv nodejs npm vim sqlite3 screen && \
   rm -rf /var/lib/apt/lists/*

# Create a virtual environment for JupyterHub
RUN python3 -m venv /opt/jupyterhub && \
   /opt/jupyterhub/bin/pip install --upgrade pip && \
   /opt/jupyterhub/bin/pip install jupyterhub jupyterhub-nativeauthenticator notebook && \
   /opt/jupyterhub/bin/pip install --upgrade jupyterhub jupyterhub-nativeauthenticator notebook && \
   npm install -g configurable-http-proxy

# Install packages
RUN /opt/jupyterhub/bin/pip install matplotlib numpy duckdb pandas scipy astropy astroquery datashader healpy pyarrow ipywidgets jupyter-resource-usage && \
   /opt/jupyterhub/bin/pip install --upgrade duckdb

# Settings, logo and helper module live in the image (NOT in a mounted volume)
RUN mkdir -p /etc/jupyterhub /var/lib/jupyterhub
COPY jupyterhub_config.py hub_extras.py logo.png /etc/jupyterhub/
RUN chmod 700 /etc/jupyterhub/

# Copy entry point and additional helper scripts
COPY entrypoint.sh /opt/jupyterhub/bin/entrypoint.sh
COPY jupyterhub-singleuser-limited /opt/jupyterhub/bin/
COPY init_home.sh /opt/jupyterhub/bin/init_home.sh
RUN chmod +x /opt/jupyterhub/bin/entrypoint.sh \
             /opt/jupyterhub/bin/jupyterhub-singleuser-limited \
             /opt/jupyterhub/bin/init_home.sh

# The hub must run as root to create Linux users and to start each
# user's server under that user's UID. Isolation is provided by the container.
WORKDIR /var/lib/jupyterhub
# Expose the default port for JupyterHub
EXPOSE 8000
# Run every time that the container starts. It changes volumes permissions.
ENTRYPOINT ["/bin/sh", "/opt/jupyterhub/bin/entrypoint.sh"]
# Set the default command to start JupyterHub
CMD ["/opt/jupyterhub/bin/jupyterhub", "-f", "/etc/jupyterhub/jupyterhub_config.py"]
