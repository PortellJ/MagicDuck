# Download base image ubuntu 24.04
FROM ubuntu:24.04

# Image information
LABEL version="0.1"
LABEL description="JupyterHub for Gaia data"

# Install required system packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv nodejs npm 

# Install optional packages
RUN apt-get install -y vim sqlite3

# Create a virtual environment for JupyterHub
RUN python3 -m venv /opt/jupyterhub && \
    /opt/jupyterhub/bin/pip install --upgrade pip && \
    /opt/jupyterhub/bin/pip install jupyterhub jupyterhub-nativeauthenticator notebook && \
    /opt/jupyterhub/bin/pip install --upgrade jupyterhub jupyterhub-nativeauthenticator notebook && \
    npm install -g configurable-http-proxy

# Install optional libraries
RUN /opt/jupyterhub/bin/pip install matplotlib numpy duckdb pandas scipy astropy astroquery datashader && \
    /opt/jupyterhub/bin/pip install --upgrade duckdb
# From Gaia/DPAC internal communication (2025-06-03) we may actually need this
# to properly use the latest Gaia Archive:
# python -m pip install git+https://github.com/astropy/astroquery.git

# Copy the settings into the directory
COPY jupyterhub_config.py /etc/jupyterhub/
# Only root have access to the settings
RUN chmod 700 /etc/jupyterhub/
# Create directory where to store JupyterHub sensitive files
RUN mkdir /var/lib/jupyterhub/
# Copy entrypoint 
COPY entrypoint.sh /opt/jupyterhub/bin/entrypoint.sh

# Expose the default port for JupyterHub
EXPOSE 8000

# Run every time that the container starts. Change volumes permissions.
ENTRYPOINT ["/bin/sh", "/opt/jupyterhub/bin/entrypoint.sh"]

# Set the default command to start JupyterHub
CMD ["/opt/jupyterhub/bin/jupyterhub", "-f", "/etc/jupyterhub/jupyterhub_config.py"]
