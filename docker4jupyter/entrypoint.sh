#!/bin/sh

# Mounted volumes are attached at runtime, so any permission changes made in 
# the Dockerfile don’t apply to them. Entrypoint script changes permissions  
# every time the container starts.

# Users cannot access sensetive JupyterHub files
chmod 700 /var/lib/jupyterhub/
# Users cannot see other users with ls /home but still have access to their own home
chmod 711 /home
# One exception to this is the shared user, where you can place example notebooks etc.
# (and we'll create softlinks to this from each existing user)
mkdir /home/shared
chmod go+rX -R /home/shared
cd /home
for f in `ls | grep -v shared`
do
	cd $f
	ln -s ../shared
	ln -s ../shared/GaiaMagicDuck.py
	cd ..
done
cd
# Users can read all files inside /data but not modify them
find /data -type f -exec chmod 644 {} \;  # Apply to files  
find /data -type d -exec chmod 755 {} \;  # Apply to directories  
# Start JupyterHub
exec "$@"
