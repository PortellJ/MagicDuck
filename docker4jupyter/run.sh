#!/bin/bash
cd "$(dirname "$0")" || exit 1

docker stop jupyterhub-container 2>/dev/null
docker rm jupyterhub-container 2>/dev/null
docker build --network=host -t jupyterhub:0.2 . || exit 1

# Below you should configure the max CPU cores,
# max RAM (swap should take the same value),
# and if needed, adjust the pids-limit (it's just
# to avoid fork bombs, for security).
# The port may also need to be adjusted to your case.

docker run \
   --init \
   --name jupyterhub-container \
   --cpus="4" --memory="8g" --memory-swap="8g" \
   --pids-limit=1000 --security-opt no-new-privileges \
   --env-file "$(pwd)/persistent/mail.env" \
   -v "$(pwd)/persistent/users":/home \
   -v "$(pwd)/persistent/var":/var/lib/jupyterhub \
   -v /data/Gaia/dr3/:/data/:ro \
   -p 8000:8000 \
   jupyterhub:0.2
