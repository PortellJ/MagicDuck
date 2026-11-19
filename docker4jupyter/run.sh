#!/bin/bash
cd "$(dirname "$0")" || exit 1

docker stop jupyterhub-container 2>/dev/null
docker rm jupyterhub-container 2>/dev/null
docker build --network=host -t jupyterhub:0.2 . || exit 1

# persistent/mail.env (chmod 600) holds SMTP_HOST, SMTP_PORT, SMTP_USER,
# SMTP_PASSWORD, MAIL_FROM, ADMIN_EMAIL, HUB_PUBLIC_URL (one VAR=value per line).
docker run \
   --init \
   --cpus="4" --memory="8g" \
   --name jupyterhub-container \
   --env-file "$(pwd)/persistent/mail.env" \
   -v "$(pwd)/persistent/users":/home \
   -v "$(pwd)/persistent/var":/var/lib/jupyterhub \
   -v /data/Gaia/dr3/:/data/:ro \
   -p 8000:8000 \
   jupyterhub:0.2
