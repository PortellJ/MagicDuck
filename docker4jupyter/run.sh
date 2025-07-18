#!/bin/bash
docker stop jupyterhub-container
docker rm jupyterhub-container
docker build --network=host -t jupyterhub:0.1 .
docker run \
    --cpus="8" --memory="16g" \
    -v ./persistent/users:/home \
    -v ./persistent/var:/var/lib/jupyterhub \
    -v /data/Gaia/dr3/:/data/ \
    -p 8000:8000 \
    --name jupyterhub-container jupyterhub:0.1
