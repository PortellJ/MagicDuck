#!/bin/bash
# Initialize a new user's home directory. Called by hub_extras.pre_spawn_hook,
# running AS that user, once (marker file ~/.gmd_init_done).

username="$1"
home="/home/$username"

# echo "Welcome to the Gaia MagicDuck!" > "$home/WELCOME.txt"

ln -sfn /data "$home/data"
ln -sfn /home/shared "$home/shared"
ln -sfn /home/shared/GaiaMagicDuck.py "$home/GaiaMagicDuck.py"
