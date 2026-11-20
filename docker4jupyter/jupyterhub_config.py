# Configuration file for jupyterhub,
# without most of the (commented) default examples,
# just keeping the necessary lines for the Gaia MagicDuck.

c = get_config()


# Load hub_extras.py and set the authenticator defined therein
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hub_extras
c.JupyterHub.authenticator_class = 'hub_extras.MyNativeAuthenticator'


# Prepare the web templates
import nativeauthenticator
c.JupyterHub.template_paths = [
    hub_extras.build_signup_template('/var/lib/jupyterhub/templates'),
    f"{os.path.dirname(nativeauthenticator.__file__)}/templates/",
]
c.JupyterHub.logo_file = '/etc/jupyterhub/logo.png'


# File in which to store the cookie secret
c.JupyterHub.cookie_secret_file = '/var/lib/jupyterhub/jupyterhub_cookie_secret'


# url for the database. e.g. `sqlite:///jupyterhub.sqlite`
c.JupyterHub.db_url = 'sqlite:////var/lib/jupyterhub/jupyterhub.sqlite'


# The command used for starting the single-user server.
# Here we use a 'limited' one to put some limits in the memory per user
c.Spawner.cmd = ['/opt/jupyterhub/bin/jupyterhub-singleuser-limited']


# A memory "limit" to be considered when showing the system load.
# It doesn't really "cut" the processes, it's not a real limit, but just
# something for the visual monitoring.
# ### TO BE CONFIGURED ###
c.Spawner.mem_limit = '8G'

# Path to the notebook directory for the single-user server
c.Spawner.notebook_dir = '~/'


# Our hook function to do some bootstrapping work before the spawner starts
c.Spawner.pre_spawn_hook = hub_extras.pre_spawn_hook


# Allow every user who can successfully authenticate to access JupyterHub
c.Authenticator.allow_all = True


# Here you must indicate a username which, when signed up and approved,
# will become the super-user (admin of the hub)
# ### TO BE CONFIGURED ###
c.NativeAuthenticator.admin_users = {'my_admin_user'}


# Configures the number of failed attempts a user can have before being blocked
c.NativeAuthenticator.allowed_failed_logins = 3


# Asks for email on signup
c.NativeAuthenticator.ask_email_on_signup = True


# Creates a verification of password strength when a new user makes signup
c.NativeAuthenticator.check_common_password = True


# Check if the length of the password is at least this size on signup
c.NativeAuthenticator.minimum_password_length = 8


# Configures the number of seconds a user has to wait after being blocked
c.NativeAuthenticator.seconds_before_next_try = 300


# Avoid weird chars in the username
c.NativeAuthenticator.username_pattern = '^[A-Za-z0-9_-]+$'


# File in which to write the PID of the proxy process.
c.ConfigurableHTTPProxy.pid_file = '/var/lib/jupyterhub/jupyterhub-proxy.pid'


# Path where your users (and you) will find the jupyterhub.
# ### TO BE CONFIGURED ###
c.JupyterHub.base_url = '/my_public_url_path'

# Accepted IPs. Limit this if you wish:
c.JupyterHub.ip = '*'

# Path of the spawner. The default should be fine
c.Spawner.default_url = '/lab'

# URL (path) and port where the jupyterhub will be found.
# ### TO BE CONFIGURED ###
c.JupyterHub.bind_url = 'https://:8000/my_public_url_path'

# Port of the hub (the internal one, to be 'captured' by Docker)
# ### TO BE CONFIGURED (but this default should be fine) ###
c.JupyterHub.port = 8000

# The local IP address (within the Docker container) of the hub
# Not sure at all that this should be really set; leaving it commented
# works perfectly fine.
#c.Spawner.ip = 'my.local.ip.address'

# The local IP address (within the Docker container) of the spawner
# Not sure at all that this should be really set; leaving it commented
# works perfectly fine.
#c.LocalProcessSpawner.ip = 'my.local.ip.address'
#c.SimpleLocalProcessSpawner.ip = 'my.local.ip.address'
