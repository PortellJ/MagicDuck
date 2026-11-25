# MagicDuck

Tools for the fast exploration and analysis of huge data tables (above one billion records), mainly aimed at ESA's Gaia space astrometry mission, but also applicable to other cases.


## Important notice on data rights

If you prepare a paper, presentation, outreach material, etc. using this package on data from public repositories, please follow their respective Data Rights or License terms.
In the specific case of the Gaia catalogue, please note the [Gaia Data License](https://www.cosmos.esa.int/web/gaia-users/license).
In most cases it should suffice to just add *"Credit: ESA, Gaia DPAC"* in the Acknowledgements.

You should please also acknowledge the use of this *MagicDuck* package.


## Description

This project relies on [DuckDB](https://duckdb.org/), [JupyterHub](https://jupyter.org/hub), [Python](https://www.python.org/) and [Docker](https://www.docker.com/).
It also makes use of the [Datashader](https://datashader.org/) Python package.

This project is composed of the following main elements:

* Dockerfile, JupyterHub configurations and scripts for the deployment of a container with JupyterHub.
* A Python package, `GaiaMagicDuck.py`, with some tools to streamline queries to DuckDB, and especially to help in creating nice plots.
* Scripts for the ingestion of data into DuckDB.


## Docker for Jupyter

Under `docker4jupyter` you will find all the setup needed to deploy a containerized JupyterHub:

* The `DockerFile` to create a container image with JupyterHub and some Python packages (including DuckDB).
* The `jupyterhub_config.py` configuration file required by JupyterHub. Please revise this (esp. the lines highlighted with `TO BE CONFIGURED`) and update as needed.
* A `hub_extras` file for additional functionalities, where you may optionally configure some variables. This was mostly created by Claude.
* A `stats_service.py` file for monitoring functionalities, allowing to see how the JupyterHub service is being used.
* A `jupyterhub-singleuser-limited` file which enforces the kernel to kill just the user server/kernel in case of out-of-memory, which should allow the Hub and other users to continue.
* An `entrypoint.sh` script, invoked when the container starts. It just sets some permissions and copies shared files.
* An `init_home.sh` script, invoked when a new user logs in for the first time, linking data and shared files to its home folder.
* A `mail.env` file under the `persistent` folder, where you may setup your mail account if you wish to generate automatic notifications of signups and approvals.
* The main `run.sh` script, which builds the Docker container and starts it. You should revise this to set your CPU cores and memory limits (for the overall jupyterhub container), the mapped port to acces it, and the folder (in the host) with the data files to be accessed by this service.
* You will also find there the `GaiaMagicDuck.py` package, which is copied into the shared folder within the container once its image is built.

To deploy the service, you just need to configure the elements indicated above, and finally, simply launch `run.sh`.

Once you have the JupyterHub running, you may notice some interesting features:

* The signup requires some additional fields such as the full name, institution and motivation. It should help you to identify new requests with legitimate interests. A basic captcha is also included.
* If the mail properly configured, new signups will send you a notification mail to approve them. And when approved, the users will also receive a notification mail.
* Both in the signup and in the login pages, some logo(s) of your institution are shown (which you can configure). We include some relevant ones in this project, under the `logos` folder (where you should place yours).
* The first time that a user logs in, the basic files (e.g., the Python package and some example notebooks) are copied into the users's home folder.
* Once in your JupyterHub (either as admin or user), in the Hub control panel (File > Hub Control Panel) you can see the system load: active users, RAM, CPU and disk.
* In the Hub control panel, but only as admin, you will also see a Usage Stats link, which will bring you to a simple page with the usage statistics of the service.
* In the Lab page (the usual page where you work), you can also see the memory consumed in the status bar at the bottom.
* In one of the persisted volumes (`/var`), you will find a CSV file with the signup requests. There is also a `uids.json` file, which is used by the system to ensure that each user gets the same UID in the container after rebuilds and restarts. These are stored there and required by the service to function properly.


## Scripts

We provide some example scripts mainly for the Gaia DR3 bulk catalogue, provided as CSV.gz files in the official [Gaia DR3 bulk download site](https://cdn.gea.esac.esa.int/Gaia/gdr3/).
We will soon provide scripts for Gaia DR4.

The GaiaSource-to-DuckDB ingestion script should be fully reliable, whereas the others (esp. for Epoch Photometry) may have some limitation (e.g. for the arrays of Booleans).
You may also be interested in the CSV-to-Parquet script.
