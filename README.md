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


## Scripts

We provide some example scripts mainly for the Gaia DR3 bulk catalogue, provided as CSV.gz files in the official [Gaia DR3 bulk download site](https://cdn.gea.esac.esa.int/Gaia/gdr3/).
We will soon provide scripts for Gaia DR4.

The GaiaSource-to-DuckDB ingestion script should be fully reliable, whereas the others (esp. for Epoch Photometry) may have some limitation (e.g. for the arrays of Booleans).
You may also be interested in the CSV-to-Parquet script.


## Docker for Jupyter

Under `docker4jupyter` you can find the basic setup needed to deploy a JupyterHub:

* The `DockerFile` to create a container with JupyterHub and some Python packages (including DuckDB).
* The `jupyterhub_config.py` configuration file required by JupyterHub. Please revise this (esp. the lines highlighted with `TO BE CONFIGURED`) and update as needed.
* A `hub_extras` file for additional functionalities, where you may optionally configure some variables. This was mostly created by Claude.
* A `jupyterhub-singleuser-limited` file which enforces the kernel to kill just the user server/kernel in case of out-of-memory, which should allow the Hub and other users to continue.
* The `entrypoint.sh` script, invoked when the container starts.
* An `init_home.sh` script, invoked when a new user logs in for the first time.
* A `mail.env` file under the `persistent` folder, where you may setup your mail account if you wish to generate automatic notifications of signups and approvals.
* The main `run.sh` script which builds the Docker container and starts it. You should revise this to set your CPU cores and memory limits (for the overall jupyterhub container), the mapped port to acces it, and the file with the Gaia data.
* You will also find there a copy of the GaiaMagicDuck.py package, which is copied into the shared folder within the container.

To deploy the service, you just need to configure the elements indicated above, and finally, simply launch `run.sh`.

Once you have the JupyterHub running, you may notice some interesting features:

* The signup requires some additional fields such as the full name, institution and motivation. It should help you to identify new requests with legitimate interests. A basic captcha is also included.
* If the mail properly configured, new signups will send you a notification mail to approve them. And when approved, the users will also receive a notification mail.
* Both in the signup and in the login pages, you can include the logo(s) of your institution.
* The first time that a user logs in, the basic files (e.g., the Python package and some example notebooks) can be copied into the users's home folder.
* Once in your JupyterHub (either as admin or user), in the Hub control panel (File > Hub Control Panel) you can see the system load: active users, RAM, CPU and disk.
* In the Lab page (the usual page where you work), you can also see the memory consumed at the bottom.
