# MagicDuck

Tools for the fast exploration and analysis of huge data tables (above one billion records), mainly aimed at ESA's Gaia space astrometry mission.


## Important notice on data rights

If you prepare a paper, presentation, outreach material, etc. using this package on data from public repositories, please follow their respective Data Rights or License terms.
In the specific case of the Gaia catalogue, please note the [Gaia Data License](https://www.cosmos.esa.int/web/gaia-users/license).
In most cases it should suffice to just add *"Credit: ESA, Gaia DPAC"* in the Acknowledgements.

You should please also acknowledge the use of this *MagicDuck* package.


## Description

This project relies on [DuckDB](https://duckdb.org/), [JupyterHub](https://jupyter.org/hub), [Python](https://www.python.org/) and [Docker](https://www.docker.com/).
It also makes use of the [Datashader](https://datashader.org/) Python package.

This project is composed of the following main elements:

* Scripts for the ingestion of data into DuckDB.
* Dockerfile and scripts for the deployment of a container with JupyterHub.
* A Python package, `GaiaMagicDuck.py`, with some tools to streamline queries to DuckDB, and especially to help in creating nice plots.


## Scripts

We provide some example scripts mainly for the Gaia DR3 bulk catalogue, provided as CSV.gz files in the official [Gaia DR3 bulk download site](https://cdn.gea.esac.esa.int/Gaia/gdr3/).

The GaiaSource-to-DuckDB ingestion script should be fully reliable, whereas the others (esp. for Epoch Photometry) may have some limitation (e.g. for the arrays of Booleans).
You may also be interested in the CSV-to-Parquet script.


## Docker for Jupyter

Under `docker4jupyter` you can find the basic setup needed to deploy a JupyterHub:

* The DockerFile to create a container with JupyterHub and some Python packages (including DuckDB).
* The `jupyterhub_config.py` configuration file required by JupyterHub. Please revise this (esp. the last lines) and update as needed.
* The `entrypoint.sh` script invoked when the container starts.
* The main `run.sh` script which builds the Docker container and starts it.
* You will also find there a copy of the GaiaMagicDuck.py package, which is the copied into the shared folder within the container.
