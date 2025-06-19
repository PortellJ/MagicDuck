# MagicDuck

Tools for the fast exploration and analysis of huge data tables, mainly aimed at ESA's Gaia space astrometry mission.

## Description

This project relies on [DuckDB](https://duckdb.org/), [JupyterHub](https://jupyter.org/hub), [Python](https://www.python.org/) and [Docker](https://www.docker.com/).
It also makes use of the [Datashader](https://datashader.org/) Python package.

This project is composed of the following main elements:

* Scripts for the ingestion of data into DuckDB.
* Dockerfile and scripts for the deployment of a container with JupyterHub.
* A Python package, `GaiaMagicDuck.py`, with some tools to streamline queries to DuckDB, and especially to help in creating some nice plots.

