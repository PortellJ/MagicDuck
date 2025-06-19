#!/bin/bash

# Simple example to ingest bulk GaiaDR3 files into DuckDB
# This one is for the SSO Observations
# jportell@fqa.ub.edu

types=`gunzip -c sso_observation/SsoObservation_00.csv.gz | grep '^#' | grep -e "  name" -e "  datatype" | sed -e 's/null//' -e "s/'//g" | awk -v c="'" '{ if (NR%2==1) printf("%c%s%c:",c,$3,c); else printf("%c%s%c, ",c,$3,c) }' | sed -e 's/float64/double/g' -e 's/float32/float/g' -e 's/string/varchar/g' -e 's/boolean/bool/g' -e "s/'', /{/" -e "s/, $/}/" | sed "s/'astrometric_outcome_ccd':'varchar'/'astrometric_outcome_ccd':'int[]'/"`

echo "set threads=4; set memory_limit='32GB'; create table sso_obs as select * from read_csv('sso_observation/SsoObservation_*.csv.gz', comment='#', nullstr='null', null_padding=true, ignore_errors=true, types={$types);" > IngestSsoObsInDdb.sql

duckdb GDR3_SsoObs.duckdb <IngestSsoObsInDdb.sql

