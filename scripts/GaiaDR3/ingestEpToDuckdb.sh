#!/bin/bash

# Simple example to ingest bulk GaiaDR3 files into DuckDB
# This one is for the EpochPhotometry
# jportell@fqa.ub.edu

# Disclaimer: the bool[] arrays don't seem to be correctly detected by DuckDB, so we just ingest them as varchar... which means that they cannot be properly handled by the array/list functions :-(
# (a possible, but slow, solution would be to previosly process these files and use 'sed' to transform ""True"" into true, and ""False"" into false. With this, we've seen (in e.g. SSO Residuals data) that it works fine.

types=`gunzip -c epoch_photometry/EpochPhotometry_000000-003111.csv.gz | grep '^#' | grep -e "  name" -e "  datatype" -e "subtype" | grep -v string | sed -e 's/null//' -e "s/'//g" | awk -v c="'" '{ if (NR%2==1) printf("%c%s%c:",c,$3,c); else printf("%c%s%c, ",c,$3,c) }' | sed -e 's/float64/double/g' -e 's/float32/float/g' -e 's/string/varchar/g' -e 's/boolean/bool/g' -e "s/'', /{/" -e "s/, $/}/" | sed 's/bool\[\]/varchar/g'`

echo "set threads=4; set memory_limit='32GB'; create table epoch_phot as select * from read_csv('epoch_photometry/EpochPhotometry_*.csv.gz', comment='#', nullstr='null', null_padding=true, ignore_errors=true, types={$types);" > IngestEpInDdb.sql

duckdb GDR3_EpochPhot.duckdb <IngestEpInDdb.sql

