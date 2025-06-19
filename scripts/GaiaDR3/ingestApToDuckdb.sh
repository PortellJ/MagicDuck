#!/bin/bash

# Simple example to ingest bulk GaiaDR3 files into DuckDB
# This one is for the AstrophysicalParameters
# jportell@fqa.ub.edu

types=`gunzip -c astrophysical_parameters/AstrophysicalParameters_000000-003111.csv.gz | grep '^#' | grep -e "  name" -e "  datatype" | sed -e 's/null//' -e "s/'//g" | awk -v c="'" '{ if (NR%2==1) printf("%c%s%c:",c,$3,c); else printf("%c%s%c, ",c,$3,c) }' | sed -e 's/float64/double/g' -e 's/float32/float/g' -e 's/string/varchar/g' -e 's/boolean/bool/g' -e "s/'', /{/" -e "s/, $/}/"`

echo "set threads=4; set memory_limit='32GB'; create table astroph_params as select * from read_csv('astrophysical_parameters/AstrophysicalParameters_*.csv.gz', comment='#', nullstr='null', null_padding=true, ignore_errors=true, types={$types);" > IngestApInDdb.sql

duckdb GDR3_AstrophParams.duckdb <IngestApInDdb.sql

