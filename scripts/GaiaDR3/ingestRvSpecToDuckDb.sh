#!/bin/bash

# Simple example to ingest bulk GaiaDR3 files into DuckDB
# This one is for the RVS Mean Spectrum
# jportell@fqa.ub.edu

types=`gunzip -c rvs_mean_spectrum/RvsMeanSpectrum_000000-003111.csv.gz | grep '^#' | grep -e "  name" -e "  datatype" -e "subtype" | grep -v string | sed -e 's/null//' -e "s/'//g" | awk -v c="'" '{ if (NR%2==1) printf("%c%s%c:",c,$3,c); else printf("%c%s%c, ",c,$3,c) }' | sed -e 's/float64/double/g' -e 's/float32/float/g' -e 's/string/varchar/g' -e 's/boolean/bool/g' -e "s/'', /{/" -e "s/, $/}/" | sed 's/bool\[\]/varchar/g'`

echo "set threads=4; set memory_limit='32GB'; create table rvs_mean_spec as select * from read_csv('rvs_mean_spectrum/RvsMeanSpectrum_*.csv.gz', comment='#', nullstr='null', null_padding=true, ignore_errors=true, types={$types);" > IngestRvSpecInDdb.sql

duckdb GDR3_RvsMeanSpec.duckdb <IngestRvSpecInDdb.sql

