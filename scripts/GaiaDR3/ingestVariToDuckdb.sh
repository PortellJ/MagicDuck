#!/bin/bash

# Simple example to ingest bulk GaiaDR3 files into DuckDB
# This one is for the RR Lyrae
# jportell@fqa.ub.edu

# Beware:
# The SQL must be manually corrected afterwards, because one of the last datatypes (classification) is a string, which we've removed before (to properly handle the subtypes).
# After this, we can invoke the ingestion as usual.
types=`gunzip -c vari_rrlyrae/VariRrlyrae_000000-003111.csv.gz | grep '^#' | grep -e "  name" -e "  datatype" -e "subtype" | grep -v string | sed -e 's/null//' -e "s/'//g" | awk -v c="'" '{ if (NR%2==1) printf("%c%s%c:",c,$3,c); else printf("%c%s%c, ",c,$3,c) }' | sed -e 's/float64/double/g' -e 's/float32/float/g' -e 's/string/varchar/g' -e 's/boolean/bool/g' -e "s/'', /{/" -e "s/, $/}/"`

echo "set threads=4; set memory_limit='32GB'; create table vari_rrlyrae as select * from read_csv('vari_rrlyrae/VariRrlyrae_*.csv.gz', comment='#', nullstr='null', null_padding=true, ignore_errors=true, types={$types);" > IngestRrlyraeInDdb.sql

# Fix the SQL, and then invoke this:
# duckdb GDR3_Vari.duckdb < IngestRrlyraeInDdb.sql

