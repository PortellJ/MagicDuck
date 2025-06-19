#!/bin/bash

# Simple example to ingest the bulk GaiaDR3 GaiaSource files into DuckDB
# jportell@fqa.ub.edu

INFOLDER="gaia_source"

# We take one random file and get its first 1000 lines (which are -exactly!- the ECSV header with the data types etc.)
types=`gunzip -c $INFOLDER/GaiaSource_000000-003111.csv.gz | head -n 1000 | grep -e name -e datatype | awk -v c="'" '{
	if (NR%2 == 0) {
		printf("%c%s%c:",c,$3,c)
	} else {
		printf("%c%s%c, ",c,$3,c)
	}
}' | sed -e 's/float64/double/g' -e 's/float32/float/g' -e 's/string/varchar/g' -e 's/boolean/bool/g' -e "s/'', /{/" -e "s/, $/}/"`

echo "set threads=16; set memory_limit='32GB'; create table gaia_source as select * from read_csv('"$INFOLDER"/GaiaSource*.csv.gz', nullstr='null', null_padding=true, ignore_errors=true, types=$types);" > IngestInDdb.sql
# You may wish to stop here and double-check that the SQL looks fine (data types etc.)

duckdb GDR3_GaiaSource.duckdb <IngestInDdb.sql

