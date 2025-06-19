#!/bin/bash

# Simple example to convert the bulk GaiaDR3 GaiaSource files to Parquet files
# jportell@fqa.ub.edu

INFOLDER="gaia_source"
OUTFOLDER="gs_parquet"

# We take one random file and get its first 1000 lines (which are -exactly!- the ECSV header with the data types etc.)
types=`gunzip -c $INFOLDER/GaiaSource_232786-232828.csv.gz | head -n 1000 | grep -e name -e datatype | awk -v c="'" '{
	if (NR%2 == 0) {
		printf("%c%s%c:",c,$3,c)
	} else {
		printf("%c%s%c, ",c,$3,c)
	}
}' | sed -e 's/float64/double/g' -e 's/float32/float/g' -e 's/string/varchar/g' -e 's/boolean/bool/g' -e "s/'', /{/" -e "s/, $/}/"`
# (not sure if I treat all the possible cases - perhaps some other tweaks are needed)

mkdir $OUTFOLDER

echo "Processing files (PID $$)..."

# Process all files
for f in `ls $INFOLDER/GaiaSource_*.csv.gz`
do
	outf=`basename $f | cut -d '.' -f 1`
	echo "Doing "$f"..."
	echo "set threads=4; set memory_limit='8GB'; COPY(select * from read_csv('"$f"', nullstr='null', null_padding=true, ignore_errors=true, types=$types)) TO '"$OUTFOLDER"/"$outf".parquet'" | ~/duckdb
done

