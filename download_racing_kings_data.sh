#!/bin/bash

dirloc="Dataset"
rm -rf "$dirloc"
mkdir "$dirloc"
cd "$dirloc"

loc="https://database.lichess.org/racingKings/lichess_db_racingKings_rated_20"

years=(18 19 20)
months=(01 02 03 04 05 06 07 08 09 10 11 12)

for year in ${years[@]}; do
    for month in ${months[@]}; do
        tmp_loc="$loc$year-$month.pgn.bz2"
        tmp_file="file$year$month"
        wget "$tmp_loc" -O "$tmp_file.bz2"
        bzip2 -d "$tmp_file.bz2"
        mv "$tmp_file" "$tmp_file.pgn"
        rm -rf "$tmp_file.bz2"
    done
done

year_month="21-01.pgn.bz2"
tmp_loc="$loc$year_month"
tmp_file="file2101.bz2"
wget "$tmp_loc" -O "$tmp_file"
bzip2 -d "$tmp_file"
rm -rf "$tmp_file.bz2"