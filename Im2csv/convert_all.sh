#!/bin/bash

ext=".png"
for csv in `ls *.csv`
do
  python csv2Im.py $csv $csv$ext
done
