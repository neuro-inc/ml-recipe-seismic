#!/bin/bash

DATA_PATH=${DATA_PATH:-/data}
TMP=$(mktemp -d)
[ ! -f $TMP/ml-recipe-seismic.zip ] && wget http://data.neu.ro/ml-recipe-seismic.zip -O $TMP/ml-recipe-seismic.zip
unzip -o $TMP/ml-recipe-seismic.zip -d $DATA_PATH
