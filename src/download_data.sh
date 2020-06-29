#!/bin/bash

DATA_PATH=${DATA_PATH:-/storage_path}
TMP=$(mktemp -d)
[ ! -f $TMP/ml-recipe-seismic.zip ] && wget http://data.neu.ro/ml-recipe-seismic.zip -O $TMP/ml-recipe-seismic.zip
unzip -q -o $TMP/ml-recipe-seismic.zip -d $DATA_PATH
