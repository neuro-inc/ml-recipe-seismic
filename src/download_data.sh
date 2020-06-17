#!/bin/bash

TMP=$(mktemp -d)
[ ! -f $TMP/ml-recipe-seismic.zip ] && wget http://data.neu.ro/ml-recipe-seismic.zip -O $TMP/ml-recipe-seismic.zip
unzip $TMP/ml-recipe-seismic.zip -d $DATA_PATH