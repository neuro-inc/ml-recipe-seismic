#!/bin/bash

DEST=/.
TMP=$(mktemp -d)

wget http://data.neu.ro/ml-recipe-seismic.zip -O $TMP/ml-recipe-seismic.zip
unzip $TMP/ml-recipe-seismic.zip -d $DEST

