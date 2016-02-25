#!/bin/bash

#
#
#

#
SRC=$1
DST=$2

#
mkdir -p $DST

# clean
(cd $DST/; rm *.bag; rm list;)

#
NKPS=75
NPIX=32
SIZE=1.5

for f in `cd $SRC; ls *.jpg`;
do
	./fast $SRC/$f $NKPS | ./extp $SRC/$f $NPIX $SIZE $DST/$f.bag
done

#
(cd $DST/; ls *.bag > list)

