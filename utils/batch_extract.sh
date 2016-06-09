#!/bin/bash

#
#
#

#
SRC=$1
DST=$2

NKPS=$3
NPIX=$4

#
mkdir -p $DST

# clean
rm $DST/*.jpg;

#
SIZE=1.5

for f in `cd $SRC; ls *.jpg`;
do
	./fast $SRC/$f $NKPS | ./extp $SRC/$f $NPIX $SIZE $DST/$f.bag.jpg
	#cat <(python surf.py $SRC/$f $NKPS) <(python orb.py $SRC/$f $NKPS) | ./extp $SRC/$f $NPIX $SIZE $DST/$f.bag.jpg
done
