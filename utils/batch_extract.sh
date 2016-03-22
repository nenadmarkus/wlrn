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
NKPS=$3
NPIX=$4

SIZE=1.5

for f in `cd $SRC; ls *.jpg`;
do
	#./fast $SRC/$f $NKPS | ./extp $SRC/$f $NPIX $SIZE $DST/$f.bag
	#cat <(python sift.py $SRC/$f $NKPS) <(./fast $SRC/$f $NKPS) | ./extp $SRC/$f $NPIX $SIZE $DST/$f.bag
	cat <(python surf.py $SRC/$f $NKPS) <(python orb.py $SRC/$f $NKPS) | ./extp $SRC/$f $NPIX $SIZE $DST/$f.bag
done

#
(cd $DST/; ls *.bag > list)

