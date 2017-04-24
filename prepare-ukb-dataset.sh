#!/bin/bash

#
# MAKE THE DATA FOLDER
#

DST=datasets
mkdir -p $DST
cd $DST

#
# DOWNLOAD THE UKB IMAGES
#

wget https://nenadmarkus.com/data/ukb.tar -O ukb.tar
tar -xvf ukb.tar
rm ukb.tar

#
# PREPARE TRAINING AND VALIDATION PATCHES
#

NKPS=256
NPIX=32
SIZE=1.5

mkdir -p ukb-trn-patches
for f in `cd ukb-trn; ls *.jpg`;
do
	cat <(python ../utils/surf.py ukb-trn/$f $NKPS) <(python ../utils/orb.py ukb-trn/$f $NKPS) | python ../utils/warper.py ukb-trn/$f $NPIX $SIZE ukb-trn-patches/$f.bag.jpg
done

mkdir -p ukb-val-patches
for f in `cd ukb-val; ls *.jpg`;
do
	cat <(python ../utils/surf.py ukb-val/$f $NKPS) <(python ../utils/orb.py ukb-val/$f $NKPS) | python ../utils/warper.py ukb-val/$f $NPIX $SIZE ukb-val-patches/$f.bag.jpg
done

#
# PREPARE tripletgen.lua
#

cp ../utils/tripletgen.lua tripletgen-ukb.lua
sed -i -e 's/--TRN-FOLDER--/"datasets\/ukb-trn-patches"/g' tripletgen-ukb.lua
sed -i -e 's/--TRN-NCHANNELS--/3/g' tripletgen-ukb.lua
sed -i -e 's/--TRN-PROBABILITY--/0.33/g' tripletgen-ukb.lua
sed -i -e 's/--VLD-FOLDER--/"datasets\/ukb-val-patches"/g' tripletgen-ukb.lua
sed -i -e 's/--VLD-NCHANNELS--/3/g' tripletgen-ukb.lua
sed -i -e 's/--VLD-PROBABILITY--/1.0/g' tripletgen-ukb.lua