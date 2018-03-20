#!/bin/bash

#
# MAKE THE DATA FOLDER
#

DST=datasets
mkdir -p $DST
cd $DST

#
# DOWNLOAD DATA
#

git clone https://github.com/hpatches/hpatches-benchmark
cd hpatches-benchmark
bash download.sh hpatches
cd ..

#
# GET THE TRAINING SUBSET (version `a`)
#

mkdir -p hpatches-trn
python -c "#
import json
import os
import shutil
FOLDERS  = json.load(open('hpatches-benchmark/tasks/splits/splits.json'))['a']['train']
HPATCHES = 'hpatches-benchmark/data/hpatches-release'
DESTINAT = 'hpatches-trn'
for root, dirs, files in os.walk(HPATCHES):
	for f in files:
		if f.endswith('.png') and root.split('/')[-1] in FOLDERS:
			src = os.path.join(root, f)
			dst = os.path.join(DESTINAT, root.split('/')[-1] + '.' +  f)
			shutil.copyfile(src, dst)"

# resize to 32x32 patches
mogrify -resize 32x hpatches-trn/*.png