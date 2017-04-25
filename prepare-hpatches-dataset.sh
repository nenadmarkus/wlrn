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

#
# PREPARE tripletgen.lua
#

cp ../utils/tripletgen.lua tripletgen-hpatches.lua
sed -i -e 's/--TRN-FOLDER--/"datasets\/hpatches-trn"/g' tripletgen-hpatches.lua
sed -i -e 's/--TRN-PROBABILITY--/0.7/g' tripletgen-hpatches.lua
sed -i -e 's/--VLD-FOLDER--/"datasets\/hpatches-trn"/g' tripletgen-hpatches.lua
sed -i -e 's/--VLD-PROBABILITY--/0.7/g' tripletgen-hpatches.lua

cp ../utils/tripletgen.py tripletgen-hpatches.py
sed -i -e 's/--TRN-FOLDER--/"datasets\/hpatches-trn"/g' tripletgen-hpatches.py
sed -i -e 's/--TRN-PROBABILITY--/0.7/g' tripletgen-hpatches.py
sed -i -e 's/--VLD-FOLDER--/"datasets\/hpatches-trn"/g' tripletgen-hpatches.py
sed -i -e 's/--VLD-PROBABILITY--/0.7/g' tripletgen-hpatches.py

#
#
#

#th models/descnet256.lua models/descnet256.t7
#th wlrn.lua models/descnet256.t7 datasets/tripletgen-hpatches.lua -w models/descnet256.t7